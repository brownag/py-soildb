"""
Response handling for SDA query results with proper data type conversion.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from .exceptions import SDAResponseError
from .spc_presets import ColumnConfig
from .spc_validator import (
    SPCColumnValidator,
    SPCDepthValidator,
    SPCValidationError,
    SPCWarnings,
    create_spc_validation_report,
)
from .type_conversion import get_default_type_map

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore
    from soilprofilecollection import SoilProfileCollection

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Unified validator for different SDA response types with consolidated logic."""

    # Standard column sets for different response types
    MAPUNIT_REQUIRED_COLUMNS = ["mukey", "musym", "muname"]
    PEDON_REQUIRED_COLUMNS = ["pedon_id", "site_id"]
    COORDINATE_COLUMNS = ["latitude", "longitude", "x", "y"]

    @staticmethod
    def _validate_base(response: "SDAResponse") -> "ValidationResult":
        """Perform base validation common to all response types.

        Checks: empty responses, column structure, metadata, data integrity, types.

        Args:
            response: SDAResponse object to validate

        Returns:
            ValidationResult with detailed validation information
        """
        import time

        start_time = time.time()
        result = ValidationResult()

        # Track metadata
        result.metadata["total_rows"] = len(response._data)
        result.metadata["total_columns"] = len(response._columns)
        result.metadata["response_size"] = len(str(response._raw_data))
        result.metadata["validation_timestamp"] = time.time()

        # Check for empty response
        if response.is_empty():
            result.add_warning("Response contains no data rows")
            result.metadata["is_empty"] = True
            result.add_transformation("empty_response_detected")
        else:
            result.metadata["is_empty"] = False

        # Validate column structure
        if not response._columns:
            result.add_error("No column names found in response")
            result.add_transformation("missing_columns_detected")
        else:
            # Check for duplicate column names
            if len(response._columns) != len(set(response._columns)):
                duplicates = [
                    col for col in response._columns if response._columns.count(col) > 1
                ]
                result.add_warning(f"Duplicate column names found: {duplicates}")
                result.add_transformation("duplicate_columns_detected")

            # Check for missing metadata
            if len(response._metadata) != len(response._columns):
                result.add_warning(
                    f"Metadata count ({len(response._metadata)}) doesn't match column count ({len(response._columns)})"
                )
                result.add_transformation("metadata_mismatch_detected")

        # Validate data integrity
        if response._data:
            # Check row lengths
            expected_length = len(response._columns)
            inconsistent_rows = []
            null_value_count = 0
            total_values = 0

            for i, row in enumerate(response._data):
                total_values += len(row)
                null_value_count += sum(
                    1
                    for val in row
                    if val is None or str(val).lower() in ["", "null", "none"]
                )

                if len(row) != expected_length:
                    inconsistent_rows.append(i)

            if inconsistent_rows:
                result.add_warning(
                    f"Rows with inconsistent lengths: {inconsistent_rows[:5]}{'...' if len(inconsistent_rows) > 5 else ''}"
                )
                result.add_transformation("inconsistent_row_lengths_detected")

            # Calculate data completeness
            completeness = (
                1.0 - (null_value_count / total_values) if total_values > 0 else 0.0
            )
            result.metadata["data_completeness"] = completeness

            if completeness < 0.8:
                result.add_error(f"Data completeness too low: {completeness:.1%}")
                result.add_transformation("low_data_completeness_detected")
            elif completeness < 0.95:
                result.add_warning(f"Data completeness moderate: {completeness:.1%}")
                result.add_transformation("moderate_data_completeness_detected")

        # Validate data types
        column_types = response.get_column_types()
        unknown_types = 0
        for col_name, sda_type in column_types.items():
            if sda_type.lower() not in SDAResponse.SDA_TYPE_MAPPING:
                result.add_warning(
                    f"Unknown SDA data type '{sda_type}' for column '{col_name}'"
                )
                unknown_types += 1

        if unknown_types > 0:
            result.add_transformation("unknown_data_types_detected")

        # Update processing statistics
        end_time = time.time()
        result.update_processing_stats(
            "validation_duration_seconds", end_time - start_time
        )
        result.update_processing_stats("columns_validated", len(response._columns))
        result.update_processing_stats("rows_validated", len(response._data))
        result.update_processing_stats("data_types_validated", len(column_types))
        result.update_processing_stats(
            "transformations_applied", len(result.transformations_applied)
        )

        return result

    @staticmethod
    def _validate_schema(
        response: "SDAResponse",
        required_columns: List[str],
        response_type: str,
        optional_columns: Optional[List[str]] = None,
    ) -> "ValidationResult":
        """Validate response schema against required columns.

        Consolidated logic for mapunit, pedon, and other domain-specific validations.

        Args:
            response: SDAResponse object to validate
            required_columns: List of required column names
            response_type: Type of response (e.g., "mapunit", "pedon")
            optional_columns: List of optional column names to check for

        Returns:
            ValidationResult with schema validation details
        """
        result = ValidationResult()
        result.metadata["response_type"] = response_type

        if not hasattr(response, "columns") or not hasattr(response, "data"):
            return result

        # Check required columns
        missing_cols = [col for col in required_columns if col not in response.columns]
        if missing_cols:
            result.add_error(f"Missing required {response_type} columns: {missing_cols}")
            result.add_transformation(f"missing_required_columns_detected_{response_type}")

        # Check optional columns
        if optional_columns:
            missing_optional = [
                col for col in optional_columns if col not in response.columns
            ]
            if missing_optional:
                result.add_warning(
                    f"Optional {response_type} columns not found: {missing_optional}"
                )
                result.add_transformation(f"missing_optional_columns_detected_{response_type}")

        # Validate data in required columns
        if required_columns and response.data:
            for req_col in required_columns:
                if req_col in response.columns:
                    col_idx = response.columns.index(req_col)
                    empty_count = sum(
                        1
                        for row in response.data
                        if not row
                        or str(row[col_idx]).strip() == ""
                        or str(row[col_idx]).lower() == "null"
                    )
                    if empty_count > 0:
                        result.add_warning(
                            f"{empty_count} {response_type} records have empty '{req_col}' values"
                        )

        return result

    @staticmethod
    def validate_general(response: "SDAResponse") -> "ValidationResult":
        """Validate a general SDA response for common issues.

        Args:
            response: SDAResponse object to validate

        Returns:
            ValidationResult with comprehensive validation details
        """
        return ResponseValidator._validate_base(response)

    @staticmethod
    def validate_mapunit(response: "SDAResponse") -> "ValidationResult":
        """Validate a mapunit response for required fields and data integrity.

        Args:
            response: SDAResponse object to validate

        Returns:
            ValidationResult with mapunit-specific validation details
        """
        # Start with base validation
        base_result = ResponseValidator._validate_base(response)

        # Add mapunit-specific schema validation
        schema_result = ResponseValidator._validate_schema(
            response,
            required_columns=ResponseValidator.MAPUNIT_REQUIRED_COLUMNS,
            response_type="mapunit",
        )

        # Merge results
        base_result.errors.extend(schema_result.errors)
        base_result.warnings.extend(schema_result.warnings)
        base_result.metadata.update(schema_result.metadata)
        base_result.transformations_applied.extend(schema_result.transformations_applied)

        # Adjust quality score for schema-specific issues
        base_result.data_quality_score = max(
            0.0,
            base_result.data_quality_score
            - 0.2 * len(schema_result.errors)
            - 0.05 * len(schema_result.warnings),
        )

        return base_result

    @staticmethod
    def validate_pedon(response: "SDAResponse") -> "ValidationResult":
        """Validate a pedon response for required fields and data integrity.

        Args:
            response: SDAResponse object to validate

        Returns:
            ValidationResult with pedon-specific validation details
        """
        # Start with base validation
        base_result = ResponseValidator._validate_base(response)

        # Add pedon-specific schema validation
        schema_result = ResponseValidator._validate_schema(
            response,
            required_columns=ResponseValidator.PEDON_REQUIRED_COLUMNS,
            response_type="pedon",
            optional_columns=ResponseValidator.COORDINATE_COLUMNS,
        )

        # Merge results
        base_result.errors.extend(schema_result.errors)
        base_result.warnings.extend(schema_result.warnings)
        base_result.metadata.update(schema_result.metadata)
        base_result.transformations_applied.extend(schema_result.transformations_applied)

        # Adjust quality score for schema-specific issues
        base_result.data_quality_score = max(
            0.0,
            base_result.data_quality_score
            - 0.2 * len(schema_result.errors)
            - 0.05 * len(schema_result.warnings),
        )

        return base_result

    @staticmethod
    def validate_type_system(
        response: "SDAResponse", data_dicts: List[Dict[str, Any]]
    ) -> "ValidationResult":
        """Validate data types in transformed data.

        Checks schema consistency, type violations, and range violations.

        Args:
            response: SDAResponse object
            data_dicts: List of dictionaries from to_dict() conversion

        Returns:
            ValidationResult with type system validation details
        """
        result = ValidationResult()
        result.metadata["validation_type"] = "transformed_data"
        result.metadata["record_count"] = len(data_dicts)

        if not data_dicts:
            result.add_warning("No data to validate")
            return result

        # Check for consistent schema across all records
        first_record = data_dicts[0]
        expected_keys = set(first_record.keys())
        schema_inconsistencies = 0

        for i, record in enumerate(data_dicts[1:], 1):
            record_keys = set(record.keys())
            if record_keys != expected_keys:
                schema_inconsistencies += 1
                if schema_inconsistencies <= 5:  # Limit logging
                    missing = expected_keys - record_keys
                    extra = record_keys - expected_keys
                    logger.warning(
                        f"Schema inconsistency in record {i}: missing={missing}, extra={extra}"
                    )

        if schema_inconsistencies > 0:
            result.add_warning(
                f"{schema_inconsistencies} records have inconsistent schemas"
            )

        # Validate data types and ranges for known columns
        column_types = response.get_column_types()
        type_violations = {}
        range_violations = {}

        for col_name, sda_type in column_types.items():
            if col_name not in expected_keys:
                continue

            sda_type_lower = sda_type.lower()
            violations = 0
            range_issues = 0

            for record in data_dicts:
                value = record.get(col_name)

                # Type validation
                if value is not None:
                    if sda_type_lower in [
                        "int",
                        "integer",
                        "bigint",
                        "smallint",
                        "tinyint",
                    ]:
                        if not isinstance(value, int):
                            violations += 1
                    elif sda_type_lower in [
                        "float",
                        "real",
                        "double",
                        "decimal",
                        "numeric",
                    ]:
                        if not isinstance(value, (int, float)):
                            violations += 1
                    elif sda_type_lower == "bit":
                        if not isinstance(value, bool):
                            violations += 1

                    # Range validation for known columns
                    if col_name.lower() in ["latitude", "lat"]:
                        if isinstance(value, (int, float)) and not (-90 <= value <= 90):
                            range_issues += 1
                    elif col_name.lower() in ["longitude", "lon", "lng"]:
                        if isinstance(value, (int, float)) and not (
                            -180 <= value <= 180
                        ):
                            range_issues += 1
                    elif col_name.lower().endswith("_depth") or col_name.lower() in [
                        "hzdept_r",
                        "hzdepb_r",
                    ]:
                        if isinstance(value, (int, float)) and value < 0:
                            range_issues += 1

            if violations > 0:
                type_violations[col_name] = violations
            if range_issues > 0:
                range_violations[col_name] = range_issues

        if type_violations:
            result.add_warning(f"Type violations found: {type_violations}")
        if range_violations:
            result.add_warning(f"Range violations found: {range_violations}")

        # Calculate overall data quality
        total_violations = sum(type_violations.values()) + sum(
            range_violations.values()
        )
        total_values = len(data_dicts) * len(expected_keys)

        if total_values > 0:
            violation_rate = total_violations / total_values
            if violation_rate > 0.1:  # More than 10% violations
                result.add_error(f"Data violation rate too high: {violation_rate:.1%}")
            elif violation_rate > 0.05:  # More than 5% violations
                result.add_warning(
                    f"Data violation rate moderate: {violation_rate:.1%}"
                )

        return result


@dataclass
class ValidationResult:
    """Result of response validation with warnings and errors."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 1.0  # 1.0 = perfect, 0.0 = unusable
    transformations_applied: List[str] = field(default_factory=list)
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.data_quality_score = max(0.0, self.data_quality_score - 0.2)

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
        self.data_quality_score = max(0.0, self.data_quality_score - 0.05)

    def add_transformation(self, transformation: str) -> None:
        """Add a record of transformation applied."""
        self.transformations_applied.append(transformation)

    def update_processing_stats(self, key: str, value: Any) -> None:
        """Update processing statistics."""
        self.processing_stats[key] = value

    def is_valid(self) -> bool:
        """Check if the response is valid (no errors)."""
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def __str__(self) -> str:
        """String representation of validation result."""
        parts = []
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        if self.transformations_applied:
            parts.append(f"Transformations: {len(self.transformations_applied)}")
        parts.append(f"Quality Score: {self.data_quality_score:.2f}")
        return f"ValidationResult({', '.join(parts)})"


class SDAResponse:
    """Represents a response from the Soil Data Access web service."""

    # SDA data type mapping to Python/pandas/polars types
    SDA_TYPE_MAPPING = {
        # Numeric types
        "int": "int64",
        "integer": "int64",
        "bigint": "int64",
        "smallint": "int32",
        "tinyint": "int16",
        "bit": "bool",
        # Floating point types
        "float": "float64",
        "real": "float32",
        "double": "float64",
        "decimal": "float64",
        "numeric": "float64",
        "money": "float64",
        "smallmoney": "float64",
        # String types
        "varchar": "string",
        "nvarchar": "string",
        "char": "string",
        "nchar": "string",
        "text": "string",
        "ntext": "string",
        # Date/time types
        "datetime": "datetime64[ns]",
        "datetime2": "datetime64[ns]",
        "smalldatetime": "datetime64[ns]",
        "date": "datetime64[ns]",
        "time": "string",  # Keep as string for time-only values
        "timestamp": "datetime64[ns]",
        # Spatial/binary types
        "geometry": "string",  # Keep WKT as string
        "geography": "string",
        "varbinary": "string",
        "binary": "string",
        "image": "string",
        # Other types
        "uniqueidentifier": "string",
        "xml": "string",
    }

    def __init__(self, raw_data: Dict[str, Any]):
        """Initialize from SDA JSON response."""
        self._raw_data = raw_data
        self._validation_result: Optional[ValidationResult] = None
        self._transformations_applied: List[str] = []
        self._processing_stats: Dict[str, Any] = {}
        self._columns: List[str] = []
        self._metadata: List[str] = []
        self._data: List[List[Any]] = []
        self._parse_response()

    def _parse_response(self) -> None:
        """Parse the SDA response format with enhanced error handling."""
        try:
            if "Table" not in self._raw_data:
                # Handle empty responses (no results) - SDA returns {} for empty result sets
                if self._raw_data == {}:
                    logger.debug("Received empty SDA response (no results)")
                    # Empty result set
                    self._columns = []
                    self._metadata = []
                    self._data = []
                    return
                else:
                    available_keys = list(self._raw_data.keys())
                    raise SDAResponseError(
                        f"Invalid SDA response format: missing 'Table' key. Available keys: {available_keys}"
                    )

            table_data = self._raw_data["Table"]

            if not isinstance(table_data, list):
                raise SDAResponseError(
                    f"Invalid SDA response format: 'Table' value is {type(table_data).__name__}, expected list"
                )

            if len(table_data) < 2:
                logger.warning(
                    f"SDA response has incomplete table structure: {len(table_data)} rows"
                )
                # Try to recover partial data
                if len(table_data) >= 1:
                    self._columns = (
                        table_data[0] if isinstance(table_data[0], list) else []
                    )
                    self._metadata = []
                    self._data = []
                    logger.info(
                        f"Recovered column names from partial response: {self._columns}"
                    )
                else:
                    raise SDAResponseError(
                        f"Invalid SDA response format: Table data has {len(table_data)} rows, minimum 2 required"
                    )
                return

            # First row contains column names
            self._columns = table_data[0] if table_data else []
            if not self._columns:
                logger.warning("SDA response has no column names")

            # Second row contains column metadata (data types, etc.)
            self._metadata = table_data[1] if len(table_data) > 1 else []
            if len(self._metadata) != len(self._columns):
                logger.warning(
                    f"Metadata count ({len(self._metadata)}) doesn't match column count ({len(self._columns)})"
                )

            # Remaining rows contain actual data
            self._data = table_data[2:] if len(table_data) > 2 else []

            # Log parsing summary
            logger.debug(
                f"Successfully parsed SDA response: {len(self._columns)} columns, {len(self._data)} rows"
            )

        except Exception as e:
            logger.error(f"Failed to parse SDA response: {e}", exc_info=True)
            raise

    @classmethod
    def from_json(cls, json_str: str) -> "SDAResponse":
        """Create response from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(data)
        except json.JSONDecodeError as e:
            raise SDAResponseError(f"Failed to parse JSON response: {e}") from e

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return self._columns

    @property
    def data(self) -> List[List[Any]]:
        """Get raw data rows."""
        return self._data

    @property
    def metadata(self) -> List[str]:
        """Get column metadata."""
        return self._metadata

    def __len__(self) -> int:
        """Return number of data rows."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Check if the response contains any data rows."""
        return len(self._data) == 0

    def __iter__(self) -> "Iterator[List[Any]]":
        """Iterate over data rows."""
        return iter(self._data)

    def validate(self, response_type: str = "general") -> ValidationResult:
        """Validate the response based on its type.

        Unified validation entry point that routes to appropriate validators.

        Args:
            response_type: Type of response to validate:
                - "general": General SDA response validation
                - "mapunit": Mapunit-specific validation
                - "pedon": Pedon-specific validation

        Returns:
            ValidationResult with comprehensive validation details

        Examples:
            >>> response = SDAResponse(data)
            >>> result = response.validate("mapunit")
            >>> if result.is_valid():
            ...     print(f"Quality score: {result.data_quality_score}")
        """
        validator_map = {
            "general": ResponseValidator.validate_general,
            "mapunit": ResponseValidator.validate_mapunit,
            "pedon": ResponseValidator.validate_pedon,
        }

        validator = validator_map.get(
            response_type.lower(), ResponseValidator.validate_general
        )
        result = validator(self)
        self._validation_result = result
        return result

    def validate_response(self) -> ValidationResult:
        """Validate the response for common issues and return detailed results.

        DEPRECATED: Use validate() instead for better API.
        This method is maintained for backward compatibility.

        Examples:
            >>> response = SDAResponse(data)
            >>> result = response.validate_response()  # Deprecated
            >>> result = response.validate()  # Preferred
        """
        return self.validate("general")

    @staticmethod
    def validate_mapunit_response(response_data: Dict[str, Any]) -> ValidationResult:
        """Validate a mapunit response for required fields and data integrity.

        DEPRECATED: Use response.validate("mapunit") instead.
        This static method is maintained for backward compatibility.

        Args:
            response_data: SDAResponse object or dict with response data

        Returns:
            ValidationResult with mapunit-specific validation details

        Examples:
            >>> result = SDAResponse.validate_mapunit_response(response)  # Deprecated
            >>> result = response.validate("mapunit")  # Preferred
        """
        if isinstance(response_data, SDAResponse):
            return ResponseValidator.validate_mapunit(response_data)

        # Fallback for dict input (backward compatibility)
        result = ValidationResult()
        result.metadata["response_type"] = "mapunit"

        # Check if it's an SDAResponse-compatible dict
        if isinstance(response_data, SDAResponse):
            base_validation = response_data.validate()
            result.errors.extend(base_validation.errors)
            result.warnings.extend(base_validation.warnings)
            result.metadata.update(base_validation.metadata)

        # Mapunit-specific validations
        required_columns = ["mukey", "musym", "muname"]
        if hasattr(response_data, "columns"):
            missing_cols = [
                col for col in required_columns if col not in response_data.columns
            ]
            if missing_cols:
                result.add_error(f"Missing required mapunit columns: {missing_cols}")

            # Check for data in key columns
            if (
                hasattr(response_data, "data")
                and response_data.data
                and "mukey" in response_data.columns
            ):
                empty_mukeys = sum(
                    1
                    for row in response_data.data
                    if not row
                    or str(row[response_data.columns.index("mukey")]).strip() == ""
                )
                if empty_mukeys > 0:
                    result.add_warning(
                        f"{empty_mukeys} mapunits have empty mukey values"
                    )

        return result

    @staticmethod
    def validate_pedon_response(response_data: Dict[str, Any]) -> ValidationResult:
        """Validate a pedon response for required fields and data integrity.

        DEPRECATED: Use response.validate("pedon") instead.
        This static method is maintained for backward compatibility.

        Args:
            response_data: SDAResponse object or dict with response data

        Returns:
            ValidationResult with pedon-specific validation details

        Examples:
            >>> result = SDAResponse.validate_pedon_response(response)  # Deprecated
            >>> result = response.validate("pedon")  # Preferred
        """
        if isinstance(response_data, SDAResponse):
            return ResponseValidator.validate_pedon(response_data)

        # Fallback for dict input (backward compatibility)
        result = ValidationResult()
        result.metadata["response_type"] = "pedon"

        # Check if it's an SDAResponse-compatible dict
        if isinstance(response_data, SDAResponse):
            base_validation = response_data.validate()
            result.errors.extend(base_validation.errors)
            result.warnings.extend(base_validation.warnings)
            result.metadata.update(base_validation.metadata)

        # Pedon-specific validations
        required_columns = ["pedon_id", "site_id"]
        if hasattr(response_data, "columns"):
            missing_cols = [
                col for col in required_columns if col not in response_data.columns
            ]
            if missing_cols:
                result.add_error(f"Missing required pedon columns: {missing_cols}")

            # Check for coordinate columns
            coord_cols = ["latitude", "longitude", "x", "y"]
            has_coords = any(col in response_data.columns for col in coord_cols)
            if not has_coords:
                result.add_warning("No coordinate columns found in pedon data")

        return result

    @staticmethod
    def handle_missing_fields(
        data: Dict[str, Any],
        required_fields: List[str],
        fallback_values: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Handle missing required fields with fallbacks or defaults.

        Args:
            data: The data dictionary to check
            required_fields: List of field names that must be present
            fallback_values: Optional mapping of field names to fallback values

        Returns:
            Tuple of (processed_data, missing_fields_list)
        """
        processed_data = data.copy()
        missing_fields = []
        fallback_values = fallback_values or {}

        for required_field in required_fields:
            if (
                required_field not in data
                or data[required_field] is None
                or str(data[required_field]).strip() == ""
            ):
                if required_field in fallback_values:
                    processed_data[required_field] = fallback_values[required_field]
                    logger.warning(
                        f"Using fallback value for missing field '{required_field}': {fallback_values[required_field]}"
                    )
                else:
                    missing_fields.append(required_field)
                    logger.error(
                        f"Required field '{required_field}' is missing and no fallback provided"
                    )

        return processed_data, missing_fields

    def validate_transformed_data(
        self, data_dicts: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate transformed data dictionaries for consistency and data quality.

        Args:
            data_dicts: List of dictionaries from to_dict() conversion

        Returns:
            ValidationResult with data quality assessment
        """
        return ResponseValidator.validate_type_system(self, data_dicts)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries with basic type conversion and error recovery.

        Returns:
            List of dictionaries where each dictionary represents a row with
            column names as keys and converted values as values.
        """
        if not self._columns:
            logger.warning("Attempting to convert response with no columns")
            return []

        # Get column types for basic conversion
        column_types = self.get_column_types()

        result = []
        conversion_errors = 0
        max_errors = min(10, len(self._data))  # Limit error logging

        for row_idx, row in enumerate(self._data):
            try:
                # Pad row with None if it's shorter than columns
                padded_row = row + [None] * (len(self._columns) - len(row))

                # Convert values based on inferred types
                converted_row = {}
                for _col_idx, (col_name, value) in enumerate(
                    zip(self._columns, padded_row[: len(self._columns)])
                ):
                    try:
                        sda_type = column_types.get(col_name, "varchar").lower()
                        converted_value = self._convert_value(value, sda_type)
                        converted_row[col_name] = converted_value
                    except Exception as e:
                        # Log conversion error but continue with original value
                        if conversion_errors < max_errors:
                            logger.warning(
                                f"Failed to convert value in row {row_idx}, column '{col_name}' (type: {sda_type}): {value} -> {e}"
                            )
                        conversion_errors += 1
                        converted_row[col_name] = value  # Keep original value

                result.append(converted_row)

            except Exception as e:
                logger.error(
                    f"Failed to process row {row_idx}: {e}",
                    exc_info=conversion_errors < max_errors,
                )
                conversion_errors += 1
                # Skip malformed rows but continue processing
                continue

        if conversion_errors > 0:
            logger.warning(
                f"Encountered {conversion_errors} conversion errors during processing"
            )

        # Validate transformed data if requested
        if len(result) > 0:
            data_validation = self.validate_transformed_data(result)
            if not data_validation.is_valid():
                logger.warning(f"Data validation found issues: {data_validation}")
            # Store validation result
            if not hasattr(self, "_data_validation_result"):
                self._data_validation_result = data_validation

        return result

    def to_records(self) -> List[Dict[str, Any]]:
        """Alias for to_dict() for compatibility."""
        return self.to_dict()

    def _convert_value(self, value: Any, sda_type: str) -> Any:
        """Convert a single value based on SDA data type using unified TypeMap.

        This delegates to the unified type conversion system which consolidates
        logic from response.py, type_processors.py, and schema_system.py.
        """
        type_map = get_default_type_map()
        return type_map.convert_value(value, sda_type, strict=False)

    def _get_pandas_dtype_mapping(self) -> Dict[str, str]:
        """Get pandas-compatible dtype mapping using unified TypeMap."""
        column_types = self.get_column_types()
        dtype_mapping = {}
        type_map = get_default_type_map()

        for col_name, sda_type in column_types.items():
            dtype_mapping[col_name] = type_map.get_pandas_dtype(sda_type)

        return dtype_mapping

    def _get_polars_dtype_mapping(self) -> Dict[str, Any]:
        """Get polars-compatible dtype mapping using unified TypeMap."""
        try:
            import polars  # noqa: F401
        except ImportError:
            return {}

        column_types = self.get_column_types()
        dtype_mapping = {}
        type_map = get_default_type_map()

        for col_name, sda_type in column_types.items():
            dtype_mapping[col_name] = type_map.get_polars_dtype(sda_type)

        return dtype_mapping

    def to_dataframe(self, library: str = "pandas", convert_types: bool = True) -> Any:
        """Convert to pandas or polars DataFrame with proper type conversion."""
        data_dict = self.to_dict()

        if library.lower() == "pandas":
            try:
                import pandas as pd

                df = pd.DataFrame(data_dict)

                if convert_types and not df.empty:
                    # Apply dtype conversion
                    dtype_mapping = self._get_pandas_dtype_mapping()

                    for col_name, dtype in dtype_mapping.items():
                        if col_name in df.columns:
                            try:
                                if dtype == "datetime64[ns]":
                                    df[col_name] = pd.to_datetime(
                                        df[col_name], errors="coerce"
                                    )
                                elif dtype in ["Int64", "Int32"]:
                                    df[col_name] = pd.to_numeric(
                                        df[col_name], errors="coerce"
                                    ).astype(dtype)
                                elif dtype == "boolean":
                                    df[col_name] = df[col_name].astype("boolean")
                                elif dtype == "float64":
                                    df[col_name] = pd.to_numeric(
                                        df[col_name], errors="coerce"
                                    )
                                else:
                                    df[col_name] = df[col_name].astype(dtype)
                            except (ValueError, TypeError):
                                # If conversion fails, keep as object/string
                                continue

                return df
            except ImportError:
                raise ImportError(
                    "pandas is required for DataFrame conversion. Install with: pip install pandas"
                ) from None

        elif library.lower() == "polars":
            try:
                import polars as pl

                df = pl.DataFrame(data_dict)

                if convert_types and not df.is_empty():
                    # Apply dtype conversion
                    dtype_mapping = self._get_polars_dtype_mapping()

                    for col_name, dtype in dtype_mapping.items():
                        if col_name in df.columns:
                            try:
                                if dtype == pl.Datetime:
                                    df = df.with_columns(
                                        pl.col(col_name).str.strptime(
                                            pl.Datetime, format=None, strict=False
                                        )
                                    )
                                else:
                                    df = df.with_columns(
                                        pl.col(col_name).cast(dtype, strict=False)  # type: ignore
                                    )
                            except Exception:
                                # If conversion fails, keep original type
                                continue

                return df
            except ImportError:
                raise ImportError(
                    "polars is required for DataFrame conversion. Install with: pip install polars"
                ) from None

        else:
            raise ValueError(
                f"Unsupported library: {library}. Choose 'pandas' or 'polars'."
            )

    def to_pandas(self, convert_types: bool = True) -> Any:
        """Convert to pandas DataFrame with proper type conversion."""
        return self.to_dataframe("pandas", convert_types=convert_types)

    def to_polars(self, convert_types: bool = True) -> Any:
        """Convert to polars DataFrame with proper type conversion."""
        return self.to_dataframe("polars", convert_types=convert_types)

    def to_geodataframe(self, convert_types: bool = True) -> Any:
        """Convert to GeoPandas GeoDataFrame if geometry column exists."""
        try:
            import geopandas as gpd
            from shapely import wkt
        except ImportError:
            raise ImportError(
                "geopandas and shapely are required for GeoDataFrame conversion. Install with: pip install geopandas shapely"
            ) from None

        df = self.to_pandas(convert_types=convert_types)

        if df.empty:
            # Return empty GeoDataFrame with appropriate schema
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

        # Look for geometry column (case-insensitive)
        geometry_col = None
        for col in df.columns:
            if col.lower() in ["geometry", "geom", "shape", "wkt"]:
                geometry_col = col
                break

        if geometry_col is None:
            raise ValueError(
                "No geometry column found. Expected column name containing 'geometry', 'geom', 'shape', or 'wkt'"
            )

        # Convert WKT strings to shapely geometries
        def wkt_to_geom(wkt_str: Any) -> Any:
            if wkt_str is None or wkt_str == "" or str(wkt_str).lower() == "null":
                return None
            try:
                return wkt.loads(str(wkt_str))
            except Exception:
                return None

        df["geometry"] = df[geometry_col].apply(wkt_to_geom)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Remove invalid geometries
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

        return gdf

    def to_soilprofilecollection(
        self,
        site_data: "pd.DataFrame | None" = None,
        site_id_col: Optional[str] = None,
        hz_id_col: Optional[str] = None,
        hz_top_col: Optional[str] = None,
        hz_bot_col: Optional[str] = None,
        preset: Union[str, ColumnConfig, None] = None,
        validate_depths: bool = True,
        warn_on_defaults: bool = True,
    ) -> "SoilProfileCollection":
        """
        Converts the response data to a soilprofilecollection.SoilProfileCollection object.

        This method is intended for horizon-level data, which can be joined with
        site-level data (e.g., from the component or pedon table) to form a complete
        soil profile collection.

        **Decision Tree: When to use to_soilprofilecollection()**

        1. Do you have horizon-level data with depths?
           - Yes → Continue
           - No → Use `to_pandas()` or `to_polars()` instead

        2. What type of data do you have?
           - Standard SDA chorizon → Use `preset="standard_sda"`
           - Lab pedon data → Use `preset="lab_pedon"`
           - Map unit + component → Use `preset="mapunit_component"`
           - Custom columns → Use `preset=CustomColumnConfig(...)`
           - Explicit column names → Provide site_id_col, hz_id_col, etc.

        3. Do you have site-level data?
           - Yes → Provide as `site_data` parameter
           - No → Leave as None (SPC will only have horizon data)

        Args:
            site_data: Optional pandas DataFrame containing site-level data.
                This will be joined with the horizon data using site_id_col.

            site_id_col: Column name for site/component identifier.
                If not provided and preset is None, defaults to "cokey".
                Can be: "cokey" (component), "pedon_id" (lab), "mukey" (mapunit)

            hz_id_col: Column name for unique horizon identifier.
                If not provided and preset is None, defaults to "chkey".
                Can be: "chkey" (SDA), "pedon_key_horizon" (lab), "hzname" (other)

            hz_top_col: Column name for horizon top depth.
                If not provided and preset is None, defaults to "hzdept_r".
                Must be in inches for standard SDA data.

            hz_bot_col: Column name for horizon bottom depth.
                If not provided and preset is None, defaults to "hzdepb_r".
                Must be in inches for standard SDA data.

            preset: Preset configuration. Can be:
                - String name: "standard_sda", "lab_pedon", "pedon_site", "mapunit_component"
                - ColumnConfig object: e.g., StandardSDAHorizonColumns()
                - None: Use explicit column parameters (defaults to standard SDA)

            validate_depths: Whether to validate depth values before conversion.
                If True and invalid depths found, issues warning but continues.
                Default: True

            warn_on_defaults: Whether to warn when using default column names.
                Set to False if you prefer silent defaults.
                Default: True

        Returns:
            A SoilProfileCollection object configured with the provided columns.

        Raises:
            ImportError: If the 'soilprofilecollection' package is not installed.
            SPCValidationError: If required columns are missing and cannot be resolved.

        Examples:
            >>> from soildb.spc_presets import StandardSDAHorizonColumns
            >>> from soildb import SDAClient, Query
            >>>
            >>> # Using preset
            >>> async with SDAClient() as client:
            ...     query = Query().select(
            ...         "cokey", "chkey", "hzname", "hzdept_r", "hzdepb_r",
            ...         "claytotal_r", "sandtotal_r", "silttotal_r", "awc_r", "ksat_r"
            ...     ).from_("chorizon")
            ...     response = await client.execute(query)
            ...     spc = response.to_soilprofilecollection(preset="standard_sda")

            >>> # Using explicit columns
            >>> spc = response.to_soilprofilecollection(
            ...     site_id_col="cokey",
            ...     hz_id_col="chkey",
            ...     hz_top_col="hzdept_r",
            ...     hz_bot_col="hzdepb_r"
            ... )

            >>> # With site data
            >>> import pandas as pd
            >>> site_data = pd.DataFrame({
            ...     "cokey": [1, 2, 3],
            ...     "compname": ["Ames", "Benton", "Clarion"]
            ... })
            >>> spc = response.to_soilprofilecollection(
            ...     site_data=site_data,
            ...     preset="standard_sda"
            ... )

        See Also:
            spc_presets: Column configuration presets for common query types
            spc_validator: Validation utilities used internally
        """
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            raise ImportError(
                "The 'soilprofilecollection' package is required to use "
                "to_soilprofilecollection(). Please install it with: "
                "pip install soildb[soil]"
            ) from None


        # Step 1: Resolve column configuration
        if preset is not None:
            if isinstance(preset, str):
                from .spc_presets import get_preset
                config = get_preset(preset)
            elif isinstance(preset, ColumnConfig):
                config = preset
            else:
                raise TypeError(
                    f"preset must be string, ColumnConfig, or None; got {type(preset)}"
                )

            site_id_col = site_id_col or config.site_id_col
            hz_id_col = hz_id_col or config.horizon_id_col
            hz_top_col = hz_top_col or config.horizon_top_col
            hz_bot_col = hz_bot_col or config.horizon_bottom_col
        else:
            # Use provided columns or defaults
            site_id_col = site_id_col or "cokey"
            hz_id_col = hz_id_col or "chkey"
            hz_top_col = hz_top_col or "hzdept_r"
            hz_bot_col = hz_bot_col or "hzdepb_r"

            if warn_on_defaults:
                SPCWarnings.warn_default_columns()

        # Step 2: Get horizon data
        horizons_df = self.to_pandas()
        required_cols = [hz_id_col, site_id_col, hz_top_col, hz_bot_col]
        available_cols = list(horizons_df.columns)

        # Step 3: Validate required columns
        valid, error, resolved = SPCColumnValidator.validate_required_columns(
            required_cols,
            available_cols,
            preset_name=preset if isinstance(preset, str) else "custom",
        )

        if not valid:
            raise error

        # Apply resolved column names
        if resolved:
            horizons_df = horizons_df.rename(columns=resolved)
            logger.info(f"Resolved columns: {resolved}")

        # Step 4: Validate depth values
        if validate_depths:
            depth_valid, depth_error, invalid_count = SPCDepthValidator.validate_depths(
                horizons_df, hz_top_col, hz_bot_col
            )
            if not depth_valid:
                SPCWarnings.warn_invalid_depths(invalid_count)

        # Step 5: Warn about missing site data
        if site_data is None:
            SPCWarnings.warn_missing_site_data()

        # Step 6: Create and return SoilProfileCollection
        try:
            spc = SoilProfileCollection(
                horizons=horizons_df,
                site=site_data,
                idname=site_id_col,
                hzidname=hz_id_col,
                depthcols=(hz_top_col, hz_bot_col),
            )
            logger.info(
                f"Successfully created SoilProfileCollection with {len(horizons_df)} horizons"
            )
            return spc
        except Exception as e:
            raise SPCValidationError(
                f"Failed to create SoilProfileCollection: {str(e)}",
                suggestion="Verify that all column values are valid and properly formatted.",
            ) from e

    def validate_for_soilprofilecollection(
        self,
        site_id_col: str = "cokey",
        hz_id_col: str = "chkey",
        hz_top_col: str = "hzdept_r",
        hz_bot_col: str = "hzdepb_r",
    ) -> dict:
        """
        Validate data suitability for SoilProfileCollection conversion without converting.

        This is useful for checking data quality before committing to a conversion.

        Args:
            site_id_col: Column name for site identifier
            hz_id_col: Column name for horizon identifier
            hz_top_col: Column name for horizon top depth
            hz_bot_col: Column name for horizon bottom depth

        Returns:
            Validation report dictionary with:
            - is_valid: Whether data can be converted
            - errors: List of validation errors
            - warnings: List of warnings
            - data_summary: Summary of data
            - validation_details: Detailed validation results

        Example:
            >>> response = await client.execute(query)
            >>> report = response.validate_for_soilprofilecollection()
            >>> if report["is_valid"]:
            ...     spc = response.to_soilprofilecollection()
            ... else:
            ...     print("Validation errors:", report["errors"])
        """
        try:
            import pandas  # noqa: F401
        except ImportError:
            raise ImportError("pandas is required for validation") from None

        df = self.to_pandas()
        available_cols = list(df.columns)
        required_cols = [site_id_col, hz_id_col, hz_top_col, hz_bot_col]

        report = create_spc_validation_report(
            df,
            required_cols,
            available_cols,
            hz_top_col,
            hz_bot_col,
        )

        return report

    def get_column_types(self) -> Dict[str, str]:
        """Extract column data types from metadata."""
        if not self._metadata:
            return {}

        types = {}
        for i, col_name in enumerate(self._columns):
            if i < len(self._metadata):
                metadata_str = self._metadata[i]
                # Parse metadata like "ColumnOrdinal=0,DataTypeName=varchar"
                if "DataTypeName=" in metadata_str:
                    type_part = metadata_str.split("DataTypeName=")[1]
                    data_type = (
                        type_part.split(",")[0] if "," in type_part else type_part
                    )
                    types[col_name] = data_type

        return types

    def get_python_types(self) -> Dict[str, str]:
        """Get Python-compatible type mapping using unified TypeMap."""
        sda_types = self.get_column_types()
        python_types = {}
        type_map = get_default_type_map()

        for col_name, sda_type in sda_types.items():
            python_type = type_map.get_python_type(sda_type)
            # Map Python types to string names used by SDA_TYPE_MAPPING
            type_name_map = {
                int: "int",
                float: "float",
                bool: "bit",
                str: "string",
                datetime: "datetime",
                bytes: "varbinary",
            }
            python_types[col_name] = type_name_map.get(python_type, "string")

        return python_types

    @property
    def validation_result(self) -> ValidationResult:
        """Get the validation result for this response."""
        if self._validation_result is None:
            self.validate_response()
        assert (
            self._validation_result is not None
        )  # validate_response() should have set this
        return self._validation_result

    @property
    def data_quality_score(self) -> float:
        """Get the data quality score (0.0 to 1.0)."""
        return self.validation_result.data_quality_score

    @property
    def has_warnings(self) -> bool:
        """Check if the response has any validation warnings."""
        return self.validation_result.has_warnings()

    @property
    def is_valid(self) -> bool:
        """Check if the response passed validation (no errors)."""
        return self.validation_result.is_valid()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        result = self.validation_result
        return {
            "is_valid": result.is_valid(),
            "has_warnings": result.has_warnings(),
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "data_quality_score": result.data_quality_score,
            "metadata": result.metadata,
            "errors": result.errors[:10],  # Limit to first 10 errors
            "warnings": result.warnings[:10],  # Limit to first 10 warnings
        }

    def recover_partial_data(
        self, max_errors: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Attempt to recover partial data by skipping malformed records.

        Args:
            max_errors: Maximum number of errors to tolerate before giving up

        Returns:
            Tuple of (valid_records, error_records)
        """
        if not self._columns:
            return [], []

        valid_records = []
        error_records = []
        error_count = 0

        column_types = self.get_column_types()

        for row_idx, row in enumerate(self._data):
            try:
                # Pad row with None if it's shorter than columns
                padded_row = row + [None] * (len(self._columns) - len(row))

                # Convert values with error recovery
                converted_row = {}
                row_errors = []

                for _col_idx, (col_name, value) in enumerate(
                    zip(self._columns, padded_row[: len(self._columns)])
                ):
                    try:
                        sda_type = column_types.get(col_name, "varchar").lower()
                        converted_value = self._convert_value(value, sda_type)
                        converted_row[col_name] = converted_value
                    except Exception as e:
                        row_errors.append(
                            {"column": col_name, "value": value, "error": str(e)}
                        )
                        # Use fallback value
                        converted_row[col_name] = value if value is not None else None

                if row_errors:
                    error_records.append(
                        {
                            "row_index": row_idx,
                            "original_row": row,
                            "converted_row": converted_row,
                            "errors": row_errors,
                        }
                    )

                valid_records.append(converted_row)

            except Exception as e:
                error_count += 1
                if error_count <= max_errors:
                    logger.warning(f"Skipping malformed row {row_idx}: {e}")
                    error_records.append(
                        {"row_index": row_idx, "original_row": row, "error": str(e)}
                    )
                else:
                    logger.error(f"Too many errors ({error_count}), stopping recovery")
                    break

        logger.info(
            f"Partial data recovery: {len(valid_records)} valid records, {len(error_records)} error records"
        )
        return valid_records, error_records

    @staticmethod
    def find_fallback_columns(
        available_columns: List[str],
        preferred_columns: List[str],
        fallback_mappings: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, str]:
        """Find fallback column names when preferred columns are not available.

        Args:
            available_columns: List of columns actually present in the data
            preferred_columns: List of preferred column names
            fallback_mappings: Optional mapping of preferred -> [fallback1, fallback2, ...]

        Returns:
            Dictionary mapping preferred column names to available fallback names
        """
        available_lower = {col.lower(): col for col in available_columns}
        mappings = {}

        # Default fallback mappings for common soil data columns
        default_fallbacks = {
            "cokey": ["compkey", "component_key", "site_key", "siteid"],
            "chkey": ["horizon_key", "hzkey", "horizon_id", "hzid"],
            "hzdept_r": ["top_depth", "depth_top", "hz_top", "top"],
            "hzdepb_r": ["bottom_depth", "depth_bottom", "hz_bottom", "bottom"],
            "latitude": ["lat", "y", "northing"],
            "longitude": ["lon", "lng", "x", "easting"],
            "mukey": ["mapunit_key", "mu_key"],
            "musym": ["mapunit_symbol", "symbol"],
            "muname": ["mapunit_name", "name"],
        }

        # Merge with provided mappings
        if fallback_mappings:
            default_fallbacks.update(fallback_mappings)

        for preferred in preferred_columns:
            if preferred in available_columns:
                mappings[preferred] = preferred
            else:
                # Look for exact matches first
                for fallback in default_fallbacks.get(preferred, []):
                    if fallback in available_columns:
                        mappings[preferred] = fallback
                        logger.info(
                            f"Using fallback column '{fallback}' for preferred '{preferred}'"
                        )
                        break
                else:
                    # Look for case-insensitive matches
                    preferred_lower = preferred.lower()
                    if preferred_lower in available_lower:
                        mappings[preferred] = available_lower[preferred_lower]
                        logger.info(
                            f"Using case-insensitive match '{available_lower[preferred_lower]}' for '{preferred}'"
                        )
                    else:
                        # Look for partial matches
                        for avail_col in available_columns:
                            if (
                                preferred_lower in avail_col.lower()
                                or avail_col.lower() in preferred_lower
                            ):
                                mappings[preferred] = avail_col
                                logger.info(
                                    f"Using partial match '{avail_col}' for preferred '{preferred}'"
                                )
                                break

        return mappings

    def __repr__(self) -> str:
        """String representation of the response."""
        return f"SDAResponse(columns={len(self._columns)}, rows={len(self._data)})"

    def __str__(self) -> str:
        """Human-readable string representation with validation info."""
        if self.is_empty():
            return "Empty SDA response"

        lines = [f"SDA Response: {len(self._data)} rows, {len(self._columns)} columns"]

        # Add validation summary if available
        if self._validation_result:
            result = self._validation_result
            if not result.is_valid():
                lines.append(f" Validation: {len(result.errors)} errors")
            elif result.has_warnings():
                lines.append(f"  Validation: {len(result.warnings)} warnings")
            else:
                lines.append(" Validation: OK")

            lines.append(f"Data Quality Score: {result.data_quality_score:.2f}")

        lines.append(
            f"Columns: {', '.join(self._columns[:5])}{'...' if len(self._columns) > 5 else ''}"
        )

        # Show column types
        column_types = self.get_column_types()
        if column_types:
            type_info = [
                f"{col}({column_types.get(col, 'unknown')})"
                for col in self._columns[:3]
            ]
            lines.append(
                f"Types: {', '.join(type_info)}{'...' if len(self._columns) > 3 else ''}"
            )

        # Show first few rows
        if self._data:
            lines.append("Sample data:")
            for i, row in enumerate(self._data[:3]):
                display_row = [str(x) if x is not None else "NULL" for x in row[:3]]
                lines.append(
                    f"  Row {i}: {', '.join(display_row)}{'...' if len(row) > 3 else ''}"
                )

            if len(self._data) > 3:
                lines.append(f"  ... and {len(self._data) - 3} more rows")

        return "\n".join(lines)
