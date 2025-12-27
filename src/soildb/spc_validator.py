"""
Validation utilities for SoilProfileCollection conversion.

This module provides robust validation logic for converting SDA query results
to SoilProfileCollection objects. It validates required columns, depth values,
and provides helpful error messages for missing or invalid data.
"""

import logging
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SPCValidationError(ValueError):
    """Raised when validation fails for SoilProfileCollection conversion."""

    def __init__(
        self,
        message: str,
        missing_columns: Optional[List[str]] = None,
        available_columns: Optional[List[str]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.missing_columns = missing_columns or []
        self.available_columns = available_columns or []
        self.suggestion = suggestion

        full_message = message
        if missing_columns:
            full_message += f"\n  Missing columns: {missing_columns}"
        if available_columns:
            full_message += f"\n  Available columns: {available_columns}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"

        super().__init__(full_message)


class SPCColumnValidator:
    """Validates that required columns exist in data for SPC conversion."""

    # Common fallback column names for key columns
    FALLBACK_COLUMNS = {
        "cokey": ["compkey", "component_key", "site_key", "component_id"],
        "chkey": ["horizon_key", "hzkey", "horizon_id", "layer_id"],
        "hzdept_r": ["top_depth", "depth_top", "hz_top", "depth_from", "top_cm"],
        "hzdepb_r": [
            "bottom_depth",
            "depth_bottom",
            "hz_bottom",
            "depth_to",
            "bottom_cm",
        ],
        "pedon_id": ["pedon_key", "site_id", "profile_id"],
        "hzname": ["horizon_name", "layer_name"],
    }

    @staticmethod
    def resolve_column(
        required_col: str,
        available_columns: List[str],
        silent: bool = False,
    ) -> Optional[str]:
        """
        Try to resolve a required column using fallback names.

        Args:
            required_col: The required column name
            available_columns: List of columns actually in the data
            silent: If True, don't log resolution messages

        Returns:
            Resolved column name, or None if not found
        """
        # First check if column exists
        if required_col in available_columns:
            return required_col

        # Try fallback names
        fallbacks = SPCColumnValidator.FALLBACK_COLUMNS.get(required_col, [])
        for fallback in fallbacks:
            if fallback in available_columns:
                if not silent:
                    logger.info(
                        f"Resolved column '{required_col}' to fallback '{fallback}'"
                    )
                return fallback

        return None

    @staticmethod
    def validate_required_columns(
        required_cols: List[str],
        available_columns: List[str],
        preset_name: str = "custom",
    ) -> Tuple[bool, Optional[SPCValidationError], dict]:
        """
        Validate that all required columns exist in data.

        Attempts to resolve missing columns using fallback names.

        Args:
            required_cols: List of required column names
            available_columns: List of columns in the data
            preset_name: Name of the preset being used (for error messages)

        Returns:
            Tuple of (is_valid, error_or_none, resolved_columns_dict)
            where resolved_columns_dict maps original names to resolved names
        """
        missing = []
        resolved = {}

        for col in required_cols:
            resolved_col = SPCColumnValidator.resolve_column(
                col, available_columns, silent=False
            )
            if resolved_col is None:
                missing.append(col)
            elif resolved_col != col:
                resolved[col] = resolved_col

        if missing:
            suggestion = (
                f"Use the '{preset_name}' preset, or provide a CustomColumnConfig "
                f"matching your data structure. See spc_presets module for examples."
            )
            error = SPCValidationError(
                "Cannot convert to SoilProfileCollection: missing required columns",
                missing_columns=missing,
                available_columns=available_columns,
                suggestion=suggestion,
            )
            return False, error, resolved

        return True, None, resolved


class SPCDepthValidator:
    """Validates depth values for horizon data."""

    @staticmethod
    def validate_depths(
        df: "pd.DataFrame",
        top_col: str,
        bottom_col: str,
        depth_units: str = "inches",
    ) -> Tuple[bool, Optional[str], int]:
        """
        Validate horizon depth values.

        Checks for:
        - Non-numeric values
        - NULL/None values
        - Top depth >= bottom depth
        - Negative depths
        - Unreasonable depth values

        Args:
            df: DataFrame with horizon data
            top_col: Column name for top depth
            bottom_col: Column name for bottom depth
            depth_units: Units of depth ('inches' or 'centimeters')

        Returns:
            Tuple of (is_valid, error_message_or_none, count_of_invalid_rows)
        """
        import pandas as pd

        # Convert to numeric, coercing errors to NaN
        top_depths = pd.to_numeric(df[top_col], errors="coerce")
        bottom_depths = pd.to_numeric(df[bottom_col], errors="coerce")

        # Find invalid rows
        invalid_mask = (
            top_depths.isna()  # Non-numeric or NULL top depth
            | bottom_depths.isna()  # Non-numeric or NULL bottom depth
            | (top_depths < 0)  # Negative top depth
            | (bottom_depths < 0)  # Negative bottom depth
            | (top_depths >= bottom_depths)  # Top >= bottom
        )

        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            # Check for unreasonable depths (sanity check)
            max_depth = 240 if depth_units == "inches" else 600  # 20 ft / 600 cm

            unreasonable = (
                (bottom_depths > max_depth) | (top_depths > max_depth)
            ).sum()

            error_msg = f"Found {invalid_count} invalid depth records"
            if unreasonable > 0:
                error_msg += f" ({unreasonable} with unreasonable depth values)"

            logger.warning(error_msg)
            return False, error_msg, invalid_count

        return True, None, 0

    @staticmethod
    def get_depth_statistics(
        df: "pd.DataFrame",
        top_col: str,
        bottom_col: str,
    ) -> dict:
        """
        Get statistics about depth values in the data.

        Args:
            df: DataFrame with horizon data
            top_col: Column name for top depth
            bottom_col: Column name for bottom depth

        Returns:
            Dictionary with depth statistics
        """
        import pandas as pd

        top_depths = pd.to_numeric(df[top_col], errors="coerce")
        bottom_depths = pd.to_numeric(df[bottom_col], errors="coerce")

        # Remove NaN for statistics
        top_valid = top_depths.dropna()
        bottom_valid = bottom_depths.dropna()

        # Count valid depths
        valid_count = top_valid.notna().sum() + bottom_valid.notna().sum()

        return {
            "total_horizons": len(df),
            "horizons_with_valid_depths": valid_count if valid_count > 0 else len(df),
            "top_depth_min": float(top_valid.min()) if len(top_valid) > 0 else None,
            "top_depth_max": float(top_valid.max()) if len(top_valid) > 0 else None,
            "top_depth_mean": float(top_valid.mean()) if len(top_valid) > 0 else None,
            "bottom_depth_min": float(bottom_valid.min())
            if len(bottom_valid) > 0
            else None,
            "bottom_depth_max": float(bottom_valid.max())
            if len(bottom_valid) > 0
            else None,
            "bottom_depth_mean": float(bottom_valid.mean())
            if len(bottom_valid) > 0
            else None,
        }


class SPCWarnings:
    """Warning messages for implicit column assumptions."""

    @staticmethod
    def warn_default_columns() -> None:
        """Placeholder for warning that default column names are being used."""
        return

    @staticmethod
    def warn_fallback_resolution(original: str, resolved: str) -> None:
        """Warn that a column was resolved to a fallback name."""
        warnings.warn(
            f"Column '{original}' not found; using fallback '{resolved}'. "
            "If this is incorrect, provide explicit column parameters.",
            UserWarning,
            stacklevel=4,
        )

    @staticmethod
    def warn_missing_site_data() -> None:
        """Placeholder for warning that site-level data was not provided."""
        return

    @staticmethod
    def warn_invalid_depths(invalid_count: int) -> None:
        """Warn about invalid depth records."""
        warnings.warn(
            f"Found {invalid_count} horizon records with invalid depth values "
            "(missing, non-numeric, or top >= bottom). These records may cause issues. "
            "Review the depth columns for data quality.",
            UserWarning,
            stacklevel=3,
        )

    @staticmethod
    def warn_missing_optional_columns(missing: List[str]) -> None:
        """Warn about missing optional columns."""
        if missing:
            warnings.warn(
                f"Optional columns not found: {missing}. ",
                UserWarning,
                stacklevel=3,
            )


def create_spc_validation_report(
    df: "pd.DataFrame",
    required_cols: List[str],
    available_cols: List[str],
    top_col: str,
    bottom_col: str,
) -> dict:
    """
    Create a comprehensive validation report for SoilProfileCollection conversion.

    Args:
        df: The horizon DataFrame
        required_cols: Required column names
        available_cols: Available column names in data
        top_col: Horizon top depth column
        bottom_col: Horizon bottom depth column

    Returns:
        Dictionary with validation report including:
        - column_validation: Results of column validation
        - depth_validation: Results of depth validation
        - data_summary: Summary of data
        - warnings: List of warnings
    """

    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "data_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
        },
        "validation_details": {},
    }

    # Check required columns
    valid, error, resolved = SPCColumnValidator.validate_required_columns(
        required_cols, available_cols
    )
    if not valid:
        report["is_valid"] = False  # type: ignore[index]
        report["errors"].append(str(error))  # type: ignore[attr-defined]
    report["validation_details"]["columns"] = {  # type: ignore[index]
        "valid": valid,
        "missing": error.missing_columns if error else [],
        "resolved_mappings": resolved,
    }

    # Check depth values
    if valid:  # Only check depths if columns exist
        depth_valid, depth_error, invalid_count = SPCDepthValidator.validate_depths(
            df, top_col, bottom_col
        )
        if not depth_valid:
            report["warnings"].append(depth_error)  # type: ignore[attr-defined]
        report["validation_details"]["depths"] = {  # type: ignore[index]
            "valid": depth_valid,
            "invalid_count": invalid_count,
            "statistics": SPCDepthValidator.get_depth_statistics(
                df, top_col, bottom_col
            ),
        }

    return report
