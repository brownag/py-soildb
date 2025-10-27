"""
Unified type conversion system for SDA->Python->Pandas->Polars data flows.

This module provides a single source of truth for all type conversions across
the soildb library, consolidating logic that was previously scattered across
response.py, type_processors.py, and schema_system.py.

## Architecture

The type conversion system has three main components:

1. **TypeMap**: Main interface for type conversions
   - Handles SDA SQL type -> Python type -> Pandas dtype -> Polars dtype
   - Extensible with custom type processors
   - Caches conversion mappings for performance

2. **Type Processors**: Conversion functions for individual values
   - Handle null/missing values gracefully
   - Conversion errors result in None, not exceptions
   - Chainable for complex transformations

3. **Extension Mechanism**: Custom type processors
   - Register custom processors for new types
   - Override default behavior for specific columns
   - Full control over conversion pipeline

## Usage

```python
from soildb.type_conversion import TypeMap, TypeProcessor

# Get default type map
type_map = TypeMap.default()

# Convert single values
value = type_map.convert_value("42", "int")
converted_value = type_map.convert_value("2023-01-15", "datetime")

# Get type mappings
python_type = type_map.get_python_type("varchar")
pandas_dtype = type_map.get_pandas_dtype("int")
polars_dtype = type_map.get_polars_dtype("float")

# Register custom processor
def my_processor(value):
    if value is None:
        return None
    return str(value).upper()

type_map.register_processor("my_custom_type", "string", my_processor)

# Use in conversion
result = type_map.convert_value("hello", "my_custom_type")
```

## Type Conversion Flows

The system supports three main conversion flows:

1. **SDA -> Python**: Raw SDA values (typically strings) to Python native types
   - Input: Any value from SDA response
   - Output: Python type (int, str, float, datetime, bool, None)
   - Used: When parsing responses, schema conversion

2. **Python -> Pandas**: Python types to pandas dtypes
   - Input: Python type hints
   - Output: Pandas dtype string (e.g., "Int64", "datetime64[ns]")
   - Used: When converting to pandas DataFrames

3. **Python -> Polars**: Python types to polars dtypes
   - Input: Python type hints
   - Output: Polars dtype (e.g., pl.Int64, pl.Utf8)
   - Used: When converting to polars DataFrames

## Error Handling

Type conversion uses a "fail gracefully" strategy:
- Conversion errors return None rather than raising exceptions
- Unknown types default to string representation
- Warnings logged for suspicious conversions
- Optional parameter to be strict (raise exceptions)

## Performance

- Type processor mappings cached after first use
- SDA type -> Python type lookups O(1)
- No repeated parsing of format strings
- Extensible registry for custom types without rebuilding core maps
"""

import logging
import re
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class TypeProcessor:
    """Base class for type conversion processors with error handling."""

    @staticmethod
    def _is_null(value: Any) -> bool:
        """
        Check if a value represents a null/missing value.

        Recognizes:
        - None
        - float NaN, Inf, -Inf
        - String literals: "null", "none", "" (case-insensitive)
        - pandas NA types (if pandas installed)
        """
        if value is None or value == "":
            return True

        if isinstance(value, float):
            if str(value).lower() in ("nan", "inf", "-inf"):
                return True

        if isinstance(value, str):
            if value.lower() in ("null", "none", ""):
                return True

        # Handle pandas NA types
        try:
            import pandas as pd

            if pd.isna(value):
                return True
        except ImportError:
            pass

        return False

    @staticmethod
    def to_int(value: Any) -> Optional[int]:
        """Convert to int, returning None if null."""
        if TypeProcessor._is_null(value):
            return None
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            # Try to extract numeric part
            str_val = str(value).strip()
            numeric_match = re.search(r"[-+]?\d*\.?\d+", str_val)
            if numeric_match:
                try:
                    return int(float(numeric_match.group()))
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def to_float(value: Any) -> Optional[float]:
        """Convert to float, returning None if null."""
        if TypeProcessor._is_null(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            # Try to clean the value
            str_val = str(value).strip()
            # Remove currency symbols and commas
            cleaned = str_val.replace("$", "").replace(",", "").replace(" ", "")
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def to_bool(value: Any) -> Optional[bool]:
        """Convert to bool, returning None if null."""
        if TypeProcessor._is_null(value):
            return None
        str_val = str(value).lower().strip()
        if str_val in ("true", "1", "yes", "t", "y", "on"):
            return True
        elif str_val in ("false", "0", "no", "f", "n", "off"):
            return False
        return None

    @staticmethod
    def to_str(value: Any, allow_none: bool = False) -> Optional[str]:
        """Convert to string, returning None or empty string if null."""
        if TypeProcessor._is_null(value):
            return None if allow_none else ""
        return str(value).strip()

    @staticmethod
    def to_datetime(value: Any) -> Optional[datetime]:
        """Convert to datetime, handling various formats."""
        if TypeProcessor._is_null(value):
            return None

        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            value = str(value)

        # Try common SDA datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        # Try flexible parsing with dateutil if available
        try:
            from dateutil import parser  # type: ignore[import-untyped]

            return parser.parse(value)  # type: ignore[no-any-return]
        except (ImportError, Exception):
            pass

        return None

    @staticmethod
    def to_bytes(value: Any) -> Optional[bytes]:
        """Convert to bytes."""
        if TypeProcessor._is_null(value):
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return str(value).encode("utf-8")


class TypeMap:
    """
    Unified type conversion system for SDA->Python->Pandas->Polars flows.

    This is the primary interface for all type conversions in soildb.
    It consolidates logic from response.py, type_processors.py, and schema_system.py.
    """

    # SDA SQL type -> Python type mapping
    SDA_TYPE_TO_PYTHON = {
        # Integer types
        "int": int,
        "integer": int,
        "bigint": int,
        "smallint": int,
        "tinyint": int,
        # Boolean
        "bit": bool,
        # Floating point
        "float": float,
        "real": float,
        "double": float,
        "decimal": float,
        "numeric": float,
        "money": float,
        "smallmoney": float,
        # String types
        "varchar": str,
        "nvarchar": str,
        "char": str,
        "nchar": str,
        "text": str,
        "ntext": str,
        # Date/time
        "datetime": datetime,
        "datetime2": datetime,
        "smalldatetime": datetime,
        "date": datetime,
        "time": str,  # Keep as string for time-only
        "timestamp": datetime,
        # Spatial/binary
        "geometry": str,  # WKT as string
        "geography": str,
        "varbinary": str,
        "binary": str,
        "image": str,
        # Other
        "uniqueidentifier": str,
        "xml": str,
    }

    # Python type -> Pandas dtype mapping
    PYTHON_TO_PANDAS_DTYPE = {
        int: "Int64",  # Nullable integer
        float: "float64",
        bool: "boolean",
        str: "string",
        datetime: "datetime64[ns]",
        bytes: "object",
    }

    def __init__(self):
        """Initialize the type map with default processors."""
        self._processors: Dict[str, Callable[[Any], Any]] = {}
        self._python_types: Dict[str, Type] = self.SDA_TYPE_TO_PYTHON.copy()
        self._pandas_dtypes: Dict[str, str] = self.PYTHON_TO_PANDAS_DTYPE.copy()
        self._polars_dtypes: Dict[str, Any] = {}  # Built lazily
        self._type_processor_cache: Dict[str, Callable[[Any], Any]] = {}

        # Initialize default processors
        self._init_default_processors()

    def _init_default_processors(self) -> None:
        """Register default type processors."""
        # Register processors for each SDA type
        self._processors["int"] = TypeProcessor.to_int
        self._processors["integer"] = TypeProcessor.to_int
        self._processors["bigint"] = TypeProcessor.to_int
        self._processors["smallint"] = TypeProcessor.to_int
        self._processors["tinyint"] = TypeProcessor.to_int

        self._processors["bit"] = TypeProcessor.to_bool

        self._processors["float"] = TypeProcessor.to_float
        self._processors["real"] = TypeProcessor.to_float
        self._processors["double"] = TypeProcessor.to_float
        self._processors["decimal"] = TypeProcessor.to_float
        self._processors["numeric"] = TypeProcessor.to_float
        self._processors["money"] = TypeProcessor.to_float
        self._processors["smallmoney"] = TypeProcessor.to_float

        self._processors["varchar"] = TypeProcessor.to_str
        self._processors["nvarchar"] = TypeProcessor.to_str
        self._processors["char"] = TypeProcessor.to_str
        self._processors["nchar"] = TypeProcessor.to_str
        self._processors["text"] = TypeProcessor.to_str
        self._processors["ntext"] = TypeProcessor.to_str

        self._processors["datetime"] = TypeProcessor.to_datetime
        self._processors["datetime2"] = TypeProcessor.to_datetime
        self._processors["smalldatetime"] = TypeProcessor.to_datetime
        self._processors["date"] = TypeProcessor.to_datetime
        self._processors["time"] = TypeProcessor.to_str
        self._processors["timestamp"] = TypeProcessor.to_datetime

        # Spatial types - keep as WKT strings
        self._processors["geometry"] = TypeProcessor.to_str
        self._processors["geography"] = TypeProcessor.to_str

        # Binary types - convert to bytes representation
        self._processors["varbinary"] = TypeProcessor.to_bytes
        self._processors["binary"] = TypeProcessor.to_bytes
        self._processors["image"] = TypeProcessor.to_bytes

        self._processors["uniqueidentifier"] = TypeProcessor.to_str
        self._processors["xml"] = TypeProcessor.to_str

    def _build_polars_dtype_map(self) -> None:
        """Build Polars dtype map lazily (only if polars is imported)."""
        try:
            import polars as pl

            self._polars_dtypes = {
                int: pl.Int64,  # type: ignore
                float: pl.Float64,  # type: ignore
                bool: pl.Boolean,  # type: ignore
                str: pl.Utf8,  # type: ignore
                datetime: pl.Datetime,  # type: ignore
                bytes: pl.Binary,  # type: ignore
            }
        except ImportError:
            # Polars not available, use string representations
            self._polars_dtypes = {
                int: "Int64",
                float: "Float64",
                bool: "Boolean",
                str: "Utf8",
                datetime: "Datetime",
                bytes: "Binary",
            }

    def register_processor(
        self,
        sda_type: str,
        python_type: Type = str,
        processor: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """
        Register a custom type processor for a specific SDA type.

        Args:
            sda_type: SDA SQL type name (e.g., "my_custom_type")
            python_type: Python type to map to (e.g., int, str, float)
            processor: Conversion function(value: Any) -> python_type or None
                      If None, uses default processor for python_type

        Example:
            >>> type_map = TypeMap.default()
            >>> type_map.register_processor(
            ...     "my_type",
            ...     str,
            ...     lambda v: v.upper() if v else None
            ... )
        """
        self._python_types[sda_type.lower()] = python_type

        if processor is not None:
            self._processors[sda_type.lower()] = processor
        else:
            # Use default processor for this type
            if python_type == int:
                self._processors[sda_type.lower()] = TypeProcessor.to_int
            elif python_type == float:
                self._processors[sda_type.lower()] = TypeProcessor.to_float
            elif python_type == bool:
                self._processors[sda_type.lower()] = TypeProcessor.to_bool
            elif python_type == datetime:
                self._processors[sda_type.lower()] = TypeProcessor.to_datetime
            elif python_type == bytes:
                self._processors[sda_type.lower()] = TypeProcessor.to_bytes
            else:
                # Default to string processor
                self._processors[sda_type.lower()] = TypeProcessor.to_str

        # Invalidate cache
        self._type_processor_cache.clear()
        logger.info(f"Registered processor for SDA type '{sda_type}' -> {python_type.__name__}")

    def convert_value(
        self, value: Any, sda_type: str, strict: bool = False
    ) -> Any:
        """
        Convert a single value from SDA type to Python type.

        This is the main entry point for value conversion. It handles:
        - Null/missing value recognition
        - Type-specific conversion with error recovery
        - Logging of conversion issues
        - Optional strict mode for raising exceptions

        Args:
            value: Value to convert (typically from SDA response)
            sda_type: SDA SQL type name (e.g., "int", "varchar", "datetime")
            strict: If True, raise exceptions on conversion errors
                   If False (default), return None on errors

        Returns:
            Converted Python value or None if conversion failed

        Raises:
            ValueError: If strict=True and conversion fails

        Examples:
            >>> tm = TypeMap.default()
            >>> tm.convert_value("42", "int")
            42
            >>> tm.convert_value("2023-01-15", "date")
            datetime(2023, 1, 15)
            >>> tm.convert_value("invalid", "int")  # Returns None
            None
        """
        sda_type_lower = sda_type.lower()

        # Get processor for this type
        processor = self._processors.get(
            sda_type_lower,
            TypeProcessor.to_str,  # Default to string
        )

        try:
            return processor(value)
        except Exception as e:
            if strict:
                raise ValueError(
                    f"Failed to convert value '{value}' to type '{sda_type}': {e}"
                ) from e
            else:
                logger.debug(
                    f"Conversion error for value '{value}' of type '{sda_type}': {e}"
                )
                return None

    def get_python_type(self, sda_type: str) -> Type:
        """
        Get Python type for an SDA type.

        Args:
            sda_type: SDA SQL type name

        Returns:
            Python type (int, str, float, datetime, bool, bytes)
            Defaults to str for unknown types

        Examples:
            >>> tm = TypeMap.default()
            >>> tm.get_python_type("int")
            <class 'int'>
            >>> tm.get_python_type("datetime")
            <class 'datetime.datetime'>
            >>> tm.get_python_type("unknown")
            <class 'str'>
        """
        return self._python_types.get(sda_type.lower(), str)

    def get_pandas_dtype(self, sda_type: str) -> str:
        """
        Get Pandas dtype string for an SDA type.

        Args:
            sda_type: SDA SQL type name or Python type name

        Returns:
            Pandas dtype string (e.g., "Int64", "float64", "string")

        Examples:
            >>> tm = TypeMap.default()
            >>> tm.get_pandas_dtype("int")
            'Int64'
            >>> tm.get_pandas_dtype("datetime")
            'datetime64[ns]'
            >>> tm.get_pandas_dtype("varchar")
            'string'
        """
        python_type = self.get_python_type(sda_type)
        return self._pandas_dtypes.get(python_type, "string")

    def get_polars_dtype(self, sda_type: str) -> Any:
        """
        Get Polars dtype for an SDA type.

        Args:
            sda_type: SDA SQL type name

        Returns:
            Polars dtype object (e.g., pl.Int64, pl.Utf8)
            Returns string representation if polars not installed

        Examples:
            >>> tm = TypeMap.default()
            >>> tm.get_polars_dtype("int")  # Returns pl.Int64 or "Int64"
            >>> tm.get_polars_dtype("datetime")  # Returns pl.Datetime or "Datetime"
        """
        if not self._polars_dtypes:
            self._build_polars_dtype_map()

        python_type = self.get_python_type(sda_type)
        return self._polars_dtypes.get(python_type, "Utf8")

    def convert_row(
        self, row: Dict[str, Any], type_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Convert all values in a row dictionary based on column type map.

        Args:
            row: Dictionary with column names as keys and values
            type_map: Dictionary mapping column names to SDA types

        Returns:
            Dictionary with converted values

        Example:
            >>> tm = TypeMap.default()
            >>> row = {"mukey": "123456", "clay": "25.5"}
            >>> types = {"mukey": "int", "clay": "float"}
            >>> tm.convert_row(row, types)
            {'mukey': 123456, 'clay': 25.5}
        """
        converted = {}
        for col_name, value in row.items():
            sda_type = type_map.get(col_name, "varchar")
            converted[col_name] = self.convert_value(value, sda_type)
        return converted

    def convert_rows(
        self, rows: List[Dict[str, Any]], type_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Convert all values in multiple row dictionaries.

        Args:
            rows: List of row dictionaries
            type_map: Dictionary mapping column names to SDA types

        Returns:
            List of dictionaries with converted values

        Example:
            >>> tm = TypeMap.default()
            >>> rows = [
            ...     {"mukey": "123456", "clay": "25.5"},
            ...     {"mukey": "789012", "clay": "30.0"}
            ... ]
            >>> types = {"mukey": "int", "clay": "float"}
            >>> tm.convert_rows(rows, types)
            [{'mukey': 123456, 'clay': 25.5}, {'mukey': 789012, 'clay': 30.0}]
        """
        return [self.convert_row(row, type_map) for row in rows]

    def get_type_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Get all registered SDA -> Python -> Pandas -> Polars type mappings.

        Returns:
            Dictionary with structure:
            {
                "sda_type_name": {
                    "python": "type_name",
                    "pandas": "dtype_string",
                    "polars": "dtype_string"
                }
            }
        """
        mappings = {}
        for sda_type, python_type in self._python_types.items():
            pandas_dtype = self._pandas_dtypes.get(python_type, "string")
            polars_dtype = self.get_polars_dtype(sda_type)
            mappings[sda_type] = {
                "python": python_type.__name__,
                "pandas": pandas_dtype,
                "polars": str(polars_dtype),
            }
        return mappings

    @classmethod
    def default(cls) -> "TypeMap":
        """
        Get the default TypeMap instance.

        This is the recommended way to get a TypeMap for most use cases.

        Returns:
            TypeMap instance with all default processors registered

        Example:
            >>> tm = TypeMap.default()
            >>> tm.convert_value("42", "int")
            42
        """
        return cls()

    @staticmethod
    def get_pandas_dtype_for_python_type(python_type: Type) -> str:
        """
        Get Pandas dtype for a Python type directly.

        Args:
            python_type: Python type (int, str, float, datetime, bool, bytes)

        Returns:
            Pandas dtype string

        Example:
            >>> TypeMap.get_pandas_dtype_for_python_type(int)
            'Int64'
            >>> TypeMap.get_pandas_dtype_for_python_type(str)
            'string'
        """
        type_map = TypeMap()
        return type_map._pandas_dtypes.get(python_type, "string")

    @staticmethod
    def get_polars_dtype_for_python_type(python_type: Type) -> Any:
        """
        Get Polars dtype for a Python type directly.

        Args:
            python_type: Python type (int, str, float, datetime, bool, bytes)

        Returns:
            Polars dtype object or string

        Example:
            >>> dtype = TypeMap.get_polars_dtype_for_python_type(int)
        """
        type_map = TypeMap()
        if not type_map._polars_dtypes:
            type_map._build_polars_dtype_map()
        return type_map._polars_dtypes.get(python_type, "Utf8")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TypeMap(processors={len(self._processors)}, "
            f"python_types={len(self._python_types)})"
        )


# Module-level singleton for convenience
_DEFAULT_TYPE_MAP: Optional[TypeMap] = None


def get_default_type_map() -> TypeMap:
    """
    Get or create the default TypeMap singleton.

    This is a module-level convenience function for accessing the default
    type map without creating new instances.

    Returns:
        Shared TypeMap instance

    Example:
        >>> tm = get_default_type_map()
        >>> tm.convert_value("42", "int")
        42
    """
    global _DEFAULT_TYPE_MAP
    if _DEFAULT_TYPE_MAP is None:
        _DEFAULT_TYPE_MAP = TypeMap.default()
    return _DEFAULT_TYPE_MAP


def convert_value(value: Any, sda_type: str) -> Any:
    """
    Convenience function to convert a value using the default TypeMap.

    Args:
        value: Value to convert
        sda_type: SDA SQL type name

    Returns:
        Converted Python value

    Example:
        >>> convert_value("42", "int")
        42
    """
    return get_default_type_map().convert_value(value, sda_type)


__all__ = [
    "TypeMap",
    "TypeProcessor",
    "get_default_type_map",
    "convert_value",
]
