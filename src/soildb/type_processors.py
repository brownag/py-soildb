"""
Type processors for converting SDA response data to appropriate Python types.

This module provides functions to convert raw values from SDA responses (typically strings)
to appropriate Python types. These processors are used by the schema_system to automatically
convert data during response parsing.

## Usage

Type processors are typically used when defining ColumnSchema objects:

    from soildb.schema_system import ColumnSchema
    from soildb.type_processors import to_optional_int, to_str

    column = ColumnSchema(
        name="mukey",
        type_hint=int,
        processor=to_optional_int,
        default=True,
        description="Map unit key"
    )

## Extending with Custom Processors

To create a custom processor, follow this pattern:

    def my_custom_processor(value: Any) -> MyType:
        '''Convert value to MyType, handling null/missing values.'''
        if not _notna(value):
            return None  # or appropriate default
        # ... your conversion logic
        return converted_value

The processor function should:
1. Handle None and null-like values gracefully
2. Return the target type or None
3. Not raise exceptions (return None on conversion failure)
4. Include clear docstrings explaining the conversion
"""

from datetime import datetime
from typing import Any, Optional


def _notna(value: Any) -> bool:
    """
    Check if a value is not NaN/null, without requiring pandas.

    Internal utility used by type processors to determine if a value should be
    converted or returned as None.

    Recognizes:
    - None values
    - float NaN, inf, -inf
    - String literals: "null", "none", "" (case-insensitive)
    - pandas NA types (if pandas is installed)
    """
    if value is None:
        return False
    if isinstance(value, float) and str(value).lower() in ("nan", "inf", "-inf"):
        return False
    if isinstance(value, str) and value.lower() in ("null", "none", ""):
        return False
    # Handle pandas NA types
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except ImportError:
        pass
    return True



def to_optional_float(value: Any) -> Optional[float]:
    """
    Convert to float, returning None if the value is null/NaN.

    Used for numeric SDA columns that may be missing or null.

    Examples:
        >>> to_optional_float("3.14")
        3.14
        >>> to_optional_float("null")
        None
        >>> to_optional_float(None)
        None
    """
    return float(value) if _notna(value) else None


def to_optional_int(value: Any) -> Optional[int]:
    """
    Convert to int, returning None if the value is null/NaN.

    Used for integer SDA columns that may be missing or null.

    Examples:
        >>> to_optional_int("42")
        42
        >>> to_optional_int("null")
        None
        >>> to_optional_int(None)
        None
    """
    return int(value) if _notna(value) else None


def to_str(value: Any) -> str:
    """
    Convert to string, returning empty string if null/NaN.

    Used for required string columns in SDA data. Always returns a string
    (never None) to ensure fields are never null.

    Examples:
        >>> to_str("hello")
        'hello'
        >>> to_str("null")
        ''
        >>> to_str(None)
        ''
    """
    return str(value) if _notna(value) else ""


def to_optional_str(value: Any) -> Optional[str]:
    """
    Convert to string or None if the value is null/NaN.

    Used for optional string columns in SDA data that may be genuinely null.

    Examples:
        >>> to_optional_str("hello")
        'hello'
        >>> to_optional_str("null")
        None
        >>> to_optional_str(None)
        None
    """
    return str(value) if _notna(value) else None


def to_datetime(value: Any) -> Optional[datetime]:
    """
    Convert value to datetime, handling various SDA datetime formats.

    Attempts to parse common datetime formats from SDA responses:
    - ISO formats: "2023-01-15T10:30:00Z", "2023-01-15T10:30:00.123456"
    - Standard SQL: "2023-01-15 10:30:00", "2023-01-15"
    - US format: "01/15/2023"

    If dateutil is installed, uses flexible parsing for other formats.
    Returns None if parsing fails or value is null.

    Examples:
        >>> to_datetime("2023-01-15").date()
        datetime.date(2023, 1, 15)
        >>> to_datetime("2023-01-15 10:30:00").hour
        10
        >>> to_datetime("null")
        None
    """
    if value is None or value == "":
        return None
    try:
        # Handle various SDA datetime formats
        if isinstance(value, str):
            # Try common date formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

            # Try parsing with dateutil if available
            try:
                from dateutil import parser  # type: ignore[import-untyped]

                return parser.parse(value)  # type: ignore[no-any-return]
            except ImportError:
                pass

        return None  # Return None if parsing fails, not string
    except (ValueError, TypeError):
        return None
