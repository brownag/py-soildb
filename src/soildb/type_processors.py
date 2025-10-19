"""
Type processors for converting SDA response data to appropriate Python types.
"""

from datetime import date, datetime
from typing import Any, Optional


def _notna(value: Any) -> bool:
    """Check if a value is not NaN/null, without requiring pandas."""
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
    """Convert to float, returning None if NaN."""
    return float(value) if _notna(value) else None


def to_optional_int(value: Any) -> Optional[int]:
    """Convert to int, returning None if NaN."""
    return int(value) if _notna(value) else None


def to_str(value: Any) -> str:
    """Convert to string."""
    return str(value) if _notna(value) else ""


def to_optional_str(value: Any) -> Optional[str]:
    """Convert to string or None."""
    return str(value) if _notna(value) else None


def to_bool(value: Any) -> bool:
    """Convert to boolean."""
    if not _notna(value):
        return False
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "t")
    return bool(value)


def to_datetime(value: Any) -> Optional[datetime]:
    """Convert value to datetime, handling SDA formats."""
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


def to_date(value: Any) -> Optional[date]:
    """Convert value to date, handling SDA formats."""
    if value is None or value == "":
        return None
    try:
        dt = to_datetime(value)
        return dt.date() if dt else None
    except (ValueError, TypeError):
        return None
