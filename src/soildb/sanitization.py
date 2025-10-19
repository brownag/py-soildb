"""
SQL input sanitization utilities for safe query building.
"""

import re
from typing import List, Optional, Union

# Known valid table names
VALID_TABLES = {
    "mapunit",
    "component",
    "chorizon",
    "legend",
    "mupolygon",
    "sapolygon",
    "lab_layer",
    "lab_combine_nasis_ncss",
    # ... add all valid tables
}

# Known valid column names (subset for common ones)
VALID_COLUMNS = {
    "mukey",
    "cokey",
    "chkey",
    "areasymbol",
    "pedon_key",
    "latitude_decimal_degrees",
    "longitude_decimal_degrees",
    # ... add commonly used columns
}


def sanitize_sql_string(value: str) -> str:
    """Sanitize a string value for SQL insertion."""
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value)}")
    # Escape single quotes
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def validate_sql_identifier(identifier: str, pattern: str = r"^[A-Za-z0-9_]+$") -> str:
    """Validate an identifier against a pattern."""
    if not re.match(pattern, identifier):
        raise ValueError(f"Invalid identifier: {identifier}")
    return identifier


def sanitize_sql_numeric(value: Union[int, float, str]) -> str:
    """Sanitize a numeric value."""
    try:
        # Convert to float to validate
        float(value)
        return str(value)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid numeric value: {value}") from err


def validate_wkt_geometry(wkt: str) -> str:
    """Basic WKT validation."""
    # Simple regex check for common WKT patterns
    wkt_pattern = r"^(POINT|POLYGON|MULTIPOLYGON|LINESTRING|MULTILINESTRING)\s*\("
    if not re.match(wkt_pattern, wkt.upper()):
        raise ValueError(f"Invalid WKT geometry: {wkt}")
    return wkt


def sanitize_sql_string_list(values: List[str]) -> List[str]:
    """Sanitize a list of string values."""
    return [sanitize_sql_string(v) for v in values]


def validate_sql_object_name(
    name: str, allowed_names: Optional[List[str]] = None
) -> str:
    """Validate table or column name."""
    if allowed_names and name not in allowed_names:
        raise ValueError(f"Invalid object name: {name}")
    return validate_sql_identifier(name)
