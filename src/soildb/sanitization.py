"""
SQL input sanitization utilities for safe query building.

This module provides functions to prevent SQL injection attacks when building
dynamic SQL queries. See OWASP SQL Injection prevention:
https://owasp.org/www-community/attacks/SQL_Injection

All user input that appears in SQL must be:

1. String values (table/column names) -> use validate_sql_identifier()
2. String data values (WHERE clause literals) -> use sanitize_sql_string()
3. Numeric values (WHERE clause comparisons) -> use sanitize_sql_numeric()
4. WKT geometries -> use validate_wkt_geometry()
"""

import re
from typing import List, Union

# SQL Identifier validation pattern: letters, numbers, underscores, periods (for alias.column)
_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$"

# WKT geometry type pattern
_WKT_PATTERN = r"^(POINT|POLYGON|MULTIPOLYGON|LINESTRING|MULTILINESTRING|GEOMETRYCOLLECTION)\s*\("


def sanitize_sql_string(value: str) -> str:
    """
    Sanitize a string value for safe SQL insertion as a string literal.

    Escapes single quotes by doubling them (''), which is the SQL standard.
    Returns the value wrapped in single quotes, ready for use in WHERE clauses.

    **Security:** Prevents SQL injection by escaping quotes in user input.
    Reference: https://owasp.org/www-community/attacks/SQL_Injection

    Args:
        value: String value to sanitize (e.g., area symbol, soil name)

    Returns:
        SQL-safe string literal (e.g., "'IA001'" or "'O'Brien'")

    Raises:
        ValueError: If value is not a string

    Example:
        >>> sanitize_sql_string("IA109")
        "'IA109'"
        >>> sanitize_sql_string("O'Brien")
        "'O''Brien'"
    """
    if not isinstance(value, str):
        raise ValueError(f"sanitize_sql_string requires str, got {type(value).__name__}")
    # Escape single quotes by doubling them (SQL standard)
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def sanitize_sql_numeric(value: Union[int, float, str]) -> str:
    """
    Sanitize a numeric value for safe SQL insertion.

    Validates that the input can be converted to a number and returns the string
    representation. Prevents SQL injection through malformed numeric values.

    **Security:** Validates input is numeric before including in SQL.
    Prevents injection attacks like: '1 OR 1=1' or '1; DROP TABLE;'

    Args:
        value: Numeric value (int, float, or numeric string)

    Returns:
        String representation of the number (e.g., "42", "3.14159")

    Raises:
        ValueError: If value cannot be converted to a number

    Example:
        >>> sanitize_sql_numeric(42)
        '42'
        >>> sanitize_sql_numeric("3.14159")
        '3.14159'
        >>> sanitize_sql_numeric("42; DROP TABLE")
        ValueError: Invalid numeric value: 42; DROP TABLE
    """
    try:
        # Validate by converting to float
        float(value)
        return str(value)
    except (ValueError, TypeError) as err:
        raise ValueError(f"sanitize_sql_numeric requires numeric value, got {value!r}") from err


def validate_sql_identifier(identifier: str) -> str:
    """
    Validate a SQL identifier (table or column name) for safe SQL insertion.

    Identifiers must be alphanumeric with underscores, optionally qualified with
    alias (e.g., 't.column', 'map_unit.mukey'). This prevents SQL injection
    through table/column name manipulation.

    **Security:** Whitelist approach - only allows valid identifier characters.
    Prevents injection attacks like: 'table; DROP TABLE other;' or 'column' OR '1'='1'

    Args:
        identifier: Table name, column name, or qualified name (e.g., "m.areasymbol")

    Returns:
        The validated identifier (unchanged if valid)

    Raises:
        ValueError: If identifier contains invalid characters

    Example:
        >>> validate_sql_identifier("mapunit")
        'mapunit'
        >>> validate_sql_identifier("m.mukey")
        'm.mukey'
        >>> validate_sql_identifier("mapunit; DROP TABLE")
        ValueError: Invalid SQL identifier: mapunit; DROP TABLE
    """
    if not re.match(_IDENTIFIER_PATTERN, identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}. Must be alphanumeric + underscore.")
    return identifier


def validate_wkt_geometry(wkt: str) -> str:
    """
    Validate Well-Known Text (WKT) geometry string for safe SQL insertion.

    Performs basic validation that the WKT string starts with a recognized
    geometry type. Does not validate full WKT syntax (spatial.py handles that).

    **Security:** Basic sanity check - prevents obviously malformed geometry
    that could be SQL injection. Full validation done by spatial database.

    Args:
        wkt: WKT geometry string (e.g., "POINT(-93.5 42.0)")

    Returns:
        The validated WKT string (unchanged if valid)

    Raises:
        ValueError: If WKT doesn't match known geometry type pattern

    Example:
        >>> validate_wkt_geometry("POINT(-93.5 42.0)")
        'POINT(-93.5 42.0)'
        >>> validate_wkt_geometry("INVALID(-93.5 42.0)")
        ValueError: Invalid WKT geometry: INVALID(-93.5 42.0)
    """
    if not re.match(_WKT_PATTERN, wkt.upper(), re.IGNORECASE):
        raise ValueError(
            f"Invalid WKT geometry: {wkt!r}. Must start with valid geometry type "
            "(POINT, POLYGON, LINESTRING, etc.)"
        )
    return wkt


def sanitize_sql_string_list(values: List[str]) -> List[str]:
    """
    Sanitize a list of string values for safe SQL insertion.

    Applies sanitize_sql_string() to each value in the list. Useful for
    building IN clauses with user-supplied values.

    **Security:** Each string is independently escaped.

    Args:
        values: List of string values to sanitize

    Returns:
        List of SQL-safe string literals

    Example:
        >>> sanitize_sql_string_list(["IA001", "IA002", "IA003"])
        ["'IA001'", "'IA002'", "'IA003'"]

    Raises:
        ValueError: If any value is not a string
    """
    return [sanitize_sql_string(v) for v in values]
