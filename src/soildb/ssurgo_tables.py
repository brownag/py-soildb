"""
SSURGO table metadata: keys, geometry columns, and default column sets.

This module provides a centralized registry of SSURGO table structures,
including primary key columns, geometry column mappings for spatial tables,
default column selections, and filter field definitions for common queries.

Keep LDM's ldm/tables.TABLE_KEY_COLUMNS separate per ADR-0003.

Public API:
  - KEY_COLUMNS: dict mapping SSURGO table names to their key columns (used by backends)
  - GEOMETRY_COLUMNS: dict mapping spatial SSURGO tables to geometry columns (used by backends)
  - DEFAULT_COLUMNS: dict mapping table names to default column lists (used by spatial.py, query_templates)
  - FILTER_FIELDS: dict mapping table names to (primary, secondary, alternate) filter columns (used by ssurgo_client.py)
  - key_column(table) -> Optional[str]: case-insensitive key column lookup
  - geometry_column(table) -> Optional[str]: case-insensitive geometry column lookup
  - default_columns(table) -> Optional[list[str]]: case-insensitive default column lookup
  - filter_fields(table) -> Optional[tuple[Optional[str], Optional[str], Optional[str]]]: case-insensitive filter field lookup

Internal:
  - TABLE_ALIASES: table name → alias mapping for spatial queries (used by spatial.py)
"""

from typing import Optional

# SSURGO table structures: table name → primary key column
KEY_COLUMNS: dict[str, str] = {
    # Core tables
    "legend": "lkey",
    "mapunit": "mukey",
    "component": "cokey",
    "chorizon": "chkey",
    "chfrags": "chfragkey",
    "chtexturegrp": "chtgkey",
    "chtexture": "chtkey",
    # Spatial tables
    "mupolygon": "mukey",
    "sapolygon": "areasymbol",  # or lkey
    "mupoint": "mukey",
    "muline": "mukey",
    "featpoint": "featkey",
    "featline": "featkey",
    # Interpretation tables
    "cointerp": "cokey",
    "chinterp": "chkey",
    "copmgrp": "copmgrpkey",
    "corestrictions": "reskeyid",
    # Administrative
    "sacatalog": "areasymbol",
    "laoverlap": "lkey",
    "legendtext": "lkey",
}

# Geometry columns for spatial SSURGO tables
GEOMETRY_COLUMNS: dict[str, str] = {
    "mupolygon": "mupolygongeo",
    "sapolygon": "sapolygongeo",
    "mupoint": "mupointgeo",
    "muline": "mulinegeo",
    "featpoint": "featpointgeo",
    "featline": "featlinegeo",
}

# Table aliases for spatial queries (internal: used by spatial.py for query construction)
# Maps table name → alias used in SQL FROM clauses
TABLE_ALIASES: dict[str, str] = {
    "mupolygon": "p",
    "mapunit": "m",
    "legend": "l",
    "sapolygon": "s",
    "mupoint": "pt",
    "muline": "ln",
    "featpoint": "fp",
    "featline": "fl",
}

# Default column selections for SSURGO tables (bare column names, no aliases)
# Used by spatial queries, ssurgo_client field defaults, and query templates
DEFAULT_COLUMNS: dict[str, list[str]] = {
    # Core tables
    "legend": ["lkey", "areasymbol", "areaname", "mlraoffice", "areaacres"],
    "mapunit": ["mukey", "musym", "muname", "mukind", "muacres"],
    "component": ["cokey", "mukey", "compname"],
    "chorizon": ["chkey", "cokey", "hzname"],
    # Spatial tables
    "mupolygon": ["mukey", "musym", "muname", "mukind", "areasymbol", "areaname"],
    "sapolygon": ["areasymbol", "spatialversion", "lkey"],
    "mupoint": ["mukey", "musym", "muname"],
    "muline": ["mukey", "musym", "muname"],
    "featpoint": ["featkey", "featsym"],
    "featline": ["featkey", "featsym"],
}

# Filter field selections for SSURGO tables (primary, secondary, alternate)
# First three columns from DEFAULT_COLUMNS, padded with None to 3-tuple
# Used by ssurgo_client.py for query field defaults
FILTER_FIELDS: dict[str, tuple[Optional[str], Optional[str], Optional[str]]] = {
    "legend": ("lkey", "areasymbol", "areaname"),
    "mapunit": ("mukey", "musym", "muname"),
    "component": ("cokey", "mukey", "compname"),
    "chorizon": ("chkey", "cokey", "hzname"),
    "mupolygon": ("mukey", "musym", "muname"),
    "sapolygon": ("areasymbol", "spatialversion", "lkey"),
    "mupoint": ("mukey", "musym", "muname"),
    "muline": ("mukey", "musym", "muname"),
    "featpoint": ("featkey", "featsym", None),
    "featline": ("featkey", "featsym", None),
}


def key_column(table: str) -> Optional[str]:
    """Get the primary key column name for a SSURGO table.

    Args:
        table: Table name (e.g., 'mapunit', 'component'). Case-insensitive.

    Returns:
        Key column name if table is recognized, None otherwise.

    Examples:
        >>> key_column("mapunit")
        'mukey'
        >>> key_column("COMPONENT")
        'cokey'
        >>> key_column("unknown")
        None
    """
    return KEY_COLUMNS.get(table.lower())


def geometry_column(table: str) -> Optional[str]:
    """Get the geometry column name for a spatial SSURGO table.

    Args:
        table: Table name (e.g., 'mupolygon', 'sapolygon'). Case-insensitive.

    Returns:
        Geometry column name if table is spatial, None otherwise.

    Examples:
        >>> geometry_column("mupolygon")
        'mupolygongeo'
        >>> geometry_column("SAPOLYGON")
        'sapolygongeo'
        >>> geometry_column("mapunit")
        None
    """
    return GEOMETRY_COLUMNS.get(table.lower())


def default_columns(table: str) -> Optional[list[str]]:
    """Get the default column selection for a SSURGO table.

    Returns bare column names without aliases or table prefixes.
    Callers (e.g., spatial.py) apply aliases and joins as needed.

    Args:
        table: Table name (e.g., 'mapunit', 'mupolygon'). Case-insensitive.

    Returns:
        List of default column names if table is recognized, None otherwise.

    Examples:
        >>> default_columns("mapunit")
        ['mukey', 'musym', 'muname', 'mukind', 'muacres']
        >>> default_columns("LEGEND")
        ['lkey', 'areasymbol', 'areaname', 'mlraoffice', 'areaacres']
        >>> default_columns("mupolygon")
        ['mukey', 'musym', 'muname', 'mukind', 'areasymbol', 'areaname']
        >>> default_columns("unknown")
        None
    """
    return DEFAULT_COLUMNS.get(table.lower())


def filter_fields(
    table: str,
) -> Optional[tuple[Optional[str], Optional[str], Optional[str]]]:
    """Get the filter field selection for a SSURGO table.

    Returns a 3-tuple of (primary, secondary, alternate) filter columns
    for use in query defaults. Columns are bare names without aliases.

    Args:
        table: Table name (e.g., 'mapunit', 'chorizon'). Case-insensitive.

    Returns:
        Tuple of (primary, secondary, alternate) column names, with None
        padding for tables with fewer than 3 default columns. Returns None
        if table is not recognized.

    Examples:
        >>> filter_fields("mapunit")
        ('mukey', 'musym', 'muname')
        >>> filter_fields("FEATPOINT")
        ('featkey', 'featsym', None)
        >>> filter_fields("unknown")
        None
    """
    return FILTER_FIELDS.get(table.lower())


__all__ = [
    "KEY_COLUMNS",
    "GEOMETRY_COLUMNS",
    "DEFAULT_COLUMNS",
    "FILTER_FIELDS",
    "key_column",
    "geometry_column",
    "default_columns",
    "filter_fields",
]
