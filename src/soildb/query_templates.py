"""
Query templates for common Soil Data Access query patterns.

This module provides convenient factory functions for building common SDA queries.
These functions are simpler and more intuitive than using the QueryBuilder class directly.

Examples:
    # Get map units for a survey area
    query = query_mapunits_by_legend("IA109")

    # Get components at a specific point
    query = query_components_at_point(-93.5, 42.5)

    # Get available survey areas
    query = query_available_survey_areas()

    # Get pedons in a bounding box
    query = query_pedons_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
"""

from typing import Dict, List, Optional

from .query import ColumnSets, Query
from .sanitization import (
    sanitize_sql_numeric,
    sanitize_sql_string,
    sanitize_sql_string_list,
    validate_sql_identifier,
)


def query_mapunits_by_legend(
    areasymbol: str, columns: Optional[List[str]] = None
) -> Query:
    """Get map units for a survey area by legend/area symbol.

    Args:
        areasymbol: Area symbol (e.g., 'IA109')
        columns: Custom columns to select (defaults to basic map unit columns)

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_mapunits_by_legend("IA109")
        >>> query = query_mapunits_by_legend("IA109", columns=["mukey", "muname"])
    """
    if columns is None:
        columns = ColumnSets.MAPUNIT_BASIC + ["l.areasymbol", "l.areaname"]

    return (
        Query()
        .select(*columns)
        .from_("mapunit m")
        .inner_join("legend l", "m.lkey = l.lkey")
        .where(f"l.areasymbol = {sanitize_sql_string(areasymbol)}")
        .order_by("m.musym")
    )


def query_components_by_legend(
    areasymbol: str, columns: Optional[List[str]] = None
) -> Query:
    """Get components for a survey area.

    Args:
        areasymbol: Area symbol (e.g., 'IA109')
        columns: Custom columns to select (defaults to basic component columns)

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_components_by_legend("IA109")
    """
    if columns is None:
        columns = ColumnSets.COMPONENT_BASIC + [
            "m.mukey",
            "m.musym",
            "m.muname",
            "l.areasymbol",
        ]

    return (
        Query()
        .select(*columns)
        .from_("component c")
        .inner_join("mapunit m", "c.mukey = m.mukey")
        .inner_join("legend l", "m.lkey = l.lkey")
        .where(f"l.areasymbol = {sanitize_sql_string(areasymbol)}")
        .order_by("m.musym, c.comppct_r DESC")
    )


def query_component_horizons_by_legend(
    areasymbol: str, columns: Optional[List[str]] = None
) -> Query:
    """Get component and horizon data for a survey area.

    Args:
        areasymbol: Area symbol (e.g., 'IA109')
        columns: Custom columns to select (defaults to detailed horizon columns)

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_component_horizons_by_legend("IA109")
    """
    if columns is None:
        # Qualify chorizon columns with table alias 'h' to avoid ambiguous column errors
        horizon_columns = [f"h.{col}" for col in ColumnSets.CHORIZON_TEXTURE]
        columns = [
            "m.mukey",
            "m.musym",
            "m.muname",
            "c.cokey",
            "c.compname",
            "c.comppct_r",
        ] + horizon_columns

    return (
        Query()
        .select(*columns)
        .from_("mapunit m")
        .inner_join("legend l", "m.lkey = l.lkey")
        .inner_join("component c", "m.mukey = c.mukey")
        .inner_join("chorizon h", "c.cokey = h.cokey")
        .where(
            f"l.areasymbol = {sanitize_sql_string(areasymbol)} AND c.majcompflag = 'Yes'"
        )
        .order_by("m.musym, c.comppct_r DESC, h.hzdept_r")
    )


def query_components_at_point(
    longitude: float, latitude: float, columns: Optional[List[str]] = None
) -> Query:
    """Get soil component data at a specific point.

    Args:
        longitude: Longitude of the point
        latitude: Latitude of the point
        columns: Custom columns to select (defaults to component and horizon columns)

    Returns:
        Query: A Query object ready for execution with spatial filter

    Examples:
        >>> query = query_components_at_point(-93.5, 42.5)
    """
    if columns is None:
        # Qualify chorizon columns with table alias 'h' to avoid ambiguous column errors
        horizon_columns = [f"h.{col}" for col in ColumnSets.CHORIZON_TEXTURE]
        columns = [
            "m.mukey",
            "m.musym",
            "m.muname",
            "c.compname",
            "c.comppct_r",
        ] + horizon_columns

    return (
        Query()
        .select(*columns)
        .from_("mupolygon p")
        .inner_join("mapunit m", "p.mukey = m.mukey")
        .inner_join("component c", "m.mukey = c.mukey")
        .inner_join("chorizon h", "c.cokey = h.cokey")
        .contains_point(longitude, latitude)
        .where("c.majcompflag = 'Yes'")
        .order_by("c.comppct_r DESC, h.hzdept_r")
    )


def query_mapunits_intersecting_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    columns: Optional[List[str]] = None,
) -> Query:
    """Get map units that intersect with a bounding box.

    Args:
        min_x: Minimum longitude (west bound)
        min_y: Minimum latitude (south bound)
        max_x: Maximum longitude (east bound)
        max_y: Maximum latitude (north bound)
        columns: Custom columns to select (defaults to basic map unit columns with geometry)

    Returns:
        Query: A Query object ready for execution with spatial filter

    Examples:
        >>> query = query_mapunits_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
    """
    if columns is None:
        columns = [
            "m.mukey",
            "m.musym",
            "m.muname",
            "mupolygongeo.STAsText() as geometry",
        ]

    return (
        Query()
        .select(*columns)
        .from_("mupolygon p")
        .inner_join("mapunit m", "p.mukey = m.mukey")
        .intersects_bbox(min_x, min_y, max_x, max_y)
    )


def query_spatial_by_legend(
    areasymbol: str, columns: Optional[List[str]] = None
) -> Query:
    """Get spatial data for map units on a legend/area symbol.

    Args:
        areasymbol: Area symbol (e.g., 'IA109')
        columns: Custom columns to select (defaults to spatial map unit columns)

    Returns:
        Query: A Query object ready for execution with geometry

    Examples:
        >>> query = query_spatial_by_legend("IA109")
    """
    if columns is None:
        columns = ColumnSets.MAPUNIT_SPATIAL + [
            "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STArea() as shape_area",
            "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STLength() as shape_length",
        ]

    return (
        Query()
        .select(*columns)
        .from_("mupolygon")
        .where(f"areasymbol = {sanitize_sql_string(areasymbol)}")
    )


def query_available_survey_areas(
    columns: Optional[List[str]] = None, table: str = "sacatalog"
) -> Query:
    """Get list of available survey areas.

    Args:
        columns: Custom columns to select
        table: Table to query (default: 'sacatalog' or 'legend')

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_available_survey_areas()
        >>> query = query_available_survey_areas(table="legend")
    """
    validate_sql_identifier(table)

    if columns is None:
        if table == "sacatalog":
            columns = ["areasymbol", "areaname", "saversion"]
        else:
            columns = ColumnSets.LEGEND_BASIC

    return Query().select(*columns).from_(table).order_by("areasymbol")


def query_survey_area_boundaries(
    columns: Optional[List[str]] = None, table: str = "sapolygon"
) -> Query:
    """Get survey area boundary polygons.

    Args:
        columns: Custom columns to select (defaults to area symbol, name, and geometry)
        table: Table to query (default: 'sapolygon')

    Returns:
        Query: A Query object ready for execution with spatial data

    Examples:
        >>> query = query_survey_area_boundaries()
    """
    validate_sql_identifier(table)

    if columns is None:
        columns = ["areasymbol", "areaname", "sapolygongeo.STAsText() as geometry"]

    return Query().select(*columns).from_(table)


def query_from_sql(sql: str) -> Query:
    """Create a query from a raw SQL string.

    Args:
        sql: The raw SQL query string

    Returns:
        Query: A Query object

    Examples:
        >>> query = query_from_sql("SELECT TOP 10 mukey, muname FROM mapunit")
    """
    return Query.from_sql(sql)


def query_pedons_intersecting_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    columns: Optional[List[str]] = None,
    base_table: str = "lab_combine_nasis_ncss",
    related_tables: Optional[List[str]] = None,
    lon_column: str = "longitude_decimal_degrees",
    lat_column: str = "latitude_decimal_degrees",
) -> Query:
    """Get pedons that intersect with a bounding box with flexible table joining.

    Args:
        min_x: Minimum longitude
        min_y: Minimum latitude
        max_x: Maximum longitude
        max_y: Maximum latitude
        columns: Columns to select (defaults to basic pedon columns)
        base_table: Base pedon/site table (default: "lab_combine_nasis_ncss")
        related_tables: Additional tables to left join
        lon_column: Name of the longitude column (default: "longitude_decimal_degrees")
        lat_column: Name of the latitude column (default: "latitude_decimal_degrees")

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_pedons_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
    """
    if columns is None:
        columns = ColumnSets.PEDON_BASIC + ["corr_name", "samp_name"]

    # Validate column and table names (security: whitelist allowed identifiers)
    validate_sql_identifier(lon_column)
    validate_sql_identifier(lat_column)
    validate_sql_identifier(base_table)

    query = (
        Query()
        .select(*columns)
        .from_(f"{base_table} p")
        .where(
            f"p.{lat_column} >= {sanitize_sql_numeric(min_y)} AND p.{lat_column} <= {sanitize_sql_numeric(max_y)}"
        )
        .where(
            f"p.{lon_column} >= {sanitize_sql_numeric(min_x)} AND p.{lon_column} <= {sanitize_sql_numeric(max_x)}"
        )
        .where(f"p.{lat_column} IS NOT NULL AND p.{lon_column} IS NOT NULL")
    )

    # Add joins for related tables
    if related_tables:
        for i, table in enumerate(related_tables):
            # Validate table name to prevent SQL injection
            validate_sql_identifier(table)
            alias = f"t{i}"
            # Most pedon-related tables join on pedon_key
            query = query.left_join(
                f"{table} {alias}", f"p.pedon_key = {alias}.pedon_key"
            )

    return query


def query_pedon_horizons_by_pedon_keys(
    pedon_keys: List[str],
    columns: Optional[List[str]] = None,
    base_table: str = "lab_layer",
    related_tables: Optional[List[str]] = None,
) -> Query:
    """Get horizon data for specified pedon keys with flexible table joining.

    Args:
        pedon_keys: List of pedon keys to query
        columns: Columns to select (defaults to basic lab horizon columns)
        base_table: Base horizon table (default: "lab_layer")
        related_tables: Additional tables to left join (default: basic lab tables)

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_pedon_horizons_by_pedon_keys(["12345", "67890"])
    """
    if related_tables is None:
        related_tables = ["lab_physical_properties", "lab_chemical_properties"]

    if columns is None:
        columns = (
            [
                "l.pedon_key",
                "l.layer_key",
                "l.layer_sequence",
                "l.hzn_top",
                "l.hzn_bot",
                "l.hzn_desgn",
            ]
            + ColumnSets.LAB_HORIZON_TEXTURE[5:]
            + ColumnSets.LAB_HORIZON_CHEMICAL[5:]
            + ColumnSets.LAB_HORIZON_PHYSICAL[5:]
        )

    # Validate table name
    validate_sql_identifier(base_table)

    # Build IN clause for pedon keys
    keys_str = ", ".join(sanitize_sql_string_list(pedon_keys))

    query = (
        Query()
        .select(*columns)
        .from_(f"{base_table} l")
        .where(f"l.pedon_key IN ({keys_str})")
        .where("l.layer_type = 'horizon'")
    )

    # Add joins for related tables
    # Most lab tables join on labsampnum
    lab_join_tables = {
        "lab_physical_properties",
        "lab_chemical_properties",
        "lab_calculations_including_estimates_and_default_values",
        "lab_rosetta_key",
        "lab_mir",
        "lab_mineralogy_glass_count",
        "lab_major_and_trace_elements_and_oxides",
        "lab_xray_and_thermal",
    }

    for i, table in enumerate(related_tables):
        # Validate table name to prevent SQL injection
        validate_sql_identifier(table)
        alias = f"t{i}"
        if table in lab_join_tables:
            # Lab tables typically join on labsampnum
            query = query.left_join(
                f"{table} {alias}", f"l.labsampnum = {alias}.labsampnum"
            )
        else:
            # For other tables, try pedon_key join (could be extended for other join keys)
            query = query.left_join(
                f"{table} {alias}", f"l.pedon_key = {alias}.pedon_key"
            )

    return query.order_by("l.pedon_key, l.layer_sequence")


def query_pedon_by_pedon_key(
    pedon_key: str,
    columns: Optional[List[str]] = None,
    base_table: str = "lab_combine_nasis_ncss",
    related_tables: Optional[List[str]] = None,
) -> Query:
    """Get a single pedon by its pedon key with flexible table joining.

    Args:
        pedon_key: Pedon key to query
        columns: Columns to select (defaults to basic pedon columns)
        base_table: Base pedon/site table (default: "lab_combine_nasis_ncss")
        related_tables: Additional tables to left join

    Returns:
        Query: A Query object ready for execution

    Examples:
        >>> query = query_pedon_by_pedon_key("12345")
    """
    if columns is None:
        columns = ColumnSets.PEDON_BASIC

    # Validate table name
    validate_sql_identifier(base_table)

    query = (
        Query()
        .select(*columns)
        .from_(f"{base_table} p")
        .where(f"p.pedon_key = {sanitize_sql_string(pedon_key)}")
    )

    # Add joins for related tables
    if related_tables:
        for i, table in enumerate(related_tables):
            # Validate table name to prevent SQL injection
            validate_sql_identifier(table)
            alias = f"t{i}"
            # Most pedon-related tables join on pedon_key
            query = query.left_join(
                f"{table} {alias}", f"p.pedon_key = {alias}.pedon_key"
            )

    return query
