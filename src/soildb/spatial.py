"""
Spatial queries for SSURGO data.

Query soil data using points, bounding boxes, and polygons.
Returns tabular data or spatial data with geometry.

PRIMARY FUNCTION:
- spatial_query() - Use this for all spatial queries

CANONICAL DESIGN:
All spatial queries should use spatial_query(). See that function's docstring for
comprehensive documentation, examples, and design rationale.
"""

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from .client import SDAClient
from .query import Query
from .response import SDAResponse
from .sanitization import validate_wkt_geometry
from .ssurgo_tables import TABLE_ALIASES, geometry_column, spatial_table_joins
from .ssurgo_tables import default_columns as get_default_columns_bare
from .utils import add_sync_version, require_client

if TYPE_CHECKING:
    try:
        from shapely.geometry.base import (  # type: ignore[import-untyped]
            BaseGeometry,
        )
    except ImportError:
        BaseGeometry = Any

# Type aliases for clarity
GeometryInput = Union[str, "BaseGeometry", dict[str, float]]
TableType = Literal[
    "legend",
    "mapunit",
    "mupolygon",
    "sapolygon",
    "mupoint",
    "muline",
    "featpoint",
    "featline",
]
ReturnType = Literal["tabular", "spatial"]
SpatialRelation = Literal[
    "intersects", "contains", "within", "touches", "crosses", "overlaps"
]


class SpatialQueryBuilder:
    """
    Generic spatial query builder for SSURGO data.

    Similar to soilDB::SDA_spatialQuery() in R, supports arbitrary input geometries
    with flexible table and return type options.
    """

    def __init__(self, client: Optional[SDAClient] = None):
        """
        Initialize spatial query builder.

        Args:
            client: Optional SDA client instance
        """
        self.client = client

    def query(
        self,
        geometry: GeometryInput,
        table: TableType = "mupolygon",
        return_type: ReturnType = "tabular",
        spatial_relation: SpatialRelation = "intersects",
        what: Optional[str] = None,
        geom_column: Optional[str] = None,
    ) -> Query:
        """
        Build a spatial query for SSURGO data.

        Args:
            geometry: Input geometry as WKT string, shapely geometry, or bbox dict
            table: Target SSURGO table name
            return_type: Whether to return 'tabular' or 'spatial' data
            spatial_relation: Spatial relationship to test
            what: Custom selection of columns (defaults based on table/return_type)
            geom_column: Custom geometry column name (defaults based on table)

        Returns:
            Query object ready for execution

        Examples:
            # Get map unit info intersecting a point
            >>> query = builder.query("POINT(-94.68 42.03)", "mupolygon", "tabular")

            # Get spatial polygons within a bounding box
            >>> bbox = {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}
            >>> query = builder.query(bbox, "mupolygon", "spatial")

            # Custom selection from survey areas
            >>> query = builder.query(polygon_wkt, "sapolygon", "tabular",
            ...                      what="areasymbol, areaname, areaacres")
        """
        # Validate table against TABLE_ALIASES
        if table not in TABLE_ALIASES:
            raise ValueError(
                f"Unknown table: '{table}'. Supported spatial tables: {list(TABLE_ALIASES.keys())}"
            )

        # Convert geometry to WKT if needed
        wkt_geom = self._geometry_to_wkt(geometry)

        # Get default columns and geometry column for the table
        if what is None:
            what = self._get_default_columns(table, return_type)
        if geom_column is None:
            geom_column = self._get_geometry_column(table)

        # For tabular queries, use efficient UDFs when available
        if return_type == "tabular" and self._can_use_udf(table, what):
            query = self._build_udf_query(table, wkt_geom, what)
        else:
            # Use regular spatial join approach
            query = Query()

            # For tabular results, use DISTINCT to avoid duplicates unless geometry keys are included
            if return_type == "tabular":
                # Check if geometry-specific keys are included (like mupolygonkey, sapolygonkey)
                has_geom_keys = any(
                    key in what.lower()
                    for key in ["mupolygonkey", "sapolygonkey", "featkey"]
                )
                # Only apply DISTINCT for default column selections to avoid ambiguity issues
                is_default_columns = what == self._get_default_columns(
                    table, return_type
                )
                if not has_geom_keys and is_default_columns:
                    # Add DISTINCT to the select clause
                    query._select_clause = f"DISTINCT {what}"
                else:
                    query.select(*[col.strip() for col in what.split(",")])
            else:
                # For spatial queries, include geometry column converted to WKT
                select_columns = [col.strip() for col in what.split(",")]
                if geom_column:
                    # Check if geometry is already included in custom what clause
                    has_geometry = any(
                        "geometry" in col.lower()
                        or "wkt" in col.lower()
                        or "geom" in col.lower()
                        or "shape" in col.lower()
                        for col in select_columns
                    )
                    if not has_geometry:
                        # Add geometry column as WKT with alias 'geometry'
                        select_columns.append(f"{geom_column}.STAsText() AS geometry")
                query.select(*select_columns)

            # Handle table aliases and joins for complex queries
            if table not in TABLE_ALIASES:
                raise ValueError(
                    f"Unknown table: '{table}'. Supported spatial tables: {list(TABLE_ALIASES.keys())}"
                )
            alias = TABLE_ALIASES[table]
            from_clause = f"{table} {alias}"

            query.from_(from_clause)
            for target, on in spatial_table_joins(table):
                query.inner_join(target, on)

            # Add spatial filter
            if geom_column:
                spatial_predicate = self._get_spatial_predicate(spatial_relation)
                spatial_filter = f"{geom_column}.{spatial_predicate}(geometry::STGeomFromText('{wkt_geom}', 4326)) = 1"
                query.where(spatial_filter)

        return query

    def _geometry_to_wkt(self, geometry: GeometryInput) -> str:
        """Convert various geometry inputs to WKT string."""
        if isinstance(geometry, str):
            # Assume it's already WKT
            return validate_wkt_geometry(geometry)
        elif isinstance(geometry, dict):
            # Assume it's a bounding box
            if all(k in geometry for k in ["xmin", "ymin", "xmax", "ymax"]):
                xmin, ymin = geometry["xmin"], geometry["ymin"]
                xmax, ymax = geometry["xmax"], geometry["ymax"]
                return f"POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"
            else:
                raise ValueError(
                    "Dictionary geometry must contain xmin, ymin, xmax, ymax keys"
                )
        else:
            # Try to use shapely if available
            try:
                if hasattr(geometry, "wkt"):
                    return str(geometry.wkt)  # type: ignore
                elif hasattr(geometry, "__geo_interface__"):
                    # Convert from GeoJSON-like interface to WKT
                    from shapely import geometry as geom  # type: ignore[import-untyped]

                    shape = geom.shape(geometry.__geo_interface__)
                    return str(shape.wkt)  # type: ignore
                else:
                    raise ValueError("Unsupported geometry type")
            except ImportError:
                raise ValueError(
                    "Shapely is required for non-string geometry inputs"
                ) from None

    def _get_default_columns(self, table: TableType, return_type: ReturnType) -> str:
        """Get default column selection for a table and return type.

        Delegates to ssurgo_tables.default_columns() for bare column names,
        then applies table aliases based on join logic.
        """
        # Map (table, column) → alias for applying to column names
        # Alias strings come from ssurgo_tables.TABLE_ALIASES
        column_alias_map = {
            "legend": {
                "lkey": TABLE_ALIASES["legend"],
                "areasymbol": TABLE_ALIASES["legend"],
                "areaname": TABLE_ALIASES["legend"],
                "mlraoffice": TABLE_ALIASES["legend"],
                "areaacres": TABLE_ALIASES["legend"],
            },
            "mapunit": {
                "mukey": TABLE_ALIASES["mapunit"],
                "musym": TABLE_ALIASES["mapunit"],
                "muname": TABLE_ALIASES["mapunit"],
                "mukind": TABLE_ALIASES["mapunit"],
                "muacres": TABLE_ALIASES["mapunit"],
            },
            "mupolygon": {
                "mukey": TABLE_ALIASES["mupolygon"],
                "musym": TABLE_ALIASES["mapunit"],
                "muname": TABLE_ALIASES["mapunit"],
                "mukind": TABLE_ALIASES["mapunit"],
                "areasymbol": TABLE_ALIASES["legend"],
                "areaname": TABLE_ALIASES["legend"],
            },
            "sapolygon": {
                "areasymbol": TABLE_ALIASES["sapolygon"],
                "spatialversion": TABLE_ALIASES["sapolygon"],
                "lkey": TABLE_ALIASES["sapolygon"],
            },
            "mupoint": {
                "mukey": TABLE_ALIASES["mupoint"],
                "musym": TABLE_ALIASES["mapunit"],
                "muname": TABLE_ALIASES["mapunit"],
            },
            "muline": {
                "mukey": TABLE_ALIASES["muline"],
                "musym": TABLE_ALIASES["mapunit"],
                "muname": TABLE_ALIASES["mapunit"],
            },
            "featpoint": {
                "featkey": TABLE_ALIASES["featpoint"],
                "featsym": TABLE_ALIASES["featpoint"],
            },
            "featline": {
                "featkey": TABLE_ALIASES["featline"],
                "featsym": TABLE_ALIASES["featline"],
            },
        }

        # Get bare column names from consolidated metadata
        bare_cols = get_default_columns_bare(table)
        if bare_cols is None:
            return "*"

        # Apply aliases based on join logic
        alias_map = column_alias_map.get(table, {})
        aliased_cols = []
        for col in bare_cols:
            alias = alias_map.get(col)
            if alias:
                aliased_cols.append(f"{alias}.{col}")
            else:
                # Fallback for columns without known alias (shouldn't happen)
                aliased_cols.append(col)

        return ", ".join(aliased_cols)

    def _get_geometry_column(self, table: TableType) -> str:
        """Get the geometry column name for a table."""
        # legend and mapunit don't have geometry columns
        if table in ["legend", "mapunit"]:
            raise ValueError(
                f"Table '{table}' does not have spatial data. Use a spatial table like 'mupolygon' or 'sapolygon'."
            )

        if table not in TABLE_ALIASES:
            raise ValueError(
                f"Unknown table: '{table}'. Supported spatial tables: {list(TABLE_ALIASES.keys())}"
            )

        # Get the bare geometry column name from ssurgo_tables
        geom_col_bare = geometry_column(table)
        if geom_col_bare is None:
            raise ValueError(f"Unknown table: {table}")

        # Prefix with the table alias from TABLE_ALIASES
        alias = TABLE_ALIASES[table]

        return f"{alias}.{geom_col_bare}"

    def _can_use_udf(self, table: TableType, what: str) -> bool:
        """Check if we can use UDFs for efficient tabular queries."""
        # UDFs are available for mupolygon and sapolygon tables with default tabular columns
        if table not in ["mupolygon", "sapolygon"]:
            return False

        # Get the expected default columns for this table
        expected_what = self._get_default_columns(table, "tabular")
        return what == expected_what

        return False

    def _build_udf_query(self, table: TableType, wkt_geom: str, what: str) -> Query:
        """Build efficient UDF-based query for tabular results."""
        if table == "mupolygon":
            # Use SDA_Get_Mukey_from_intersection_with_WktWgs84 UDF
            # Build CTE to get intersecting mukeys, then join to get attributes
            udf_sql = f"""
            WITH geom_data AS (
                SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt_geom}')
            )
            SELECT g.mukey, l.areasymbol, m.musym, m.nationalmusym, m.muname, m.mukind
            FROM geom_data g
            INNER JOIN mapunit m ON g.mukey = m.mukey
            INNER JOIN legend l ON m.lkey = l.lkey
            """
            query = Query.from_sql(udf_sql)

        elif table == "sapolygon":
            # Use SDA_Get_Sapolygonkey_from_intersection_with_WktWgs84 UDF
            udf_sql = f"""
            WITH geom_data AS (
                SELECT DISTINCT sapolygonkey, areasymbol, spatialversion, lkey FROM sapolygon
                WHERE sapolygonkey IN (
                    SELECT DISTINCT sapolygonkey FROM SDA_Get_Sapolygonkey_from_intersection_with_WktWgs84('{wkt_geom}')
                )
            )
            SELECT areasymbol, spatialversion, lkey
            FROM geom_data
            """
            query = Query.from_sql(udf_sql)

        return query

    def _get_spatial_predicate(self, relation: SpatialRelation) -> str:
        """Get SQL Server spatial predicate method."""
        predicates = {
            "intersects": "STIntersects",
            "contains": "STContains",
            "within": "STWithin",
            "touches": "STTouches",
            "crosses": "STCrosses",
            "overlaps": "STOverlaps",
        }
        return predicates.get(relation, "STIntersects")


@add_sync_version
async def spatial_query(
    geometry: GeometryInput,
    table: TableType = "mupolygon",
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    what: Optional[str] = None,
    geom_column: Optional[str] = None,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Execute a spatial query against SSURGO data (CANONICAL FUNCTION).

    This is the primary and recommended spatial query function for py-soildb.
    It is similar to soilDB::SDA_spatialQuery() in R and supports arbitrary input
    geometries with flexible table and return type options.

    **WHEN TO USE THIS:**
    - You need to query spatial features (map units, survey areas, features)
    - You have point, polygon, line, or bounding box geometry
    - You want tabular or spatial results
    - You prefer a single unified function for all spatial queries

    **DESIGN:**
    - Canonical function: ALL spatial queries should use this function
    - Supports multiple geometry input formats (WKT, shapely, bbox dict)
    - Flexible table selection (mupolygon, sapolygon, featpoint, featline)
    - Flexible return types (tabular for analysis, spatial with geometry)
    - Configurable spatial relationships (intersects, contains, within, etc.)

    **PERFORMANCE NOTES:**
    - Tabular queries use optimized UDFs when possible (faster than spatial joins)
    - Spatial queries with geometry return large result sets; use selective bounding boxes
    - Complex geometries (many vertices) may timeout; simplify when possible
    - Point queries are fastest; polygon queries scale with geometry complexity
    - Bounding box queries are fast and recommended for large areas

    **TABLE TYPES:**
    - "mupolygon": Map unit polygons (most common)
    - "sapolygon": Survey area polygons
    - "mupoint": Map unit points
    - "muline": Map unit lines
    - "featpoint": Feature points (e.g., soil pits, monuments)
    - "featline": Feature lines (e.g., linear features)
    - "mapunit": Map unit table (spatial join required)
    - "legend": Legend table (for legend-based queries)

    **SPATIAL RELATIONSHIPS:**
    - "intersects": Overlaps or touches (default, most common)
    - "contains": Query geometry contains SSURGO feature
    - "within": Query geometry within SSURGO feature
    - "touches": Features share a boundary but don't overlap
    - "crosses": Features cross each other
    - "overlaps": Features overlap but one doesn't contain the other

    **RETURN TYPES:**
    - "tabular": Returns attribute data as DataFrame (lighter weight, faster)
    - "spatial": Returns with geometry column (heavier, for mapping/analysis)

    **GEOMETRY INPUT FORMATS:**
    - WKT string: "POINT(-94.68 42.03)" or "POLYGON((-94.7 42.0, ...))"
    - Shapely geometry: geometry.Point(-94.68, 42.03)
    - Bounding box dict: {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}

    Args:
        geometry: Input geometry as WKT string, shapely geometry object, or bbox dict
        table: Target SSURGO table (default: "mupolygon"). See TABLE TYPES above.
        return_type: "tabular" for data only (default), "spatial" for with geometry
        spatial_relation: Type of spatial relationship to test (default: "intersects")
        what: Custom comma-separated column list (e.g., "mukey,muname,musym").
              If None, uses sensible defaults based on table and return_type.
        geom_column: Custom geometry column name (e.g., "geometry", "the_geom").
                     If None, auto-detected based on table.
        client: Optional SDA client instance. If not provided, one is created and closed
                automatically. Use `.sync()` to call this function from synchronous code.

    Returns:
        SDAResponse: Query results with methods like .to_pandas(), .to_geopandas()

    Raises:
        ValueError: If geometry cannot be parsed

    Examples:
        # Example 1: Get map unit info for a point (simplest case)
        ```python
        point_wkt = "POINT(-94.6859 42.0285)"
        response = await spatial_query(point_wkt, "mupolygon", "tabular")
        df = response.to_pandas()
        print(df[['mukey', 'muname', 'musym']])
        ```

        # Example 2: Get spatial polygons in a bounding box (for mapping)
        ```python
        bbox = {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}
        response = await spatial_query(bbox, "mupolygon", "spatial")
        gdf = response.to_geopandas()
        gdf.plot()  # Visualize map units
        ```

        # Example 3: Get survey areas intersecting a custom polygon
        ```python
        polygon_wkt = "POLYGON((-94.7 42.0, -94.6 42.0, -94.6 42.1, -94.7 42.1, -94.7 42.0))"
        response = await spatial_query(polygon_wkt, "sapolygon", "tabular")
        df = response.to_pandas()
        print(df[['areasymbol', 'areaname', 'areaacres']])
        ```

        # Example 4: Use shapely geometry
        ```python
        from shapely.geometry import Point
        location = Point(-94.68, 42.03)
        response = await spatial_query(location, "mupolygon", "tabular")
        ```

        # Example 5: Get feature points within area (archeological/historical sites)
        ```python
        response = await spatial_query(point_wkt, "featpoint", "spatial")
        gdf = response.to_geopandas()
        ```

        # Example 6: Custom columns and spatial relationship
        ```python
        response = await spatial_query(
            bbox,
            table="mupolygon",
            return_type="tabular",
            spatial_relation="within",
            what="mukey,muname,musym,compname,comppct_r"
        )
        ```

    **COMPARISON WITH CONVENIENCE FUNCTIONS:**

    For common use cases, consider using these instead:
    - `point_query(lat, lon, table, return_type)` - Simplified point queries
    - `bbox_query(xmin, ymin, xmax, ymax, table, return_type)` - Simplified bbox queries

    **DEPRECATION NOTE:**
    - SpatialQueryBuilder is an internal implementation detail; use spatial_query() instead.

    **SEE ALSO:**
    - point_query() - Simplified point-based queries
    - bbox_query() - Simplified bounding box queries
    - Query builder (query.py) - For non-spatial attribute queries
    - soilDB R package documentation - For additional spatial query patterns
    """
    client = require_client(client)

    builder = SpatialQueryBuilder(client)
    query = builder.query(
        geometry, table, return_type, spatial_relation, what, geom_column
    )

    return await client.execute(query)


# ============================================================================
# HELPER FUNCTIONS - Simplified API for common use cases
# ============================================================================


@add_sync_version
async def point_query(
    latitude: float,
    longitude: float,
    table: TableType = "mupolygon",
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Simplified point-based spatial query.

    Convenience wrapper for point locations. Use this instead of spatial_query()
    when you have latitude/longitude coordinates.

    **WHEN TO USE THIS:**
    - You have a single (lat, lon) coordinate
    - You want tabular or spatial results
    - You want simpler syntax than spatial_query()

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        table: Target SSURGO table (default: "mupolygon")
        return_type: "tabular" for data only (default), "spatial" for with geometry
        spatial_relation: Spatial relationship to test (default: "intersects")
        client: Optional SDA client instance

    Returns:
        SDAResponse: Query results

    Examples:
        ```python
        # Get map unit info at a location
        response = await point_query(latitude=42.0, longitude=-93.6)
        df = response.to_pandas()

        # Get spatial features for mapping
        response = await point_query(42.0, -93.6, "mupolygon", "spatial")
        gdf = response.to_geopandas()
        ```

    See Also:
        spatial_query() - For full control over geometry and parameters
        bbox_query() - For bounding box queries
    """
    client = require_client(client)

    point_wkt = f"POINT({longitude} {latitude})"
    return await spatial_query(
        point_wkt, table, return_type, spatial_relation, client=client
    )


@add_sync_version
async def bbox_query(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    table: TableType = "mupolygon",
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Simplified bounding box-based spatial query.

    Convenience wrapper for bounding box coordinates. Use this instead of
    spatial_query() when you have a rectangular area.

    **WHEN TO USE THIS:**
    - You have a bounding box (xmin, ymin, xmax, ymax)
    - You want faster queries than complex polygons
    - You want tabular or spatial results
    - You want simpler syntax than spatial_query()

    Args:
        xmin: Western boundary (longitude)
        ymin: Southern boundary (latitude)
        xmax: Eastern boundary (longitude)
        ymax: Northern boundary (latitude)
        table: Target SSURGO table (default: "mupolygon")
        return_type: "tabular" for data only (default), "spatial" for with geometry
        spatial_relation: Spatial relationship to test (default: "intersects")
        client: Optional SDA client instance

    Returns:
        SDAResponse: Query results

    Examples:
        ```python
        # Get map units in a region
        response = await bbox_query(-94.7, 42.0, -94.6, 42.1)
        df = response.to_pandas()

        # Get spatial polygons for mapping
        response = await bbox_query(-94.7, 42.0, -94.6, 42.1, return_type="spatial")
        gdf = response.to_geopandas()
        gdf.plot()
        ```

    See Also:
        spatial_query() - For full control over geometry and parameters
        point_query() - For point-based queries
    """
    client = require_client(client)

    bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    return await spatial_query(
        bbox, table, return_type, spatial_relation, client=client
    )
