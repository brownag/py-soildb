"""
Spatial queries for SSURGO data.

Query soil data using points, bounding boxes, and polygons.
Returns tabular data or spatial data with geometry.
"""

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from .client import SDAClient
from .query import Query
from .response import SDAResponse

if TYPE_CHECKING:
    try:
        from shapely.geometry.base import BaseGeometry
    except ImportError:
        BaseGeometry = Any

# Type aliases for clarity
GeometryInput = Union[str, "BaseGeometry", Dict[str, float]]
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
            if table == "mupolygon":
                query.from_("mupolygon p")
                query.inner_join("mapunit m", "p.mukey = m.mukey")
                query.inner_join("legend l", "m.lkey = l.lkey")
            elif table == "sapolygon":
                query.from_("sapolygon s")
            elif table == "featpoint":
                query.from_("featpoint fp")
            elif table == "featline":
                query.from_("featline fl")
            else:
                query.from_(table)

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
            return geometry
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
                    from shapely import geometry as geom

                    shape = geom.shape(geometry.__geo_interface__)
                    return str(shape.wkt)  # type: ignore
                else:
                    raise ValueError("Unsupported geometry type")
            except ImportError:
                raise ValueError(
                    "Shapely is required for non-string geometry inputs"
                ) from None

    def _get_default_columns(self, table: TableType, return_type: ReturnType) -> str:
        """Get default column selection for a table and return type."""

        # Common columns for different tables with proper aliases
        table_columns = {
            "legend": "l.lkey, l.areasymbol, l.areaname, l.mlraoffice, l.areaacres",
            "mapunit": "m.mukey, m.musym, m.muname, m.mukind, m.muacres",
            "mupolygon": "p.mukey, m.musym, m.muname, m.mukind, l.areasymbol, l.areaname",
            "sapolygon": "s.areasymbol, s.spatialversion, s.lkey",
            "mupoint": "pt.mukey, m.musym, m.muname",
            "muline": "ln.mukey, m.musym, m.muname",
            "featpoint": "fp.featkey, fp.featsym",
            "featline": "fl.featkey, fl.featsym",
        }

        base_columns = table_columns.get(table, "*")
        return base_columns

    def _get_geometry_column(self, table: TableType) -> str:
        """Get the geometry column name for a table."""
        geometry_columns = {
            "legend": None,  # No geometry in legend table
            "mapunit": None,  # No geometry in mapunit table
            "mupolygon": "p.mupolygongeo",
            "sapolygon": "s.sapolygongeo",
            "mupoint": "pt.mupointgeo",
            "muline": "ln.mulinegeo",
            "featpoint": "fp.featpointgeo",
            "featline": "fl.featlinegeo",
        }

        geom_col = geometry_columns.get(table)
        if geom_col is None and table in ["legend", "mapunit"]:
            raise ValueError(
                f"Table '{table}' does not have spatial data. Use a spatial table like 'mupolygon' or 'sapolygon'."
            )
        elif geom_col is None:
            raise ValueError(f"Unknown table: {table}")

        return geom_col

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
    Execute a generic spatial query against SSURGO data.

    This is the main spatial query function, similar to soilDB::SDA_spatialQuery() in R.
    Supports arbitrary input geometries with flexible table and return type options.

    Performance Notes:
    - Tabular queries use optimized UDFs when possible (faster than spatial joins)
    - Spatial queries with geometry return large result sets; use selective bounding boxes
    - Complex geometries (many vertices) may timeout; simplify when possible
    - Point queries are fastest; polygon queries scale with geometry complexity

    Args:
        geometry: Input geometry as WKT string, shapely geometry, or bbox dict
        table: Target SSURGO table name
        return_type: Whether to return 'tabular' or 'spatial' data
        spatial_relation: Spatial relationship to test
        what: Custom selection of columns (defaults based on table/return_type)
        geom_column: Custom geometry column name (defaults based on table)
        client: Optional SDA client instance

    Returns:
        SDAResponse with query results

    Examples:
        # Get map unit tabular data intersecting a point
        >>> point_wkt = "POINT(-94.6859 42.0285)"
        >>> response = await spatial_query(point_wkt, "mupolygon", "tabular")
        >>> df = response.to_pandas()

        # Get spatial polygons within a bounding box
        >>> bbox = {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}
        >>> response = await spatial_query(bbox, "mupolygon", "spatial")
        >>> gdf = response.to_geodataframe()

        # Get survey area info intersecting a polygon
        >>> polygon_wkt = "POLYGON((-94.7 42.0, -94.6 42.0, -94.6 42.1, -94.7 42.1, -94.7 42.0))"
        >>> response = await spatial_query(polygon_wkt, "sapolygon", "tabular")
    """
    from .convenience import _get_default_client

    if client is None:
        client = _get_default_client()

    builder = SpatialQueryBuilder(client)
    query = builder.query(
        geometry, table, return_type, spatial_relation, what, geom_column
    )

    return await client.execute(query)


# Convenience functions for specific table types
async def query_mupolygon(
    geometry: GeometryInput,
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Query map unit polygons with a geometry."""
    return await spatial_query(
        geometry, "mupolygon", return_type, spatial_relation, client=client
    )


async def query_sapolygon(
    geometry: GeometryInput,
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Query survey area polygons with a geometry."""
    return await spatial_query(
        geometry, "sapolygon", return_type, spatial_relation, client=client
    )


async def query_featpoint(
    geometry: GeometryInput,
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Query feature points with a geometry."""
    return await spatial_query(
        geometry, "featpoint", return_type, spatial_relation, client=client
    )


async def query_featline(
    geometry: GeometryInput,
    return_type: ReturnType = "tabular",
    spatial_relation: SpatialRelation = "intersects",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Query feature lines with a geometry."""
    return await spatial_query(
        geometry, "featline", return_type, spatial_relation, client=client
    )


# Bounding box convenience functions
async def mupolygon_in_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    return_type: ReturnType = "tabular",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Get map unit polygons in a bounding box."""
    bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    return await query_mupolygon(bbox, return_type, client=client)


async def sapolygon_in_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    return_type: ReturnType = "tabular",
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """Get survey area polygons in a bounding box."""
    bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    return await query_sapolygon(bbox, return_type, client=client)


# Compatibility function - keep the original get_mapunits_in_bbox name
async def get_mapunits_in_bbox(
    min_longitude: float,
    min_latitude: float,
    max_longitude: float,
    max_latitude: float,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Get map units within a bounding box (compatibility function).

    This maintains backward compatibility. For new code, consider using
    mupolygon_in_bbox() for more explicit naming.
    """
    return await mupolygon_in_bbox(
        min_longitude,
        min_latitude,
        max_longitude,
        max_latitude,
        return_type="tabular",
        client=client,
    )
