"""
SQL query building classes for SDA queries.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .sanitization import (
    sanitize_sql_numeric,
    sanitize_sql_string,
    sanitize_sql_string_list,
    validate_sql_identifier,
)


# Standard column sets for common query patterns
class ColumnSets:
    """Standardized column sets for common SDA query patterns."""

    # Map unit columns
    MAPUNIT_BASIC = ["mukey", "musym", "muname", "mukind", "muacres"]
    MAPUNIT_DETAILED = MAPUNIT_BASIC + [
        "mustatus",
        "muhelcl",
        "muwathelcl",
        "muwndhelcl",
        "interpfocus",
        "invesintens",
    ]
    MAPUNIT_SPATIAL = [
        "mukey",
        "musym",
        "muname",
        "mupolygongeo.STAsText() as geometry",
    ]

    # Component columns
    COMPONENT_BASIC = ["cokey", "compname", "comppct_r", "majcompflag"]
    COMPONENT_DETAILED = COMPONENT_BASIC + [
        "compkind",
        "localphase",
        "drainagecl",
        "geomdesc",
        "taxclname",
        "taxorder",
        "taxsuborder",
        "taxgrtgroup",
        "taxsubgrp",
        "taxpartsize",
        "taxpartsizemod",
        "taxceactcl",
        "taxreaction",
        "taxtempcl",
        "taxmoistscl",
        "tempregime",
        "taxminalogy",
        "taxother",
    ]

    # Horizon columns
    CHORIZON_BASIC = ["chkey", "hzname", "hzdept_r", "hzdepb_r"]
    CHORIZON_TEXTURE = CHORIZON_BASIC + [
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        # Note: "texture" column not available on chorizon table
        # Texture information is stored in chtexture/chtexturegrp tables
    ]
    CHORIZON_CHEMICAL = CHORIZON_BASIC + [
        "ph1to1h2o_r",
        "om_r",
        "caco3_r",
        "gypsum_r",
        "sar_r",
        "cec7_r",
        "ecec_r",
    ]
    CHORIZON_PHYSICAL = CHORIZON_BASIC + [
        "dbthirdbar_r",
        "dbovendry_r",
        "ksat_r",
        "awc_r",
        "wfifteenbar_r",
        "wthirdbar_r",
        "wtenthbar_r",
    ]
    CHORIZON_DETAILED = (
        CHORIZON_BASIC
        + CHORIZON_TEXTURE[4:]
        + CHORIZON_CHEMICAL[4:]
        + CHORIZON_PHYSICAL[4:]
    )

    # Legend/Survey Area columns
    LEGEND_BASIC = ["lkey", "areasymbol", "areaname", "saversion"]
    LEGEND_DETAILED = LEGEND_BASIC + [
        "mlraoffice",
        "projectscale",
        "cordate",
        "saverest",
    ]

    # Pedon/Site columns
    PEDON_BASIC = [
        "pedon_key",
        "upedonid",
        "latitude_decimal_degrees",
        "longitude_decimal_degrees",
    ]
    PEDON_SITE = PEDON_BASIC + [
        "samp_name",
        "corr_name",
        "site_key",
        "usiteid",
        "site_obsdate",
    ]
    PEDON_DETAILED = PEDON_SITE + [
        "descname",
        "taxonname",
        "taxclname",
        "pedlabsampnum",
        "pedoniid",
    ]

    # Lab horizon columns
    LAB_HORIZON_BASIC = [
        "layer_key",
        "layer_sequence",
        "hzn_top",
        "hzn_bot",
        "hzn_desgn",
    ]
    LAB_HORIZON_TEXTURE = LAB_HORIZON_BASIC + [
        "sand_total",
        "silt_total",
        "clay_total",
        "texture_lab",
    ]
    LAB_HORIZON_CHEMICAL = LAB_HORIZON_BASIC + [
        "ph_h2o",
        "organic_carbon_walkley_black",
        "total_carbon_ncs",
        "caco3_lt_2_mm",
    ]
    LAB_HORIZON_PHYSICAL = LAB_HORIZON_BASIC + [
        "bulk_density_third_bar",
        "le_third_fifteen_lt2_mm",
        "water_retention_third_bar",
        "water_retention_15_bar",
    ]
    LAB_HORIZON_CALCULATIONS = [
        "estimated_om",
        "estimated_c_tot",
        "estimated_n_tot",
        "estimated_sand",
        "estimated_silt",
        "estimated_clay",
    ]
    LAB_HORIZON_ROSETTA = ["theta_r", "theta_s", "alpha", "npar", "ksat", "ksat_class"]
    LAB_HORIZON_DETAILED = (
        LAB_HORIZON_BASIC
        + LAB_HORIZON_TEXTURE[5:]
        + LAB_HORIZON_CHEMICAL[5:]
        + LAB_HORIZON_PHYSICAL[5:]
        + LAB_HORIZON_CALCULATIONS
        + LAB_HORIZON_ROSETTA
    )


class BaseQuery(ABC):
    """Base class for SDA queries."""

    @abstractmethod
    def to_sql(self) -> str:
        """Convert the query to SQL string.

        Returns:
            str: The SQL query string representation.
        """
        pass


class Query(BaseQuery):
    """Builder for SQL queries against Soil Data Access.

    Unified query builder supporting both regular SQL queries and spatial queries.

    Spatial queries can be constructed by chaining spatial filter methods:
    - intersects_bbox(): Filter by bounding box intersection
    - contains_point(): Filter by point containment
    - intersects_geometry(): Filter by geometry intersection using WKT

    Examples:
        # Regular query
        query = Query().select("mukey", "muname").from_("mapunit").where("areasymbol = 'IA109'")

        # Spatial query (same Query class!)
        query = Query().select("mukey").from_("mupolygon").contains_point(-93.5, 42.5)
    """

    def __init__(self) -> None:
        self._raw_sql: Optional[str] = None
        self._select_clause: str = "*"
        self._from_clause: str = ""
        self._where_conditions: List[str] = []
        self._join_clauses: List[str] = []
        self._order_by_clause: Optional[str] = None
        self._limit_count: Optional[int] = None
        # Spatial query support
        self._geometry_filter: Optional[str] = None
        self._spatial_relationship: str = "STIntersects"

    @classmethod
    def from_sql(cls, sql: str) -> "Query":
        """Create a query from raw SQL.

        Args:
            sql: Raw SQL query string.

        Returns:
            Query: A new Query instance with the provided SQL.
        """
        query = cls()
        query._raw_sql = sql
        return query

    def select(self, *columns: str) -> "Query":
        """Set the SELECT clause.

        Args:
            *columns: Column names to select. Use "*" for all columns.

        Returns:
            Query: This Query instance for method chaining.
        """
        if columns:
            self._select_clause = ", ".join(columns)
        return self

    def from_(self, table: str) -> "Query":
        """Set the FROM clause.

        Args:
            table: Name of the table to query from.

        Returns:
            Query: This Query instance for method chaining.
        """
        self._from_clause = table
        return self

    def where(self, condition: str) -> "Query":
        """Add a WHERE condition.

        Args:
            condition: SQL WHERE condition string.

        Returns:
            Query: This Query instance for method chaining.
        """
        self._where_conditions.append(condition)
        return self

    def join(self, table: str, on_condition: str, join_type: str = "INNER") -> "Query":
        """Add a JOIN clause.

        Args:
            table: Name of the table to join.
            on_condition: JOIN condition (ON clause).
            join_type: Type of join ("INNER", "LEFT", "RIGHT", "FULL").

        Returns:
            Query: This Query instance for method chaining.
        """
        self._join_clauses.append(f"{join_type} JOIN {table} ON {on_condition}")
        return self

    def inner_join(self, table: str, on_condition: str) -> "Query":
        """Add an INNER JOIN clause.

        Args:
            table: Name of the table to join.
            on_condition: JOIN condition (ON clause).

        Returns:
            Query: This Query instance for method chaining.
        """
        return self.join(table, on_condition, "INNER")

    def left_join(self, table: str, on_condition: str) -> "Query":
        """Add a LEFT JOIN clause.

        Args:
            table: Name of the table to join.
            on_condition: JOIN condition (ON clause).

        Returns:
            Query: This Query instance for method chaining.
        """
        return self.join(table, on_condition, "LEFT")

    def order_by(self, column: str, direction: str = "ASC") -> "Query":
        """Set the ORDER BY clause.

        Args:
            column: Column name to order by.
            direction: Sort direction ("ASC" or "DESC").

        Returns:
            Query: This Query instance for method chaining.
        """
        self._order_by_clause = f"{column} {direction}"
        return self

    def limit(self, count: int) -> "Query":
        """Set the LIMIT (uses TOP in SQL Server).

        Args:
            count: Maximum number of rows to return.

        Returns:
            Query: This Query instance for method chaining.
        """
        self._limit_count = count
        return self

    def intersects_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> "Query":
        """Add a bounding box intersection filter (spatial query).

        Args:
            min_x: Minimum longitude (west bound).
            min_y: Minimum latitude (south bound).
            max_x: Maximum longitude (east bound).
            max_y: Maximum latitude (north bound).

        Returns:
            Query: This Query instance for method chaining.
        """
        bbox_wkt = f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
        self._geometry_filter = bbox_wkt
        self._spatial_relationship = "STIntersects"
        return self

    def contains_point(self, x: float, y: float) -> "Query":
        """Add a point containment filter (spatial query).

        Args:
            x: Longitude of the point.
            y: Latitude of the point.

        Returns:
            Query: This Query instance for method chaining.
        """
        point_wkt = f"POINT({x} {y})"
        self._geometry_filter = point_wkt
        self._spatial_relationship = "STContains"
        return self

    def intersects_geometry(self, wkt: str) -> "Query":
        """Add a geometry intersection filter using WKT (spatial query).

        Args:
            wkt: Well-Known Text representation of the geometry.

        Returns:
            Query: This Query instance for method chaining.
        """
        self._geometry_filter = wkt
        self._spatial_relationship = "STIntersects"
        return self

    def to_sql(self) -> str:
        """Build the SQL query string.

        Supports both regular SQL queries and spatial queries with geometry filters.

        Returns:
            str: The complete SQL query string.
        """
        if self._raw_sql:
            return self._raw_sql

        # Build SELECT clause with TOP if limit is specified
        if self._limit_count:
            sql = f"SELECT TOP {self._limit_count} {self._select_clause}"
        else:
            sql = f"SELECT {self._select_clause}"

        # Add FROM clause
        if self._from_clause:
            sql += f" FROM {self._from_clause}"

        # Add JOIN clauses
        for join_clause in self._join_clauses:
            sql += f" {join_clause}"

        # Add WHERE conditions (including spatial filters if present)
        where_parts = list(self._where_conditions)

        if self._geometry_filter:
            from_clause = self._from_clause
            alias = None
            if " " in from_clause:
                alias = from_clause.split(" ")[-1]

            geom_column = "mupolygongeo"
            if alias:
                geom_column = f"{alias}.{geom_column}"

            spatial_condition = (
                f"{geom_column}.{self._spatial_relationship}"
                f"(geometry::STGeomFromText('{self._geometry_filter}', 4326)) = 1"
            )
            where_parts.insert(0, spatial_condition)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        # Add ORDER BY
        if self._order_by_clause:
            sql += f" ORDER BY {self._order_by_clause}"

        return sql


class SpatialQuery(Query):
    """Deprecated: Use Query class directly instead.

    SpatialQuery is now an alias for Query class. Both regular and spatial
    query methods are available on the Query class itself.

    All spatial methods (intersects_bbox, contains_point, intersects_geometry)
    are now available directly on Query objects:

        # Old way (still works)
        query = SpatialQuery().select("mukey").from_("mupolygon").contains_point(-93.5, 42.5)

        # New way (preferred)
        query = Query().select("mukey").from_("mupolygon").contains_point(-93.5, 42.5)

    This class is kept for backward compatibility only. It will be removed in a future version.
    """

    def __init__(self) -> None:
        """Initialize a SpatialQuery (deprecated alias for Query)."""
        super().__init__()




# Deprecated: Use query_templates module functions instead
class QueryBuilder:
    """
    DEPRECATED: Factory class for common SDA query patterns.

    This class is deprecated in favor of module-level functions in query_templates.
    Use the query_templates module for new code:

        from soildb.query_templates import (
            query_mapunits_by_legend,
            query_components_at_point,
            ...
        )

    This class is kept for backward compatibility and will be removed in a future version.
    All methods now delegate to the query_templates module functions.
    """

    @staticmethod
    def mapunits_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """DEPRECATED: Use query_templates.query_mapunits_by_legend() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.mapunits_by_legend() is deprecated. "
            "Use query_templates.query_mapunits_by_legend() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_mapunits_by_legend(areasymbol, columns)

    @staticmethod
    def components_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """DEPRECATED: Use query_templates.query_components_by_legend() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.components_by_legend() is deprecated. "
            "Use query_templates.query_components_by_legend() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_components_by_legend(areasymbol, columns)

    @staticmethod
    def component_horizons_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """DEPRECATED: Use query_templates.query_component_horizons_by_legend() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.component_horizons_by_legend() is deprecated. "
            "Use query_templates.query_component_horizons_by_legend() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_component_horizons_by_legend(areasymbol, columns)

    @staticmethod
    def components_at_point(
        longitude: float, latitude: float, columns: Optional[List[str]] = None
    ) -> SpatialQuery:
        """DEPRECATED: Use query_templates.query_components_at_point() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.components_at_point() is deprecated. "
            "Use query_templates.query_components_at_point() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_components_at_point(longitude, latitude, columns)

    @staticmethod
    def spatial_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> SpatialQuery:
        """DEPRECATED: Use query_templates.query_spatial_by_legend() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.spatial_by_legend() is deprecated. "
            "Use query_templates.query_spatial_by_legend() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_spatial_by_legend(areasymbol, columns)

    @staticmethod
    def mapunits_intersecting_bbox(
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        columns: Optional[List[str]] = None,
    ) -> SpatialQuery:
        """DEPRECATED: Use query_templates.query_mapunits_intersecting_bbox() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.mapunits_intersecting_bbox() is deprecated. "
            "Use query_templates.query_mapunits_intersecting_bbox() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_mapunits_intersecting_bbox(
            min_x, min_y, max_x, max_y, columns
        )

    @staticmethod
    def available_survey_areas(
        columns: Optional[List[str]] = None, table: str = "sacatalog"
    ) -> Query:
        """DEPRECATED: Use query_templates.query_available_survey_areas() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.available_survey_areas() is deprecated. "
            "Use query_templates.query_available_survey_areas() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_available_survey_areas(columns, table)

    @staticmethod
    def survey_area_boundaries(
        columns: Optional[List[str]] = None, table: str = "sapolygon"
    ) -> SpatialQuery:
        """DEPRECATED: Use query_templates.query_survey_area_boundaries() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.survey_area_boundaries() is deprecated. "
            "Use query_templates.query_survey_area_boundaries() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_survey_area_boundaries(columns, table)

    @staticmethod
    def from_sql(query: str) -> Query:
        """DEPRECATED: Use query_templates.query_from_sql() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.from_sql() is deprecated. "
            "Use query_templates.query_from_sql() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_from_sql(query)

    @staticmethod
    def pedons_intersecting_bbox(
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
        """DEPRECATED: Use query_templates.query_pedons_intersecting_bbox() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.pedons_intersecting_bbox() is deprecated. "
            "Use query_templates.query_pedons_intersecting_bbox() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_pedons_intersecting_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            columns,
            base_table,
            related_tables,
            lon_column,
            lat_column,
        )

    @staticmethod
    def pedon_horizons_by_pedon_keys(
        pedon_keys: List[str],
        columns: Optional[List[str]] = None,
        base_table: str = "lab_layer",
        related_tables: Optional[List[str]] = None,
    ) -> Query:
        """DEPRECATED: Use query_templates.query_pedon_horizons_by_pedon_keys() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.pedon_horizons_by_pedon_keys() is deprecated. "
            "Use query_templates.query_pedon_horizons_by_pedon_keys() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_pedon_horizons_by_pedon_keys(
            pedon_keys, columns, base_table, related_tables
        )

    @staticmethod
    def pedon_by_pedon_key(
        pedon_key: str,
        columns: Optional[List[str]] = None,
        base_table: str = "lab_combine_nasis_ncss",
        related_tables: Optional[List[str]] = None,
    ) -> Query:
        """DEPRECATED: Use query_templates.query_pedon_by_pedon_key() instead."""
        from . import query_templates
        import warnings

        warnings.warn(
            "QueryBuilder.pedon_by_pedon_key() is deprecated. "
            "Use query_templates.query_pedon_by_pedon_key() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return query_templates.query_pedon_by_pedon_key(
            pedon_key, columns, base_table, related_tables
        )
