"""
SQL query building classes for SDA queries.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .sanitization import (
    sanitize_sql_string,
    sanitize_sql_numeric,
    sanitize_sql_string_list,
    validate_sql_object_name,
    VALID_COLUMNS,
    VALID_TABLES,
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
        "texture",
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
    """Builder for SQL queries against Soil Data Access."""

    def __init__(self) -> None:
        self._raw_sql: Optional[str] = None
        self._select_clause: str = "*"
        self._from_clause: str = ""
        self._where_conditions: List[str] = []
        self._join_clauses: List[str] = []
        self._order_by_clause: Optional[str] = None
        self._limit_count: Optional[int] = None

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

    def to_sql(self) -> str:
        """Build the SQL query string.

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

        # Add WHERE conditions
        if self._where_conditions:
            sql += " WHERE " + " AND ".join(self._where_conditions)

        # Add ORDER BY
        if self._order_by_clause:
            sql += f" ORDER BY {self._order_by_clause}"

        return sql


class SpatialQuery(BaseQuery):
    """Builder for spatial queries with geometry filters."""

    def __init__(self) -> None:
        self._base_query = Query()
        self._geometry_filter: Optional[str] = None
        self._spatial_relationship: str = "STIntersects"

    def select(self, *columns: str) -> "SpatialQuery":
        """Set the SELECT clause.

        Args:
            *columns: Column names to select. Use "*" for all columns.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.select(*columns)
        return self

    def from_(self, table: str) -> "SpatialQuery":
        """Set the FROM clause.

        Args:
            table: Name of the table to query from.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.from_(table)
        return self

    def where(self, condition: str) -> "SpatialQuery":
        """Add a WHERE condition.

        Args:
            condition: SQL WHERE condition string.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.where(condition)
        return self

    def inner_join(self, table: str, on_condition: str) -> "SpatialQuery":
        """Add an INNER JOIN clause.

        Args:
            table: Name of the table to join.
            on_condition: JOIN condition (ON clause).

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.inner_join(table, on_condition)
        return self

    def left_join(self, table: str, on_condition: str) -> "SpatialQuery":
        """Add a LEFT JOIN clause.

        Args:
            table: Name of the table to join.
            on_condition: JOIN condition (ON clause).

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.left_join(table, on_condition)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> "SpatialQuery":
        """Set the ORDER BY clause.

        Args:
            column: Column name to order by.
            direction: Sort direction ("ASC" or "DESC").

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.order_by(column, direction)
        return self

    def limit(self, count: int) -> "SpatialQuery":
        """Set the LIMIT.

        Args:
            count: Maximum number of rows to return.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._base_query.limit(count)
        return self

    def intersects_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> "SpatialQuery":
        """Add a bounding box intersection filter.

        Args:
            min_x: Minimum longitude (west bound).
            min_y: Minimum latitude (south bound).
            max_x: Maximum longitude (east bound).
            max_y: Maximum latitude (north bound).

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        bbox_wkt = f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
        self._geometry_filter = bbox_wkt
        self._spatial_relationship = "STIntersects"
        return self

    def contains_point(self, x: float, y: float) -> "SpatialQuery":
        """Add a point containment filter.

        Args:
            x: Longitude of the point.
            y: Latitude of the point.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        point_wkt = f"POINT({x} {y})"
        self._geometry_filter = point_wkt
        self._spatial_relationship = "STContains"
        return self

    def intersects_geometry(self, wkt: str) -> "SpatialQuery":
        """Add a geometry intersection filter using WKT.

        Args:
            wkt: Well-Known Text representation of the geometry.

        Returns:
            SpatialQuery: This SpatialQuery instance for method chaining.
        """
        self._geometry_filter = wkt
        self._spatial_relationship = "STIntersects"
        return self

    def to_sql(self) -> str:
        """Build the spatial SQL query string.

        Returns:
            str: The complete SQL query string with spatial filters applied.
        """
        base_sql = self._base_query.to_sql()

        if self._geometry_filter:
            spatial_condition = (
                f"mupolygongeo.{self._spatial_relationship}"
                f"(geometry::STGeomFromText('{self._geometry_filter}', 4326)) = 1"
            )

            if " WHERE " in base_sql:
                # Insert spatial condition at the beginning of WHERE clause
                base_sql = base_sql.replace(
                    " WHERE ", f" WHERE {spatial_condition} AND ", 1
                )
            else:
                base_sql += f" WHERE {spatial_condition}"

        return base_sql


# Predefined query builders for common operations
class QueryBuilder:
    """Factory class for common SDA query patterns."""

    @staticmethod
    def mapunits_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """Get map units for a survey area by legend/area symbol."""
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

    @staticmethod
    def components_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """Get components for a survey area."""
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

    @staticmethod
    def component_horizons_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> Query:
        """Get component and horizon data for a survey area."""
        if columns is None:
            columns = [
                "m.mukey",
                "m.musym",
                "m.muname",
                "c.cokey",
                "c.compname",
                "c.comppct_r",
            ] + ColumnSets.CHORIZON_TEXTURE

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

    @staticmethod
    def components_at_point(
        longitude: float, latitude: float, columns: Optional[List[str]] = None
    ) -> SpatialQuery:
        """Get soil component data at a specific point."""
        if columns is None:
            columns = [
                "m.mukey",
                "m.musym",
                "m.muname",
                "c.compname",
                "c.comppct_r",
            ] + ColumnSets.CHORIZON_TEXTURE

        return (
            SpatialQuery()
            .select(*columns)
            .from_("mupolygon p")
            .inner_join("mapunit m", "p.mukey = m.mukey")
            .inner_join("component c", "m.mukey = c.mukey")
            .inner_join("chorizon h", "c.cokey = h.cokey")
            .contains_point(longitude, latitude)
            .where("c.majcompflag = 'Yes'")
            .order_by("c.comppct_r DESC, h.hzdept_r")
        )

    @staticmethod
    def spatial_by_legend(
        areasymbol: str, columns: Optional[List[str]] = None
    ) -> SpatialQuery:
        """Get spatial data for map units on a legend/area symbol."""
        if columns is None:
            columns = ColumnSets.MAPUNIT_SPATIAL + [
                "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STArea() as shape_area",
                "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STLength() as shape_length",
            ]

        return (
            SpatialQuery()
            .select(*columns)
            .from_("mupolygon")
            .where(f"areasymbol = {sanitize_sql_string(areasymbol)}")
        )

    @staticmethod
    def mapunits_intersecting_bbox(
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        columns: Optional[List[str]] = None,
    ) -> SpatialQuery:
        """Get map units that intersect with a bounding box."""
        if columns is None:
            columns = [
                "m.mukey",
                "m.musym",
                "m.muname",
                "mupolygongeo.STAsText() as geometry",
            ]

        return (
            SpatialQuery()
            .select(*columns)
            .from_("mupolygon p")
            .inner_join("mapunit m", "p.mukey = m.mukey")
            .intersects_bbox(min_x, min_y, max_x, max_y)
        )

    @staticmethod
    def available_survey_areas(
        columns: Optional[List[str]] = None, table: str = "sacatalog"
    ) -> Query:
        """Get list of available survey areas."""
        if columns is None:
            if table == "sacatalog":
                columns = ["areasymbol", "areaname", "saversion"]
            else:
                columns = ColumnSets.LEGEND_BASIC

        return Query().select(*columns).from_(table).order_by("areasymbol")

    @staticmethod
    def survey_area_boundaries(
        columns: Optional[List[str]] = None, table: str = "sapolygon"
    ) -> SpatialQuery:
        """Get survey area boundary polygons."""
        if columns is None:
            columns = ["areasymbol", "areaname", "sapolygongeo.STAsText() as geometry"]

        return SpatialQuery().select(*columns).from_(table)

    @staticmethod
    def from_sql(query: str) -> Query:
        """
        Create a query from a raw SQL string.

        Args:
            query: The raw SQL query string.

        Returns:
            A Query object.
        """
        return Query.from_sql(query)

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
            Query object ready for execution
        """
        if columns is None:
            columns = ColumnSets.PEDON_BASIC + ["corr_name", "samp_name"]

        # Validate column and table names
        validate_sql_object_name(lon_column)
        validate_sql_object_name(lat_column)
        validate_sql_object_name(base_table)

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
                alias = f"t{i}"
                # Most pedon-related tables join on pedon_key
                query = query.left_join(
                    f"{table} {alias}", f"p.pedon_key = {alias}.pedon_key"
                )

        return query

    @staticmethod
    def pedon_horizons_by_pedon_keys(
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
            Query object ready for execution
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
        validate_sql_object_name(base_table)

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

    @staticmethod
    def pedon_by_pedon_key(
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
            Query object ready for execution
        """
        if columns is None:
            columns = ColumnSets.PEDON_BASIC

        # Validate table name
        validate_sql_object_name(base_table)

        query = (
            Query()
            .select(*columns)
            .from_(f"{base_table} p")
            .where(f"p.pedon_key = {sanitize_sql_string(pedon_key)}")
        )

        # Add joins for related tables
        if related_tables:
            for i, table in enumerate(related_tables):
                alias = f"t{i}"
                # Most pedon-related tables join on pedon_key
                query = query.left_join(
                    f"{table} {alias}", f"p.pedon_key = {alias}.pedon_key"
                )

        return query
