"""
SQL query building classes for SDA queries.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


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
    def mapunits_by_legend(areasymbol: str) -> Query:
        """Get map units for a survey area by legend/area symbol."""
        return (
            Query()
            .select(
                "m.mukey",
                "m.musym",
                "m.muname",
                "m.mukind",
                "m.muacres",
                "l.areasymbol",
                "l.areaname",
            )
            .from_("mapunit m")
            .inner_join("legend l", "m.lkey = l.lkey")
            .where(f"l.areasymbol = '{areasymbol}'")
            .order_by("m.musym")
        )

    @staticmethod
    def components_by_legend(areasymbol: str) -> Query:
        """Get components for a survey area."""
        return (
            Query()
            .select(
                "c.cokey",
                "c.compname",
                "c.comppct_r",
                "c.majcompflag",
                "m.mukey",
                "m.musym",
                "m.muname",
                "l.areasymbol",
            )
            .from_("component c")
            .inner_join("mapunit m", "c.mukey = m.mukey")
            .inner_join("legend l", "m.lkey = l.lkey")
            .where(f"l.areasymbol = '{areasymbol}'")
            .order_by("m.musym, c.comppct_r DESC")
        )

    @staticmethod
    def component_horizons_by_legend(areasymbol: str) -> Query:
        """Get component and horizon data for a survey area."""
        return (
            Query()
            .select(
                "m.mukey",
                "m.musym",
                "m.muname",
                "c.cokey",
                "c.compname",
                "c.comppct_r",
                "h.chkey",
                "h.hzname",
                "h.hzdept_r",
                "h.hzdepb_r",
                "h.sandtotal_r",
                "h.silttotal_r",
                "h.claytotal_r",
                "h.om_r",
                "h.ph1to1h2o_r",
            )
            .from_("mapunit m")
            .inner_join("legend l", "m.lkey = l.lkey")
            .inner_join("component c", "m.mukey = c.mukey")
            .inner_join("chorizon h", "c.cokey = h.cokey")
            .where(f"l.areasymbol = '{areasymbol}' AND c.majcompflag = 'Yes'")
            .order_by("m.musym, c.comppct_r DESC, h.hzdept_r")
        )

    @staticmethod
    def components_at_point(longitude: float, latitude: float) -> SpatialQuery:
        """Get soil component data at a specific point."""
        return (
            SpatialQuery()
            .select(
                "m.mukey",
                "m.musym",
                "m.muname",
                "c.compname",
                "c.comppct_r",
                "h.hzname",
                "h.hzdept_r",
                "h.hzdepb_r",
                "h.sandtotal_r",
                "h.silttotal_r",
                "h.claytotal_r",
                "h.om_r",
                "h.ph1to1h2o_r",
            )
            .from_("mupolygon p")
            .inner_join("mapunit m", "p.mukey = m.mukey")
            .inner_join("component c", "m.mukey = c.mukey")
            .inner_join("chorizon h", "c.cokey = h.cokey")
            .contains_point(longitude, latitude)
            .where("c.majcompflag = 'Yes'")
            .order_by("c.comppct_r DESC, h.hzdept_r")
        )

    @staticmethod
    def spatial_by_legend(areasymbol: str) -> SpatialQuery:
        """Get spatial data for map units on a legend/area symbol."""
        return (
            SpatialQuery()
            .select(
                "areasymbol",
                "mukey",
                "musym",
                "mupolygongeo.STAsText() as geometry",
                "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STArea() as shape_area",
                "GEOGRAPHY::STGeomFromWKB(mupolygongeo.STUnion(mupolygongeo.STStartPoint()).STAsBinary(), 4326).MakeValid().STLength() as shape_length",
            )
            .from_("mupolygon")
            .where(f"areasymbol = '{areasymbol}'")
        )

    @staticmethod
    def mapunits_intersecting_bbox(
        min_x: float, min_y: float, max_x: float, max_y: float
    ) -> SpatialQuery:
        """Get map units that intersect with a bounding box."""
        return (
            SpatialQuery()
            .select(
                "m.mukey", "m.musym", "m.muname", "mupolygongeo.STAsText() as geometry"
            )
            .from_("mupolygon p")
            .inner_join("mapunit m", "p.mukey = m.mukey")
            .intersects_bbox(min_x, min_y, max_x, max_y)
        )

    @staticmethod
    def available_survey_areas() -> Query:
        """Get list of available survey areas."""
        return (
            Query()
            .select("areasymbol", "areaname", "saversion")
            .from_("sacatalog")
            .order_by("areasymbol")
        )

    @staticmethod
    def survey_area_boundaries() -> SpatialQuery:
        """Get survey area boundary polygons."""
        return (
            SpatialQuery()
            .select("areasymbol", "areaname", "sapolygongeo.STAsText() as geometry")
            .from_("sapolygon")
        )

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
