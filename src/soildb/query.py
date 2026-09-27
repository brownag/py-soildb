"""
SQL query building classes for SDA queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Union

from . import ssurgo_tables
from .sanitization import (
    sanitize_sql_numeric,
    sanitize_sql_string,
    validate_sql_identifier,
)


def in_condition(
    column: str,
    values: Iterable[Union[str, int, float]],
    case_insensitive: bool = False,
) -> str:
    """Build a WHERE IN condition with automatic escaping.

    Validates the column identifier and safely formats all values,
    automatically escaping strings to prevent SQL injection.

    Args:
        column: Column name (can be qualified like 'alias.column').
            Validated as a SQL identifier.
        values: Iterable of values to include in the IN clause
            (strings, ints, or floats).
        case_insensitive: If True, wraps column and string values
            with LOWER() for case-insensitive comparison. Numeric
            values are never wrapped in LOWER().

    Returns:
        str: The formatted SQL IN condition (e.g., "mukey IN (100, 200)"
            or "LOWER(areasymbol) IN (LOWER('ia001'))").

    Raises:
        ValueError: If column is not a valid SQL identifier,
            if values is empty, or if a value is not str, int, or float.

    Examples:
        >>> in_condition("mukey", [100, 200, 300])
        'mukey IN (100, 200, 300)'

        >>> in_condition("musym", ["IA001", "O'Brien"])
        "musym IN ('IA001', 'O''Brien')"

        >>> in_condition("areasymbol", ["ia001"], case_insensitive=True)
        "LOWER(areasymbol) IN (LOWER('ia001'))"
    """
    # Validate column identifier
    validate_sql_identifier(column)

    # Convert to list to check length and iterate
    values_list = list(values)
    if not values_list:
        raise ValueError("in_condition() requires at least one value")

    # Format each value
    formatted_values = []
    for value in values_list:
        if isinstance(value, str):
            formatted_values.append(sanitize_sql_string(value))
        elif isinstance(value, (int, float)):
            formatted_values.append(sanitize_sql_numeric(value))
        else:
            raise ValueError(
                f"in_condition() values must be str, int, or float, "
                f"got {type(value).__name__}"
            )

    # Build the IN clause
    if case_insensitive:
        # Apply LOWER() to string values; leave numerics as-is
        formatted_values_lower = []
        for i, value in enumerate(values_list):
            orig_formatted = formatted_values[i]
            if isinstance(value, str):
                # Already quoted; wrap with LOWER()
                formatted_values_lower.append(f"LOWER({orig_formatted})")
            else:
                # Numeric; include as-is
                formatted_values_lower.append(orig_formatted)
        values_str = ", ".join(formatted_values_lower)
        return f"LOWER({column}) IN ({values_str})"
    else:
        values_str = ", ".join(formatted_values)
        return f"{column} IN ({values_str})"


def eq_condition(
    column: str,
    value: Union[str, int, float],
    case_insensitive: bool = False,
) -> str:
    """Build a WHERE equality condition with automatic escaping.

    Validates the column identifier and safely formats the value,
    automatically escaping strings to prevent SQL injection.

    Args:
        column: Column name (can be qualified like 'alias.column').
            Validated as a SQL identifier.
        value: Value for equality comparison (string, int, or float).
        case_insensitive: If True, wraps column and string value
            with LOWER() for case-insensitive comparison. Numeric
            values are never wrapped in LOWER().

    Returns:
        str: The formatted SQL equality condition (e.g., "mukey = 100"
            or "LOWER(areasymbol) = LOWER('ia109')").

    Raises:
        ValueError: If column is not a valid SQL identifier,
            or if value is not str, int, or float.

    Examples:
        >>> eq_condition("mukey", 100)
        'mukey = 100'

        >>> eq_condition("areasymbol", "IA109")
        "areasymbol = 'IA109'"

        >>> eq_condition("areasymbol", "ia109", case_insensitive=True)
        "LOWER(areasymbol) = LOWER('ia109')"
    """
    # Validate column identifier
    validate_sql_identifier(column)

    # Format value
    if isinstance(value, str):
        formatted_value = sanitize_sql_string(value)
    elif isinstance(value, (int, float)):
        formatted_value = sanitize_sql_numeric(value)
    else:
        raise ValueError(
            f"eq_condition() value must be str, int, or float, "
            f"got {type(value).__name__}"
        )

    # Build the equality condition
    if case_insensitive:
        if isinstance(value, str):
            return f"LOWER({column}) = LOWER({formatted_value})"
        else:
            # Numeric; no LOWER needed
            return f"{column} = {formatted_value}"
    else:
        return f"{column} = {formatted_value}"


# Standard column sets for common query patterns
class ColumnSets:
    """Standardized column sets for common SDA query patterns."""

    # Map unit columns
    MAPUNIT_BASIC = list(ssurgo_tables.DEFAULT_COLUMNS["mapunit"])
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
        self._where_conditions: list[str] = []
        self._join_clauses: list[str] = []
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

    def where_in(
        self,
        column: str,
        values: Iterable[Union[str, int, float]],
        *,
        case_insensitive: bool = False,
    ) -> "Query":
        """Add a WHERE IN condition with automatic escaping.

        Validates the column identifier and safely formats all values,
        automatically escaping strings to prevent SQL injection.

        Args:
            column: Column name (can be qualified like 'alias.column').
                Validated as a SQL identifier.
            values: Iterable of values to include in the IN clause
                (strings, ints, or floats).
            case_insensitive: If True, wraps column and string values
                with LOWER() for case-insensitive comparison. Uses the
                form `LOWER(col) IN (LOWER('a'), ...)`.

        Returns:
            Query: This Query instance for method chaining.

        Raises:
            ValueError: If column is not a valid SQL identifier,
                or if values is empty.

        Examples:
            # Simple IN clause with mixed types
            >>> query = Query().select("mukey").from_("mapunit")
            >>> query.where_in("mukey", [100, 200, 300])
            # Generates: WHERE mukey IN (100, 200, 300)

            # String values with quotes are escaped
            >>> query.where_in("musym", ["IA001", "O'Brien"])
            # Generates: WHERE musym IN ('IA001', 'O''Brien')

            # Case-insensitive match (used by LDM)
            >>> query.where_in("areasymbol", ["IA001"], case_insensitive=True)
            # Generates: WHERE LOWER(areasymbol) IN (LOWER('IA001'))

            # Chaining with other conditions
            >>> query.where_in("cokey", [1, 2]).where("majcompflag = 'Y'")
            # Generates: WHERE cokey IN (1, 2) AND majcompflag = 'Y'
        """
        condition = in_condition(column, values, case_insensitive)
        self._where_conditions.append(condition)
        return self

    def where_eq(
        self,
        column: str,
        value: Union[str, int, float],
        *,
        case_insensitive: bool = False,
    ) -> "Query":
        """Add a WHERE equality condition with automatic escaping.

        Validates the column identifier and safely formats the value,
        automatically escaping strings to prevent SQL injection.

        Args:
            column: Column name (can be qualified like 'alias.column').
                Validated as a SQL identifier.
            value: Value for equality comparison (string, int, or float).
            case_insensitive: If True, wraps column and value (if string)
                with LOWER() for case-insensitive comparison. Uses the
                form `LOWER(col) = LOWER('value')`.

        Returns:
            Query: This Query instance for method chaining.

        Raises:
            ValueError: If column is not a valid SQL identifier.

        Examples:
            # Simple equality with string
            >>> query = Query().select("mukey").from_("mapunit")
            >>> query.where_eq("areasymbol", "IA109")
            # Generates: WHERE areasymbol = 'IA109'

            # String with quote is escaped
            >>> query.where_eq("musym", "O'Brien")
            # Generates: WHERE musym = 'O''Brien'

            # Numeric value
            >>> query.where_eq("mukey", 100)
            # Generates: WHERE mukey = 100

            # Case-insensitive match
            >>> query.where_eq("areasymbol", "ia109", case_insensitive=True)
            # Generates: WHERE LOWER(areasymbol) = LOWER('ia109')

            # Chaining
            >>> query.where_eq("mukey", 100).where_eq("cokey", 200)
            # Generates: WHERE mukey = 100 AND cokey = 200
        """
        condition = eq_condition(column, value, case_insensitive)
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
