"""
LDM-specific SQL query builder.

Constructs SQL queries for Lab Data Mart tables with proper joins,
filtering, and chunking support.
"""

import logging
from collections.abc import Sequence
from typing import Optional, Union

from .exceptions import (
    LDMParameterError,
    LDMQueryError,
    LDMTableError,
)
from .tables import (
    DEFAULT_ANALYZED_SIZE_FRACTIONS,
    DEFAULT_AREA_TYPE,
    DEFAULT_LAYER_TYPES,
    DEFAULT_PREP_CODES,
    DEFAULT_TABLES,
    LAB_LAYER_TABLE,
    PEDON_TABLE,
    SITE_TABLE,
    all_valid_area_types,
    all_valid_layer_types,
    is_valid_prep_code,
    is_valid_size_fraction,
    is_valid_table,
    validate_tables,
)

logger = logging.getLogger(__name__)

# Mirror soilDB::fetchLDM filter behavior.
# prep_code applies to selected flat property tables (excluding rosetta/mir),
# analyzed_size_frac applies only to fractionated tables.
PREP_FILTER_TABLES = {
    "lab_physical_properties",
    "lab_chemical_properties",
    "lab_calculations_including_estimates_and_default_values",
    "lab_major_and_trace_elements_and_oxides",
    "lab_xray_and_thermal",
    "lab_xrd_and_thermal",
    "lab_mineralogy_glass_count",
    "lab_mineralogy_glass_count_and_optical_properties",
}

FRACTION_FILTER_TABLES = {
    "lab_mineralogy_glass_count",
    "lab_mineralogy_glass_count_and_optical_properties",
    "lab_xray_and_thermal",
    "lab_xrd_and_thermal",
}


class LDMQueryBuilder:
    """Builder for LDM SQL queries with filtering and chunking support."""

    def __init__(
        self,
        tables: Optional[list[str]] = None,
        layer_type: Union[str, Sequence[str], None] = DEFAULT_LAYER_TYPES,
        area_type: Optional[str] = DEFAULT_AREA_TYPE,
        prep_code: Union[str, Sequence[str], None] = DEFAULT_PREP_CODES,
        analyzed_size_frac: Union[
            str, Sequence[str], None
        ] = DEFAULT_ANALYZED_SIZE_FRACTIONS,
        dialect: str = "sql_server",
    ):
        """Initialize query builder with filtering options.

        Args:
            tables: List of LDM tables to query. If None, uses defaults.
            layer_type: Filter by horizon type (single value or list)
            area_type: Filter by geographic classification
            prep_code: Sample preparation code(s). Can be string or list.
                      Pass None for no filter. Defaults to ('S', '').
            analyzed_size_frac: Analyzed size fraction(s). String or list.
                               Pass None for no filter. Defaults to ('<2 mm', '').
            dialect: SQL dialect ('sql_server' or 'sqlite'). Defaults to
                    'sql_server' for SDA compatibility.

        Raises:
            LDMParameterError: If invalid parameters provided
            LDMTableError: If invalid table names provided
        """
        self.dialect = dialect
        self.prep_codes: Optional[list[str]] = None
        self.analyzed_size_fracs: Optional[list[str]] = None
        self.layer_types: Optional[list[str]] = None

        # Validate and set tables
        self.tables = tables or DEFAULT_TABLES
        if not validate_tables(self.tables):
            invalid = [t for t in self.tables if not is_valid_table(t)]
            raise LDMTableError(f"Invalid table names: {invalid}")

        # Validate filtering parameters
        if prep_code is not None:
            if isinstance(prep_code, str):
                if not is_valid_prep_code(prep_code):
                    raise LDMParameterError(
                        f"Invalid prep_code: '{prep_code}'",
                    )
                self.prep_codes = [prep_code]
            else:
                for code in prep_code:
                    if not is_valid_prep_code(code):
                        raise LDMParameterError(
                            f"Invalid prep_code: '{code}'",
                        )
                self.prep_codes = list(prep_code)

        if analyzed_size_frac is not None:
            if isinstance(analyzed_size_frac, str):
                if not is_valid_size_fraction(analyzed_size_frac):
                    raise LDMParameterError(
                        f"Invalid analyzed_size_frac: '{analyzed_size_frac}'",
                    )
                self.analyzed_size_fracs = [analyzed_size_frac]
            else:
                for frac in analyzed_size_frac:
                    if not is_valid_size_fraction(frac):
                        raise LDMParameterError(
                            f"Invalid analyzed_size_frac: '{frac}'",
                        )
                self.analyzed_size_fracs = list(analyzed_size_frac)

        if layer_type is not None:
            if isinstance(layer_type, str):
                if layer_type not in all_valid_layer_types():
                    raise LDMParameterError(
                        f"Invalid layer_type: '{layer_type}'",
                    )
                self.layer_types = [layer_type]
            else:
                for value in layer_type:
                    if value not in all_valid_layer_types():
                        raise LDMParameterError(
                            f"Invalid layer_type: '{value}'",
                        )
                self.layer_types = list(layer_type)

        if area_type is not None and area_type not in all_valid_area_types():
            raise LDMParameterError(
                f"Invalid area_type: '{area_type}'",
            )

        self.layer_type = (
            self.layer_types[0]
            if self.layer_types and len(self.layer_types) == 1
            else None
        )
        self.area_type = area_type

    def _coalesce(self, column: str, default: str) -> str:
        """Generate SQL coalesce function for the current dialect.

        Args:
            column: Column name
            default: Default value if column is NULL

        Returns:
            Dialect-appropriate coalesce expression
        """
        if self.dialect == "sqlite":
            return f"IFNULL({column}, {default})"
        else:  # sql_server
            return f"ISNULL({column}, {default})"

    def build_query(
        self,
        keys: Optional[list[Union[str, int]]] = None,
        key_column: str = "pedon_key",
        custom_where: Optional[str] = None,
    ) -> str:
        """Build SQL query for LDM data.

        Args:
            keys: List of key values to filter by (e.g., pedon IDs)
            key_column: Column name to filter on (default: "pedon_key")
            custom_where: Custom WHERE clause (overrides keys parameter)

        Returns:
            SQL query string

        Raises:
            LDMQueryError: If query cannot be built
        """
        try:
            # Build SELECT clause
            select_clause = self._build_select()

            # Build FROM clause with joins
            from_clause = self._build_from()

            # Build WHERE clause
            where_clause = self._build_where(
                keys=keys,
                key_column=key_column,
                custom_where=custom_where,
            )

            # Combine clauses
            query = f"{select_clause}\n{from_clause}"
            if where_clause:
                query += f"\n{where_clause}"

            query += (
                f"\nORDER BY {LAB_LAYER_TABLE}.pedon_key, {LAB_LAYER_TABLE}.layer_key"
            )

            return query

        except Exception as e:
            if isinstance(e, LDMQueryError):
                raise
            raise LDMQueryError(
                f"Failed to build LDM query: {str(e)}",
            ) from e

    def build_chunked_queries(
        self,
        keys: list[Union[str, int]],
        key_column: str = "pedon_key",
        chunk_size: int = 1000,
    ) -> list[str]:
        """Build multiple queries for chunked key processing.

        Args:
            keys: List of key values
            key_column: Column name to filter on
            chunk_size: Size of each chunk

        Returns:
            List of SQL query strings, one per chunk
        """
        queries = []
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            query = self.build_query(keys=chunk, key_column=key_column)
            queries.append(query)
        return queries

    def _build_select(self) -> str:
        """Build SELECT clause with all columns from selected tables.

        Returns:
            SELECT clause string
        """
        # Start with lab_layer table
        columns = [f"{LAB_LAYER_TABLE}.*"]

        # Add columns from data tables
        for table in self.tables:
            if table not in [PEDON_TABLE, SITE_TABLE, LAB_LAYER_TABLE]:
                columns.append(f"{table}.*")

        select_clause = "SELECT " + ", ".join(columns)
        return select_clause

    def _build_from(self) -> str:
        """Build FROM clause with joins between tables.

        Returns:
            FROM clause with all necessary JOINs
        """
        # Start with lab_layer as base (contains actual lab data)
        from_parts = [f"FROM {LAB_LAYER_TABLE}"]

        # Join data tables based on their key type
        for table in self.tables:
            if table not in [PEDON_TABLE, SITE_TABLE, LAB_LAYER_TABLE]:
                # Tables that join on labsampnum
                if table in [
                    "lab_physical_properties",
                    "lab_chemical_properties",
                    "lab_calculations_including_estimates_and_default_values",
                    "lab_major_and_trace_elements_and_oxides",
                    "lab_mir",
                ]:
                    join = (
                        f"LEFT JOIN {table} ON {LAB_LAYER_TABLE}.labsampnum "
                        f"= {table}.labsampnum"
                    )
                    from_parts.append(join)
                # Tables that join on layer_key
                elif table in ["lab_rosetta_Key"]:
                    join = (
                        f"LEFT JOIN {table} ON {LAB_LAYER_TABLE}.layer_key "
                        f"= {table}.layer_key"
                    )
                    from_parts.append(join)
                # Fractionated tables with special handling
                elif table in ["lab_mineralogy_glass_count", "lab_xray_and_thermal"]:
                    join = (
                        f"LEFT JOIN {table} ON {LAB_LAYER_TABLE}.labsampnum "
                        f"= {table}.labsampnum"
                    )
                    from_parts.append(join)

        return " ".join(from_parts)

    def _build_where(
        self,
        keys: Optional[list[Union[str, int]]] = None,
        key_column: str = "pedon_key",
        custom_where: Optional[str] = None,
    ) -> str:
        """Build WHERE clause with filters.

        Args:
            keys: List of key values to filter by
            key_column: Column name to filter on
            custom_where: Custom WHERE clause (overrides other filters)

        Returns:
            WHERE clause string, or empty string if no filters
        """
        conditions = []

        # Custom WHERE clause takes precedence
        if custom_where:
            return f"WHERE {custom_where}"

        # Add key filter
        if keys:
            # Escape string keys, use numeric keys as-is
            formatted_keys = []
            for key in keys:
                if isinstance(key, str):
                    # Escape single quotes
                    escaped = key.replace("'", "''")
                    formatted_keys.append(f"'{escaped}'")
                else:
                    formatted_keys.append(str(key))

            key_list = ", ".join(formatted_keys)
            conditions.append(f"{LAB_LAYER_TABLE}.{key_column} IN ({key_list})")

        # Add layer_type filter
        if self.layer_types:
            if len(self.layer_types) == 1:
                conditions.append(
                    f"{LAB_LAYER_TABLE}.layer_type = '{self.layer_types[0]}'"
                )
            else:
                layer_type_list = ", ".join(f"'{value}'" for value in self.layer_types)
                conditions.append(
                    f"{LAB_LAYER_TABLE}.layer_type IN ({layer_type_list})"
                )

        # Add prep_code filter for selected tables that support this field.
        # Use AND across selected tables to match the reference implementation.
        if self.prep_codes and self.tables:
            prep_conditions = []
            prep_code_list = ", ".join(f"'{code}'" for code in self.prep_codes)
            for table in self.tables:
                if table in PREP_FILTER_TABLES:
                    coalesce = self._coalesce(f"{table}.prep_code", "''")
                    prep_conditions.append(f"{coalesce} IN ({prep_code_list})")
            if prep_conditions:
                conditions.append(f"({' AND '.join(prep_conditions)})")

        # Add analyzed_size_frac filter only for fractionated tables.
        # This prevents invalid column errors on non-fractionated tables.
        if self.analyzed_size_fracs and self.tables:
            frac_conditions = []
            frac_list = ", ".join(f"'{frac}'" for frac in self.analyzed_size_fracs)
            for table in self.tables:
                if table in FRACTION_FILTER_TABLES:
                    coalesce = self._coalesce(f"{table}.analyzed_size_frac", "''")
                    frac_conditions.append(f"{coalesce} IN ({frac_list})")
            if frac_conditions:
                conditions.append(f"({' AND '.join(frac_conditions)})")

        if not conditions:
            return ""

        return "WHERE " + " AND ".join(conditions)


def build_ldm_site_query(
    WHERE: Optional[str] = None,
) -> str:
    """Build SQL query to get pedon_keys from lab_combine_nasis_ncss.

    First-stage query retrieves site/pedon metadata and returns pedon_keys
    needed for the second-stage layer query.

    Args:
        WHERE: Custom WHERE clause to filter on site/pedon columns

    Returns:
        SQL query string for site/pedon lookup
    """
    query = "SELECT DISTINCT pedon_key FROM lab_combine_nasis_ncss"

    if WHERE:
        query += f" WHERE {WHERE}"

    return query


def build_ldm_query(
    x: Optional[list[Union[str, int]]] = None,
    what: str = "pedon_key",
    tables: Optional[list[str]] = None,
    WHERE: Optional[str] = None,
    layer_type: Union[str, Sequence[str], None] = DEFAULT_LAYER_TYPES,
    area_type: Optional[str] = DEFAULT_AREA_TYPE,
    prep_code: Union[str, Sequence[str], None] = DEFAULT_PREP_CODES,
    analyzed_size_frac: Union[
        str, Sequence[str], None
    ] = DEFAULT_ANALYZED_SIZE_FRACTIONS,
) -> str:
    """Convenience function to build a single LDM query.

    Args:
        x: List of values to filter by
        what: Column name to filter on (if x is provided)
        tables: List of LDM tables to query
        WHERE: Custom WHERE clause (overrides x/what)
        layer_type: Filter by horizon type
        area_type: Filter by geographic classification
        prep_code: Sample preparation code
        analyzed_size_frac: Analyzed size fraction

    Returns:
        SQL query string

    Raises:
        LDMParameterError: If invalid parameters
        LDMTableError: If invalid table names
    """
    builder = LDMQueryBuilder(
        tables=tables,
        layer_type=layer_type,
        area_type=area_type,
        prep_code=prep_code,
        analyzed_size_frac=analyzed_size_frac,
    )

    return builder.build_query(
        keys=x,
        key_column=what,
        custom_where=WHERE,
    )
