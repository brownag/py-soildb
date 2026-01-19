"""
LDM-specific SQL query builder.

Constructs SQL queries for Lab Data Mart tables with proper joins,
filtering, and chunking support.
"""

import logging
from typing import List, Optional, Union

from .exceptions import (
    LDMParameterError,
    LDMQueryError,
    LDMTableError,
)
from .tables import (
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


class LDMQueryBuilder:
    """Builder for LDM SQL queries with filtering and chunking support."""

    def __init__(
        self,
        tables: Optional[List[str]] = None,
        layer_type: Optional[str] = None,
        area_type: Optional[str] = None,
        prep_code: str = "S",
        analyzed_size_frac: str = "<2 mm",
    ):
        """Initialize query builder with filtering options.

        Args:
            tables: List of LDM tables to query. If None, uses defaults.
            layer_type: Filter by horizon type (None for all)
            area_type: Filter by geographic classification (None for all)
            prep_code: Sample preparation code (default: "S")
            analyzed_size_frac: Analyzed size fraction (default: "<2 mm")

        Raises:
            LDMParameterError: If invalid parameters provided
            LDMTableError: If invalid table names provided
        """
        # Validate and set tables
        self.tables = tables or DEFAULT_TABLES
        if not validate_tables(self.tables):
            invalid = [t for t in self.tables if not is_valid_table(t)]
            raise LDMTableError(f"Invalid table names: {invalid}")

        # Validate filtering parameters
        if not is_valid_prep_code(prep_code):
            raise LDMParameterError(
                f"Invalid prep_code: '{prep_code}'",
            )

        if not is_valid_size_fraction(analyzed_size_frac):
            raise LDMParameterError(
                f"Invalid analyzed_size_frac: '{analyzed_size_frac}'",
            )

        if layer_type is not None and layer_type not in all_valid_layer_types():
            raise LDMParameterError(
                f"Invalid layer_type: '{layer_type}'",
            )

        if area_type is not None and area_type not in all_valid_area_types():
            raise LDMParameterError(
                f"Invalid area_type: '{area_type}'",
            )

        self.layer_type = layer_type
        self.area_type = area_type
        self.prep_code = prep_code
        self.analyzed_size_frac = analyzed_size_frac

    def build_query(
        self,
        keys: Optional[List[Union[str, int]]] = None,
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

            query += "\nORDER BY pedon_key, lab_layer_key"

            return query

        except Exception as e:
            if isinstance(e, LDMQueryError):
                raise
            raise LDMQueryError(
                f"Failed to build LDM query: {str(e)}",
            ) from e

    def build_chunked_queries(
        self,
        keys: List[Union[str, int]],
        key_column: str = "pedon_key",
        chunk_size: int = 1000,
    ) -> List[str]:
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
        # Start with pedon/site/layer tables
        columns = [
            f"{PEDON_TABLE}.*",
            f"{SITE_TABLE}.*",
            f"{LAB_LAYER_TABLE}.*",
        ]

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

        # Join to pedon table
        from_parts.append(
            f"INNER JOIN {PEDON_TABLE} ON {LAB_LAYER_TABLE}.pedon_key = {PEDON_TABLE}.pedon_key"
        )

        # Join to site table
        from_parts.append(
            f"INNER JOIN {SITE_TABLE} ON {PEDON_TABLE}.site_key = {SITE_TABLE}.site_key"
        )

        # Join data tables
        for table in self.tables:
            if table not in [PEDON_TABLE, SITE_TABLE, LAB_LAYER_TABLE]:
                from_parts.append(
                    f"LEFT JOIN {table} ON {LAB_LAYER_TABLE}.lab_layer_key = {table}.lab_layer_key"
                )

        return " ".join(from_parts)

    def _build_where(
        self,
        keys: Optional[List[Union[str, int]]] = None,
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
            conditions.append(f"{key_column} IN ({key_list})")

        # Add prep_code filter
        if self.prep_code:
            conditions.append(f"prep_code = '{self.prep_code}'")

        # Add analyzed_size_frac filter
        if self.analyzed_size_frac:
            conditions.append(f"analyzed_size_frac = '{self.analyzed_size_frac}'")

        # Add layer_type filter
        if self.layer_type:
            conditions.append(f"layer_type = '{self.layer_type}'")

        # Add area_type filter
        if self.area_type:
            conditions.append(f"area_type = '{self.area_type}'")

        if not conditions:
            return ""

        return "WHERE " + " AND ".join(conditions)


def build_ldm_query(
    x: Optional[List[Union[str, int]]] = None,
    what: str = "pedon_key",
    tables: Optional[List[str]] = None,
    WHERE: Optional[str] = None,
    layer_type: Optional[str] = None,
    area_type: Optional[str] = None,
    prep_code: str = "S",
    analyzed_size_frac: str = "<2 mm",
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
