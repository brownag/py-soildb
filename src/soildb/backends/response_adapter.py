"""Response adapter for converting database query results to SDAResponse.

This is one of the key infrastructure pieces that eliminates duplication.
The adapter converts results from any database (SQLite, PostgreSQL, etc.)
into SDAResponse format, which all backends then return uniformly.
"""

import logging
from typing import Any, Dict, List, Optional

from soildb.response import SDAResponse
from soildb.type_conversion import TypeMap, get_default_type_map

logger = logging.getLogger(__name__)


class ResponseAdapter:
    """Adapt query results from any database to SDAResponse format.

    Handles conversion of:
    - Tuple-based results (from aiosqlite, asyncpg, etc.)
    - Dict-based results (from ORM, asyncpg.Record, etc.)
    - Column name and type inference

    Result: All backends return SDAResponse consistently, enabling
    validation, conversion, and export in one place.
    """

    @staticmethod
    async def from_rows(
        rows: List[tuple],
        columns: List[str],
        type_map: Optional[TypeMap] = None,
    ) -> SDAResponse:
        """Convert database rows to SDAResponse.

        Args:
            rows: List of tuples from database query
            columns: Column names in order
            type_map: Database-specific type map. If None, uses default.

        Returns:
            SDAResponse: Formatted response compatible with SDA format

        Note:
            This is THE place where response conversion happens for all backends.
            By centralizing here, we avoid duplication and ensure consistency.
        """
        if type_map is None:
            type_map = get_default_type_map()

        # Infer types from first row
        metadata = []
        if rows:
            first_row = rows[0]
            for value in first_row:
                type_name = ResponseAdapter._infer_type(value, type_map)
                metadata.append(type_name)
        else:
            # Empty result: use object type for all columns
            metadata = ["object"] * len(columns)

        # Convert to SDA response format
        # Format: {"Table": [columns, metadata, row1, row2, ...]}
        table_data = [columns, metadata]
        table_data.extend([list(row) for row in rows])

        response_dict = {"Table": table_data}
        return SDAResponse(response_dict)

    @staticmethod
    async def from_dict_rows(
        rows: List[Dict],
        columns: List[str],
        type_map: Optional[TypeMap] = None,
    ) -> SDAResponse:
        """Convert dict-based rows to SDAResponse.

        Args:
            rows: List of dicts from database query
            columns: Column names in order (for dict extraction)
            type_map: Database-specific type map

        Returns:
            SDAResponse: Formatted response

        Example:
            Used by PostgreSQL backend (asyncpg.Record is dict-like):
            >>> rows = await conn.fetch("SELECT * FROM mapunit")
            >>> response = await ResponseAdapter.from_dict_rows(
            ...     [dict(row) for row in rows],
            ...     ['mukey', 'muname', 'musym']
            ... )
        """
        # Convert dicts to tuples in column order
        tuple_rows = []
        for row in rows:
            tuple_rows.append(tuple(row.get(col) for col in columns))

        return await ResponseAdapter.from_rows(tuple_rows, columns, type_map)

    @staticmethod
    async def combine_responses(responses: List[SDAResponse]) -> SDAResponse:
        """Merge multiple SDAResponse objects into one.

        Args:
            responses: List of SDAResponse objects to combine

        Returns:
            SDAResponse: Combined response with all rows

        Example:
            Used by LDMClient and SSURGOClient to combine chunked queries:
            >>> chunk_responses = [response1, response2, response3]
            >>> combined = await ResponseAdapter.combine_responses(chunk_responses)
        """
        if not responses:
            return SDAResponse({"Table": [[], []]})

        if len(responses) == 1:
            return responses[0]

        # Get unified structure from first non-empty response
        combined_columns = None
        combined_metadata = None
        combined_data = []

        for response in responses:
            if response.is_empty():
                continue

            # Initialize columns/metadata from first non-empty response
            if combined_columns is None:
                combined_columns = response._columns
                combined_metadata = response._metadata

            # Add all rows from this response
            combined_data.extend(response.to_dict())

        # If all responses were empty
        if combined_columns is None:
            return SDAResponse({"Table": [[], []]})

        # Reconstruct response with combined data
        # combined_columns and combined_metadata are verified not None above
        table_data: List[Any] = [combined_columns, combined_metadata]

        # Convert dict rows back to column-ordered tuples
        for row in combined_data:
            table_data.append([row.get(col) for col in combined_columns])  # type: ignore

        response_dict = {"Table": table_data}
        return SDAResponse(response_dict)

    @staticmethod
    def _infer_type(value: Any, type_map: TypeMap) -> str:
        """Infer SQL type from Python value.

        Args:
            value: Python value from database
            type_map: Type mapper for conversion

        Returns:
            SQL type name (e.g., 'int', 'varchar', 'geometry')

        Note:
            Type inference is approximate when database doesn't provide
            type information. Most databases (SQLite, PostgreSQL) do.
        """
        if value is None:
            return "object"
        elif isinstance(value, bool):
            # bool must come before int (bool is subclass of int)
            return "bit"
        elif isinstance(value, int):
            # Distinguish between int and bigint based on value size
            return "int" if value < 2**31 else "bigint"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Detect WKT geometry (common in GeoPackage, PostGIS)
            if value and value.startswith(
                (
                    "POINT",
                    "POLYGON",
                    "MULTIPOLYGON",
                    "LINESTRING",
                    "GEOMETRYCOLLECTION",
                    "MULTIPOINT",
                    "MULTILINESTRING",
                )
            ):
                return "geometry"
            return "varchar"
        elif isinstance(value, (bytes, bytearray)):
            # WKB geometry (GeoPackage stores geometry as BLOB)
            # Could detect magic bytes, but for now just treat as blob
            return "geometry"
        else:
            # Unknown type
            return "object"
