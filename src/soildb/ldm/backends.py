"""
Backend implementations for LDM data access.

Provides pluggable backends for accessing LDM data via different sources:
- SDABackend: HTTP queries via Soil Data Access
- SQLiteBackend: Local SQLite database queries

Both backends return SDAResponse objects for consistency.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import aiosqlite

from soildb.client import SDAClient
from soildb.response import SDAResponse

from .exceptions import (
    LDMResponseError,
    LDMSDAError,
    LDMSQLiteError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Protocol (Interface)
# ============================================================================


class LDMBackend(Protocol):
    """Protocol defining the interface for LDM backends.

    Any backend must implement these methods to be compatible with LDMClient.
    """

    async def execute_query(self, sql: str) -> SDAResponse:
        """Execute a query and return results as SDAResponse.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results

        Raises:
            LDMBackendError: If query execution fails
        """
        ...

    async def get_available_tables(self) -> List[str]:
        """Get list of available tables in the data source.

        Returns:
            List of table names available in this backend
        """
        ...

    async def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get column names and types for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to their types
        """
        ...


# ============================================================================
# SDA Backend Implementation
# ============================================================================


class SDABackend:
    """Backend for LDM queries via Soil Data Access (HTTP API).

    This backend wraps the existing SDAClient and reuses all HTTP logic,
    error handling, and retry behavior.
    """

    def __init__(self, client: Optional[SDAClient] = None):
        """Initialize SDA backend.

        Args:
            client: Optional SDAClient instance. If None, a default client
                   will be created when needed.
        """
        self._client = client
        self._owned_client = client is None

    async def execute_query(self, sql: str) -> SDAResponse:
        """Execute query via SDA HTTP API.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results from SDA

        Raises:
            LDMSDAError: If query execution fails
        """
        try:
            client = await self._get_client()
            response = await client.execute_sql(sql)
            return response
        except Exception as e:
            if isinstance(e, SDAResponse):
                return e
            raise LDMSDAError(
                f"Failed to execute LDM query via SDA: {str(e)}",
                details=str(e),
            ) from e

    async def get_available_tables(self) -> List[str]:
        """Get available LDM tables from SDA.

        Returns:
            List of table names from SDA metadata
        """
        # For SDA, we return the known LDM tables
        # In production, could query information_schema for accurate list
        from .tables import ALL_TABLES

        return ALL_TABLES

    async def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get schema for a specific LDM table from SDA.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to types

        Note:
            For SDA backend, returns approximate schema since SDA SQL Server
            backend may have types we can't easily query.
        """
        # Query information_schema to get column info
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """

        try:
            client = await self._get_client()
            response = await client.execute_sql(query)
            schema = {}
            for row in response.to_dict():
                col_name = row.get("COLUMN_NAME", "")
                col_type = row.get("DATA_TYPE", "")
                if col_name:
                    schema[col_name] = col_type
            return schema
        except Exception as e:
            logger.warning(f"Failed to get schema for {table_name} via SDA: {e}")
            return {}

    async def _get_client(self) -> SDAClient:
        """Get or create SDAClient.

        Returns:
            SDAClient instance

        Raises:
            LDMSDAError: If client cannot be initialized
        """
        if self._client is None:
            try:
                self._client = SDAClient()
                await self._client.connect()
            except Exception as e:
                raise LDMSDAError(
                    "Failed to initialize SDA client for LDM queries",
                    details=str(e),
                ) from e
        return self._client

    async def close(self) -> None:
        """Close the SDA client if we own it."""
        if self._owned_client and self._client is not None:
            await self._client.close()


# ============================================================================
# SQLite Backend Implementation
# ============================================================================


class SQLiteBackend:
    """Backend for LDM queries via local SQLite database snapshot.

    Provides access to LDM data from downloaded SQLite snapshots.
    """

    def __init__(self, db_path: Union[str, Path]):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file

        Raises:
            LDMSQLiteError: If database file doesn't exist
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise LDMSQLiteError(
                f"SQLite database not found at {db_path}",
                details=f"Expected file: {self.db_path.resolve()}",
            )

    async def execute_query(self, sql: str) -> SDAResponse:
        """Execute query against local SQLite database.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results converted from SQLite

        Raises:
            LDMSQLiteError: If query execution fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                # Set row factory to get columns
                db.row_factory = aiosqlite.Row

                async with db.execute(sql) as cursor:
                    # Get column names from cursor description
                    if not cursor.description:
                        columns = []
                        rows: List[Any] = []
                    else:
                        columns = [desc[0] for desc in cursor.description]
                        rows = list(await cursor.fetchall())

                    # Convert aiosqlite.Row objects to lists
                    data = [list(row) for row in rows]

                    # Create SDAResponse from SQLite results
                    response = self._create_response(columns, data)
                    return response

        except Exception as e:
            if isinstance(e, LDMSQLiteError):
                raise
            raise LDMSQLiteError(
                f"Failed to execute query on SQLite database: {str(e)}",
                details=f"Database: {self.db_path}, Query: {sql[:100]}...",
            ) from e

    async def get_available_tables(self) -> List[str]:
        """Get list of available tables in SQLite database.

        Returns:
            List of table names in the database

        Raises:
            LDMSQLiteError: If query fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
        except Exception as e:
            raise LDMSQLiteError(
                f"Failed to get table list from SQLite database: {str(e)}",
                details=f"Database: {self.db_path}",
            ) from e

    async def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get schema for a specific table in SQLite database.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to types

        Raises:
            LDMSQLiteError: If query fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                    rows = await cursor.fetchall()
                    schema = {}
                    for row in rows:
                        col_name = row[1]  # Column name is second field
                        col_type = row[2]  # Type is third field
                        schema[col_name] = col_type
                    return schema
        except Exception as e:
            raise LDMSQLiteError(
                f"Failed to get schema for {table_name}: {str(e)}",
                details=f"Database: {self.db_path}",
            ) from e

    def _create_response(
        self, columns: List[str], data: List[List[Any]]
    ) -> SDAResponse:
        """Convert SQLite results to SDAResponse format.

        Args:
            columns: List of column names
            data: List of data rows

        Returns:
            SDAResponse object

        Raises:
            LDMResponseError: If conversion fails
        """
        try:
            # SDAResponse expects data in specific format:
            # Row 0: Column names
            # Row 1: Column types (metadata)
            # Rows 2+: Data

            # Build metadata row (SQLite doesn't have rich type info)
            # Use Python type names as approximation
            metadata = []
            for value in data[0] if data else [None] * len(columns):
                if value is None:
                    type_name = "object"
                elif isinstance(value, bool):
                    type_name = "bool"
                elif isinstance(value, int):
                    type_name = "int"
                elif isinstance(value, float):
                    type_name = "float"
                elif isinstance(value, str):
                    type_name = "str"
                else:
                    type_name = "object"
                metadata.append(type_name)

            # Format as required by SDAResponse
            # SDAResponse expects raw JSON in specific format
            raw_data = {
                "Table": [columns, metadata] + data,
            }

            # Create SDAResponse
            response = SDAResponse(raw_data)
            return response

        except Exception as e:
            raise LDMResponseError(
                f"Failed to convert SQLite results to SDAResponse: {str(e)}",
                details=f"Columns: {len(columns)}, Rows: {len(data)}",
            ) from e

    async def close(self) -> None:
        """Close backend resources (SQLite doesn't need explicit close)."""
        pass


# ============================================================================
# Backend Factory
# ============================================================================


async def create_backend(
    dsn: Optional[Union[str, Path]] = None,
    sda_client: Optional[SDAClient] = None,
) -> Union[SDABackend, SQLiteBackend]:
    """Create appropriate backend based on parameters.

    Args:
        dsn: Path to SQLite database. If None, uses SDA backend.
        sda_client: Optional SDAClient for SDA backend.

    Returns:
        Appropriate backend instance (SDABackend or SQLiteBackend)

    Raises:
        LDMSQLiteError: If dsn is provided but file doesn't exist
        LDMSDAError: If SDA backend cannot be initialized
    """
    if dsn is not None:
        return SQLiteBackend(dsn)
    else:
        backend = SDABackend(sda_client)
        await backend._get_client()  # Ensure client is initialized
        return backend
