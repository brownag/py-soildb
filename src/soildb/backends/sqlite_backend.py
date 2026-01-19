"""SQLite backend implementation.

Provides access to local SQLite database snapshots.
This is commonly used for LDM (Lab Data Mart) snapshots and other
soil database exports that are distributed as SQLite files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiosqlite

from soildb.response import SDAResponse

from .base import BaseBackend
from .exceptions import BackendConnectionError, BackendQueryError, BackendSchemaError
from .response_adapter import ResponseAdapter
from .type_mapper import TypeMapperFactory

logger = logging.getLogger(__name__)


class SQLiteBackend(BaseBackend):
    """Backend for local SQLite database queries.

    Provides access to SQLite database files, commonly used for:
    - LDM (Lab Data Mart) snapshots
    - SSURGO data exports
    - GeoPackage files (via GeoPackageBackend subclass)
    """

    def __init__(self, db_path: Union[str, Path], config=None):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
            config: Client configuration (timeout, retries, etc.)

        Raises:
            BackendConnectionError: If database file doesn't exist
        """
        super().__init__(config)
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise BackendConnectionError(
                f"SQLite database not found",
                details=f"Expected file: {self.db_path.resolve()}",
            )

        self._type_mapper = TypeMapperFactory.for_sqlite()
        self._connected = False

    async def connect(self) -> bool:
        """Connect to SQLite database.

        Returns:
            True if connection successful

        Raises:
            BackendConnectionError: If connection fails
        """
        try:
            # Test connection by opening and closing immediately
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("SELECT 1")
            self._connected = True
            return True
        except Exception as e:
            raise BackendConnectionError(
                "Failed to connect to SQLite database",
                details=f"Database: {self.db_path}, Error: {str(e)}",
            ) from e

    async def execute(self, sql: str) -> SDAResponse:
        """Execute query against SQLite database.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results converted from SQLite

        Raises:
            BackendQueryError: If query execution fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(sql) as cursor:
                    # Get column names and data
                    if not cursor.description:
                        columns = []
                        rows = []
                    else:
                        columns = [desc[0] for desc in cursor.description]
                        rows = await cursor.fetchall()

                    # Convert aiosqlite.Row objects to tuples
                    tuple_rows = [tuple(row) for row in rows]

                    # Use ResponseAdapter for consistent conversion
                    response = await ResponseAdapter.from_rows(
                        tuple_rows,
                        columns,
                        self._type_mapper,
                    )

                    return response

        except BackendQueryError:
            raise
        except Exception as e:
            raise BackendQueryError(
                "Failed to execute query on SQLite database",
                details=f"Database: {self.db_path}, Query: {sql[:100]}..., Error: {str(e)}",
            ) from e

    async def get_tables(self) -> List[str]:
        """Get list of available tables in SQLite database.

        Returns:
            List of table names in the database

        Raises:
            BackendSchemaError: If query fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
        except Exception as e:
            raise BackendSchemaError(
                "Failed to get table list from SQLite database",
                details=f"Database: {self.db_path}, Error: {str(e)}",
            ) from e

    async def get_columns(self, table_name: str) -> Dict[str, str]:
        """Get schema for a specific table in SQLite database.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to types

        Raises:
            BackendSchemaError: If query fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                    rows = await cursor.fetchall()

                    schema = {}
                    for row in rows:
                        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                        col_name = row[1]  # Column name
                        col_type = row[2]  # Type
                        schema[col_name] = col_type

                    return schema

        except Exception as e:
            raise BackendSchemaError(
                f"Failed to get schema for {table_name}",
                details=f"Database: {self.db_path}, Error: {str(e)}",
            ) from e

    async def close(self) -> None:
        """Close backend resources.

        SQLite connections are automatically closed, but this method
        is provided for consistency with other backends.
        """
        self._connected = False


__all__ = ["SQLiteBackend"]
