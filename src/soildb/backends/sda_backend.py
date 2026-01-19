"""SDA (Soil Data Access) backend implementation.

Provides access to soil data via the USDA Soil Data Access web service.
This backend wraps the existing SDAClient and reuses all HTTP logic,
error handling, and retry behavior.
"""

import logging
from typing import Any, Dict, List, Optional

from soildb.client import SDAClient
from soildb.response import SDAResponse

from .base import BaseBackend
from .exceptions import BackendConnectionError, BackendQueryError, BackendSchemaError

logger = logging.getLogger(__name__)


class SDABackend(BaseBackend):
    """Backend for data queries via Soil Data Access (HTTP API).

    This backend wraps the existing SDAClient for all HTTP operations.
    It provides a unified interface matching BaseBackend.
    """

    def __init__(
        self, client: Optional[SDAClient] = None, config: Optional[Any] = None
    ):
        """Initialize SDA backend.

        Args:
            client: Optional SDAClient instance. If None, a default client
                   will be created when needed.
            config: Client configuration (timeout, retries, etc.)
        """
        super().__init__(config)
        self._client = client
        self._owned_client = client is None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to SDA service.

        Returns:
            True if connection successful

        Raises:
            BackendConnectionError: If connection fails
        """
        try:
            await self._get_client()
            self._connected = True
            return True
        except BackendConnectionError:
            raise
        except Exception as e:
            raise BackendConnectionError(
                "Failed to connect to SDA service",
                details=str(e),
            ) from e

    async def execute(self, sql: str) -> SDAResponse:
        """Execute query via SDA HTTP API.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results from SDA

        Raises:
            BackendQueryError: If query execution fails
        """
        try:
            client = await self._get_client()
            response = await client.execute_sql(sql)
            return response
        except BackendQueryError:
            raise
        except Exception as e:
            raise BackendQueryError(
                "Failed to execute query via SDA",
                details=str(e),
            ) from e

    async def get_tables(self) -> List[str]:
        """Get available tables from SDA.

        Returns:
            List of table names available in SDA

        Raises:
            BackendSchemaError: If schema discovery fails
        """
        try:
            # For SDA, we could query information_schema for an accurate list
            # For now, return an empty list to allow dynamic discovery
            # Actual tables are determined by the user's query
            return []
        except Exception as e:
            raise BackendSchemaError(
                "Failed to get table list from SDA",
                details=str(e),
            ) from e

    async def get_columns(self, table_name: str) -> Dict[str, str]:
        """Get schema for a specific table from SDA.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to types

        Raises:
            BackendSchemaError: If schema discovery fails

        Note:
            For SDA backend, queries INFORMATION_SCHEMA to get column info.
            This provides approximate schema since SDA SQL Server backend
            may have types that need interpretation.
        """
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
            if not response.is_empty():
                for row in response.to_dict():
                    col_name = row.get("COLUMN_NAME", "")
                    col_type = row.get("DATA_TYPE", "")
                    if col_name:
                        schema[col_name] = col_type

            return schema

        except Exception as e:
            logger.warning(f"Failed to get schema for {table_name} via SDA: {e}")
            raise BackendSchemaError(
                f"Failed to get schema for {table_name}",
                details=str(e),
            ) from e

    async def close(self) -> None:
        """Close the SDA client if we own it."""
        if self._owned_client and self._client is not None:
            await self._client.close()
        self._connected = False

    async def _get_client(self) -> SDAClient:
        """Get or create SDAClient.

        Returns:
            SDAClient instance

        Raises:
            BackendConnectionError: If client cannot be initialized
        """
        if self._client is None:
            try:
                self._client = SDAClient(config=self.config)
                await self._client.connect()
            except Exception as e:
                raise BackendConnectionError(
                    "Failed to initialize SDA client",
                    details=str(e),
                ) from e

        return self._client


__all__ = ["SDABackend"]
