"""Base backend abstract class for unified multi-backend data access.

Provides abstract interface that all backends (SDA, SQLite, GeoPackage, PostgreSQL)
must implement. Handles connection lifecycle, query execution, and schema introspection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from soildb.base_client import ClientConfig
from soildb.response import SDAResponse


class BaseBackend(ABC):
    """Abstract base for all data backends.

    All backends (SDA HTTP, SQLite, GeoPackage, PostgreSQL, etc.) inherit from
    this class and implement the required methods. This ensures consistent
    interface and lifecycle management across all data sources.

    Subclasses must implement:
    - connect(): Test connection to backend
    - execute(): Execute query and return SDAResponse
    - get_tables(): List available tables
    - get_columns(): Get column info for a table

    Optional methods for backends that support them:
    - query_intersects(): Execute spatial queries (GeoPackage, PostGIS)
    - get_geometry_column(): Get geometry column name (spatial backends)
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        """Initialize backend with configuration.

        Args:
            config: ClientConfig with timeout, retry, and other settings.
                   If None, uses ClientConfig.default().
        """
        self.config = config or ClientConfig.default()

    @abstractmethod
    async def connect(self) -> bool:
        """Test connection to backend.

        Raises:
            BackendConnectionError: If connection fails
        """
        ...

    @abstractmethod
    async def execute(self, sql: str) -> SDAResponse:
        """Execute query and return SDAResponse.

        Args:
            sql: SQL query string

        Returns:
            SDAResponse: Query results with type information

        Raises:
            BackendQueryError: If query execution fails
        """
        ...

    @abstractmethod
    async def get_tables(self) -> List[str]:
        """Get list of available tables in backend.

        Returns:
            List of table names

        Raises:
            BackendSchemaError: If schema introspection fails
        """
        ...

    @abstractmethod
    async def get_columns(self, table: str) -> Dict[str, str]:
        """Get column names and types for a table.

        Args:
            table: Table name

        Returns:
            Dict mapping column names to their database types

        Raises:
            BackendSchemaError: If schema introspection fails
        """
        ...

    async def execute_many(self, queries: List[str]) -> List[SDAResponse]:
        """Execute multiple queries concurrently.

        Args:
            queries: List of SQL query strings

        Returns:
            List of SDAResponse objects in same order as input queries

        Note:
            Default implementation executes sequentially. Subclasses can
            override for parallel execution (e.g., connection pooling).
        """
        import asyncio

        tasks = [self.execute(sql) for sql in queries]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close backend resources.

        Override in subclasses that maintain persistent connections
        (e.g., connection pools).

        Default implementation does nothing (safe for stateless backends).
        """
        pass

    async def __aenter__(self):
        """Support async context manager protocol."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager protocol."""
        await self.close()
        return False

    # Optional methods for spatial backends

    async def query_intersects(
        self,
        table: str,
        geometry_wkt: str,
        predicate: str = "intersects",
    ) -> SDAResponse:
        """Execute spatial predicate query.

        Override in spatial backends (GeoPackage, PostGIS).

        Args:
            table: Table name
            geometry_wkt: WKT geometry string
            predicate: Spatial predicate (intersects, contains, within)

        Returns:
            SDAResponse with matching records

        Raises:
            NotImplementedError: If backend doesn't support spatial queries
            BackendQueryError: If query fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support spatial queries"
        )

    async def get_geometry_column(self, table: str) -> Optional[str]:
        """Get name of geometry column for a table.

        Override in spatial backends (GeoPackage, PostGIS).

        Args:
            table: Table name

        Returns:
            Geometry column name, or None if table is not spatial

        Raises:
            BackendSchemaError: If schema introspection fails
        """
        return None
