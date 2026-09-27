"""
LDMClient - Main client for accessing Lab Data Mart data.

Provides unified interface for querying LDM data via SDA web service
or local SQLite snapshots with automatic chunking and retry logic.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from soildb.backends import SDABackend, SQLiteBackend
from soildb.base_client import BaseDataAccessClient, ClientConfig
from soildb.chunked import fetch_chunked
from soildb.client import SDAClient
from soildb.query import in_condition
from soildb.response import SDAResponse

from .exceptions import (
    LDMBackendSelectionError,
    LDMParameterError,
    LDMQueryError,
)
from .query_builder import LDMQueryBuilder, build_ldm_site_query
from .tables import (
    DEFAULT_ANALYZED_SIZE_FRACTIONS,
    DEFAULT_AREA_TYPE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LAYER_TYPES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PREP_CODES,
)

logger = logging.getLogger(__name__)


class LDMClient(BaseDataAccessClient):
    """Client for Kellogg Soil Survey Laboratory Data Mart access.

    Provides unified interface for querying lab soil characterization data
    from the NRCS Kellogg Soil Survey Laboratory (KSSL) via two backends:

    1. **Soil Data Access (SDA) web service** - Real-time data via HTTP API
    2. **Local SQLite snapshots** - Offline analysis using downloaded databases

    The client automatically handles:
    - Backend selection (SDA vs SQLite)
    - Query chunking for large datasets
    - Retry logic with halved chunk sizes on failure
    - Response format conversion to SDAResponse

    Examples:
        Query via SDA web service::

            async with LDMClient() as client:
                response = await client.query(
                    x=['85P0234', '40A3306'],
                    what='pedlabsampnum'
                )
                df = response.to_pandas()

        Query via local SQLite snapshot::

            async with LDMClient(dsn='LDM_FY2025.sqlite') as client:
                response = await client.query(
                    x=['85P0234'],
                    what='pedlabsampnum'
                )

        With manual client management::

            client = LDMClient()
            try:
                response = await client.query(x=['85P0234'], what='pedlabsampnum')
            finally:
                await client.close()
    """

    def __init__(
        self,
        dsn: Optional[Union[str, Path]] = None,
        config: Optional[ClientConfig] = None,
        sda_client: Optional[SDAClient] = None,
    ):
        """Initialize LDM client with backend selection.

        Args:
            dsn: Path to SQLite database. If None, uses SDA web service.
            config: Client configuration (timeout, retries, etc.)
            sda_client: Optional SDAClient for SDA backend reuse

        Raises:
            LDMBackendSelectionError: If backend cannot be initialized
            FileNotFoundError: If dsn path doesn't exist
        """
        super().__init__(config or ClientConfig.default())

        self.dsn = dsn
        self.sda_client = sda_client
        self._backend: Optional[Union[SDABackend, SQLiteBackend]] = (
            None  # Generic backends
        )

    async def connect(self) -> bool:
        """Test connection to the data source.

        Returns:
            True if connection successful

        Raises:
            LDMBackendSelectionError: If connection fails
        """
        try:
            backend = await self._get_backend()

            # Test with a simple query
            if isinstance(backend, SDABackend):
                # For SDA, the _get_client call already tested connection
                return True
            else:
                # For SQLite, try getting table list
                await backend.get_tables()
                return True

        except Exception as e:
            raise LDMBackendSelectionError(
                f"Failed to connect to LDM backend: {str(e)}"
            ) from e

    async def query(
        self,
        x: Optional[Union[list[Union[str, int]], str, int]] = None,
        what: str = "pedlabsampnum",
        bycol: str = "pedon_key",
        tables: Optional[list[str]] = None,
        WHERE: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_concurrency: int = 4,
        layer_type: Union[str, Sequence[str], None] = DEFAULT_LAYER_TYPES,
        area_type: Optional[str] = DEFAULT_AREA_TYPE,
        prep_code: Union[str, Sequence[str], None] = DEFAULT_PREP_CODES,
        analyzed_size_frac: Union[
            str, Sequence[str], None
        ] = DEFAULT_ANALYZED_SIZE_FRACTIONS,
    ) -> SDAResponse:
        """Execute LDM query with automatic chunking and retry logic.

        Implements two-stage query pattern:
        1. Query lab_combine_nasis_ncss for matching pedon_keys
        2. Query lab_layer for data from those pedons

        Args:
            x: Values to search for in 'what' column. Can be single value or list.
            what: Column name for filtering (e.g., 'pedlabsampnum', 'upedonid')
            bycol: Column for chunking operations (default: 'pedon_key')
            tables: List of LDM tables to retrieve
            WHERE: Custom SQL WHERE clause (overrides x/what parameters)
            chunk_size: Number of records per query chunk (default: 1000)
            max_retries: Maximum retry attempts with halved chunk_size (default: 3)
            max_concurrency: Maximum concurrent chunk requests (default: 4)
            layer_type: Filter by horizon type(s)
            area_type: Filter by geographic classification (default: 'ssa')
            prep_code: Sample preparation code(s) (default: ('S', ''))
            analyzed_size_frac: Analyzed size fraction(s) (default: ('<2 mm', ''))

        Returns:
            SDAResponse: Query results

        Raises:
            LDMParameterError: If invalid parameters provided
            LDMQueryError: If query execution fails
            LDMBackendSelectionError: If backend unavailable
        """
        # Validate parameters
        if x is not None and WHERE is not None:
            raise LDMParameterError("Cannot specify both 'x' and 'WHERE' parameters")

        # Convert single values to lists
        if x is not None and not isinstance(x, (list, tuple)):
            x = [x]

        # If no keys and no WHERE clause, return empty result
        if x is None and WHERE is None:
            logger.warning("No query criteria provided (no x or WHERE)")
            return SDAResponse({"Table": [[], []]})

        # Build site WHERE clause if x and what provided
        site_where = None
        if x is not None and not WHERE:
            # Use case-insensitive matching only when all values are strings
            # (numeric columns shouldn't have LOWER() for proper type handling)
            all_strings = all(isinstance(v, str) for v in x)
            site_where = in_condition(what, x, case_insensitive=all_strings)
        elif WHERE:
            site_where = WHERE

        # Get backend once - needed for both site query and dialect detection
        backend = await self._get_backend()
        dialect = "sqlite" if isinstance(backend, SQLiteBackend) else "sql_server"

        # Stage 1: Get pedon_keys from lab_combine_nasis_ncss
        pedon_keys = None
        if site_where:
            try:
                site_query = build_ldm_site_query(WHERE=site_where)
                logger.debug(f"Site query: {site_query}")
                site_response = await backend.execute(site_query)

                if not site_response.is_empty():
                    # Extract pedon_keys from response
                    response_data = site_response.to_dict()
                    if response_data:
                        # Each row is a dict with 'pedon_key' column
                        pedon_keys = [row["pedon_key"] for row in response_data]

                if not pedon_keys:
                    logger.warning("No pedons matched site query criteria")
                    return SDAResponse({"Table": [[], []]})

            except Exception as e:
                raise LDMQueryError(
                    f"Failed to query site data: {str(e)}",
                    details=str(e),
                ) from e

        # Stage 2: Query layer data
        try:
            query_builder = LDMQueryBuilder(
                tables=tables,
                layer_type=layer_type,
                area_type=area_type,
                prep_code=prep_code,
                analyzed_size_frac=analyzed_size_frac,
                dialect=dialect,
            )
        except Exception as e:
            raise LDMParameterError(f"Invalid query parameters: {str(e)}") from e

        # Use pedon_keys from site query, or the original x values
        # Ensure keys is a list for fetch_chunked type signature
        keys_for_layer = list(pedon_keys) if pedon_keys is not None else list(x or [])

        # Execute layer query with chunking and retry logic
        if keys_for_layer and len(keys_for_layer) > chunk_size:
            # Need to chunk
            backend = await self._get_backend()
            return await fetch_chunked(
                keys=keys_for_layer,
                build_query=lambda chunk: query_builder.build_query(
                    keys=chunk, key_column=bycol
                ),
                execute=backend.execute,
                chunk_size=chunk_size,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
            )
        else:
            # Single query
            sql = query_builder.build_query(keys=keys_for_layer, key_column=bycol)
            return await self._execute_query(sql, max_retries=max_retries)

    async def get_available_tables(self) -> list[str]:
        """Get list of available LDM tables in data source.

        Returns:
            List of table names

        Raises:
            LDMBackendSelectionError: If backend unavailable
        """
        backend = await self._get_backend()
        return await backend.get_tables()

    async def get_table_schema(self, table_name: str) -> dict[str, str]:
        """Get column names and types for an LDM table.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column names to their types

        Raises:
            LDMBackendSelectionError: If backend unavailable
        """
        backend = await self._get_backend()
        return await backend.get_columns(table_name)

    async def close(self) -> None:
        """Close the client and clean up resources."""
        if self._backend is not None:
            await self._backend.close()
        await super().close()

    # ========================================================================
    # Private Methods
    # ========================================================================

    async def _get_backend(
        self,
    ) -> Union[SDABackend, SQLiteBackend]:
        """Get or create backend instance.

        Selects backend based on parameters:
        - If dsn is provided: SQLiteBackend(path)
        - Otherwise: SDABackend(sda_client)

        Returns:
            Backend instance (SDABackend or SQLiteBackend)

        Raises:
            LDMBackendSelectionError: If backend cannot be initialized
        """
        if self._backend is None:
            try:
                if self.dsn is not None:
                    # SQLite backend for local database
                    self._backend = SQLiteBackend(self.dsn)
                else:
                    # SDA backend for web service
                    backend = SDABackend(client=self.sda_client)
                    # Ensure SDA client is initialized
                    await backend.connect()
                    self._backend = backend
            except Exception as e:
                raise LDMBackendSelectionError(
                    f"Failed to initialize LDM backend: {str(e)}"
                ) from e
        return self._backend

    async def _execute_query(
        self, sql: str, max_retries: int = DEFAULT_MAX_RETRIES
    ) -> SDAResponse:
        """Execute a single query with retry logic.

        Args:
            sql: SQL query string
            max_retries: Maximum number of retries

        Returns:
            SDAResponse: Query results

        Raises:
            LDMQueryError: If query fails after all retries
        """
        backend = await self._get_backend()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await backend.execute(sql)
                return response

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Query attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying... (attempt {attempt + 2}/{max_retries + 1})"
                    )
                    # Exponential backoff
                    delay = self._config.retry_delay * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        raise LDMQueryError(
            f"Query failed after {max_retries + 1} attempts",
            details=f"Last error: {str(last_error)}",
        ) from last_error
