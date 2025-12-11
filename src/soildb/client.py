"""
HTTP client for Soil Data Access web service.
"""

import asyncio
from typing import Optional, Union

import httpx

from .base_client import BaseDataAccessClient, ClientConfig
from .exceptions import (
    SDAConnectionError,
    SDAMaintenanceError,
    SDAQueryError,
    SDAResponseError,
    SDATimeoutError,
)
from .query import BaseQuery, Query
from .response import SDAResponse
from .utils import add_sync_version


class SDAClient(BaseDataAccessClient):
    """Async HTTP client for Soil Data Access web service."""

    def __init__(
        self,
        base_url: str = "https://sdmdataaccess.sc.egov.usda.gov",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[ClientConfig] = None,
    ):
        """
        Initialize SDA client.

        Can be initialized either with individual parameters or with a ClientConfig object.
        If config is provided, it takes precedence over individual parameters.

        Args:
            base_url: Base URL for SDA service (default: official SDA endpoint)
            timeout: Request timeout in seconds (default: 60.0)
            max_retries: Number of retries for failed requests (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            config: ClientConfig instance with timeout, retry, and URL settings.
                   If provided, takes precedence over individual parameters.

        Examples:
            >>> # Using individual parameters
            >>> client = SDAClient(timeout=120.0, max_retries=5)

            >>> # Using ClientConfig with presets
            >>> config = ClientConfig.reliable()
            >>> client = SDAClient(config=config)

            >>> # Using custom ClientConfig
            >>> config = ClientConfig(
            ...     base_url="https://custom.endpoint.gov",
            ...     timeout=90.0,
            ...     max_retries=4
            ... )
            >>> client = SDAClient(config=config)
        """
        if config is None:
            config = ClientConfig(
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        else:
            # If config is provided but base_url is explicitly different from default,
            # use the provided base_url (for backward compatibility)
            if base_url != "https://sdmdataaccess.sc.egov.usda.gov":
                config.base_url = base_url

        super().__init__(config)
        self.base_url = config.base_url.rstrip("/") if config.base_url else "https://sdmdataaccess.sc.egov.usda.gov"

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client with SDA-specific configuration.

        Returns:
            httpx.AsyncClient: Configured client with SDA headers
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "soildb-python-client/0.1.0",
            },
        )

    @add_sync_version
    async def close(self) -> None:  # type: ignore[override]
        """Close the HTTP client and clean up resources."""
        await super().close()


    @add_sync_version
    async def connect(self) -> bool:  # type: ignore[override]
        """
        Test connection to SDA service.

        Returns:
            True if connection successful

        Raises:
            SDAConnectionError: If connection fails
        """
        test_query = Query().select("COUNT(*)").from_("sacatalog").limit(1)

        try:
            response = await self.execute(test_query)
            return len(response) >= 0  # Even 0 results means connection worked
        except Exception as e:
            raise SDAConnectionError(f"Connection test failed: {e}") from e

    @add_sync_version
    async def execute(self, query: Union[BaseQuery, str]) -> SDAResponse:
        """
        Execute a query against SDA.

        Args:
            query: Query object or raw SQL string to execute

        Returns:
            SDAResponse containing query results

        Raises:
            SDAQueryError: If query execution fails
            SDAMaintenanceError: If service is under maintenance
            SDATimeoutError: If request times out
            SDAConnectionError: If connection fails
        """
        if isinstance(query, str):
            return await self.execute_sql(query)
        return await self._execute_query_obj(query)

    async def _execute_query_obj(self, query: BaseQuery) -> SDAResponse:
        sql = query.to_sql()
        return await self.execute_sql(sql)

    @add_sync_version
    async def execute_sql(self, sql: str) -> SDAResponse:
        """
        Execute a raw SQL query against SDA.

        Args:
            sql: The raw SQL query string.

        Returns:
            SDAResponse containing query results

        Raises:
            SDAQueryError: If query execution fails
            SDAMaintenanceError: If service is under maintenance
            SDATimeoutError: If request times out
            SDAConnectionError: If connection fails
        """
        request_body = {"query": sql, "format": "json+columnname+metadata"}

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._make_request(
                    "POST",
                    f"{self.base_url}/tabular/post.rest",
                    json=request_body,
                )

                # Check HTTP status and handle SDA errors
                if response.status_code != 200:
                    error_details = (
                        response.text.strip()
                        if response.text
                        else response.reason_phrase
                    )

                    # Try to extract meaningful error message from SDA response
                    if response.status_code == 400:
                        # SDA returns detailed error info in 400 responses
                        if "Invalid column name" in error_details:
                            raise SDAQueryError(
                                "Invalid column name in query. Check table schema.",
                                query=sql,
                                details=error_details,
                            )
                        elif "Invalid object name" in error_details:
                            raise SDAQueryError(
                                "Invalid table name in query. Check table exists.",
                                query=sql,
                                details=error_details,
                            )
                        elif (
                            "Syntax error" in error_details
                            or "syntax" in error_details.lower()
                        ):
                            raise SDAQueryError(
                                "SQL syntax error in query.",
                                query=sql,
                                details=error_details,
                            )
                        else:
                            raise SDAQueryError(
                                "Query failed with 400 error.",
                                query=sql,
                                details=error_details,
                            )
                    elif response.status_code == 500:
                        raise SDAQueryError(
                            "Server error (500). Query may be too complex or hit resource limits.",
                            query=sql,
                            details=error_details,
                        )
                    else:
                        raise SDAConnectionError(
                            f"HTTP {response.status_code}: {response.reason_phrase}",
                            details=error_details,
                        )

                response_text = response.text

                # Check for maintenance message
                if "Site is under daily maintenance" in response_text:
                    raise SDAMaintenanceError(
                        "SDA service is currently under maintenance. Please try again later."
                    )

                # Parse response
                try:
                    return SDAResponse.from_json(response_text)
                except SDAResponseError as e:
                    raise SDAQueryError(
                        f"Failed to parse SDA response: {e}", query=sql
                    ) from e

            except httpx.TimeoutException:
                if attempt < self._config.max_retries:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise SDATimeoutError(
                        f"Request timed out after {self._config.timeout} seconds"
                    ) from None

            except httpx.NetworkError as e:
                if attempt < self._config.max_retries:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise SDAConnectionError(f"Network error: {e}") from e

            except Exception as e:
                # Don't retry on other exceptions
                raise SDAQueryError(f"Query execution failed: {e}", query=sql) from e

        # Should never reach here, but just in case
        raise SDAQueryError("Maximum retries exceeded", query=sql)

    @add_sync_version
    async def execute_many(self, queries: list[BaseQuery]) -> list[SDAResponse]:
        """
        Execute multiple queries concurrently.

        Args:
            queries: List of query objects to execute

        Returns:
            List of SDAResponse objects in same order as input queries
        """
        tasks = [self.execute(query) for query in queries]
        return await asyncio.gather(*tasks)
