"""
HTTP client for Soil Data Access web service.
"""

import asyncio
from typing import Any, Optional, Union

import httpx

from .exceptions import (
    SDAConnectionError,
    SDAMaintenanceError,
    SDAQueryError,
    SDAResponseError,
    SDATimeoutError,
)
from .query import BaseQuery, Query
from .response import SDAResponse


class SDAClient:
    """Async HTTP client for Soil Data Access web service."""

    def __init__(
        self,
        base_url: str = "https://sdmdataaccess.sc.egov.usda.gov",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize SDA client.

        Args:
            base_url: Base URL for SDA service
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "SDAClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: type, exc_val: Exception, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        current_loop = asyncio.get_running_loop()
        
        # If we have a client but it's from a different event loop, close it and recreate
        if self._client is not None and self._event_loop is not None and self._event_loop != current_loop:
            try:
                await self._client.aclose()
            except Exception:
                pass  # Ignore errors when closing
            self._client = None
            self._event_loop = None
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "soildb-python-client/0.1.0",
                },
            )
            self._event_loop = current_loop

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._event_loop = None

    async def connect(self) -> bool:
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
        await self._ensure_client()
        assert self._client is not None  # _ensure_client should have set this

        request_body = {"query": sql, "format": "json+columnname+metadata"}

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/tabular/post.rest", json=request_body
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
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise SDATimeoutError(
                        f"Request timed out after {self.timeout} seconds"
                    ) from None

            except httpx.NetworkError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise SDAConnectionError(f"Network error: {e}") from e

            except Exception as e:
                # Don't retry on other exceptions
                raise SDAQueryError(f"Query execution failed: {e}", query=sql) from e

        # Should never reach here, but just in case
        raise SDAQueryError("Maximum retries exceeded", query=sql)

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
