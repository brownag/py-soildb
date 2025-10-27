"""
Abstract base class for data access clients.

This module defines the common interface and shared functionality for all client types
(SDAClient, AWDBClient) in the soildb package. The base class handles connection lifecycle,
retry logic, and error handling patterns consistently across all clients.

## Architecture

The base class follows this design pattern:

1. **Async Context Manager**: All clients are async context managers that handle setup/teardown
2. **Lazy Initialization**: Clients use lazy initialization for HTTP connections
3. **Event Loop Safety**: Clients detect event loop changes and reconnect if needed
4. **Retry Logic**: Configurable exponential backoff for transient failures
5. **Consistent Error Handling**: All clients raise errors through their namespace (SDAConnectionError, AWDBConnectionError)

## Usage

```python
# Direct instantiation with context manager
async with SDAClient(config=ClientConfig(timeout=60.0)) as client:
    result = await client.execute(query)

# Manual lifecycle
client = AWDBClient(config=ClientConfig(max_retries=5))
await client.connect()
try:
    result = await client.get_stations()
finally:
    await client.close()
```

## Configuration

ClientConfig dataclass provides unified configuration across all clients:

- **timeout**: Request timeout in seconds
- **max_retries**: Maximum number of retries for transient failures
- **retry_delay**: Base delay between retries (exponential backoff applied)
- **base_url**: Base URL for the service (defaults vary by client type)

Example preset configurations are provided via class methods:
- `ClientConfig.default()`: Standard settings (60s timeout, 3 retries)
- `ClientConfig.fast()`: Quick requests (30s timeout, 1 retry)
- `ClientConfig.reliable()`: Network resilient (120s timeout, 5 retries)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class ClientConfig:
    """Unified configuration for data access clients.

    Attributes:
        timeout: Request timeout in seconds (default: 60.0)
        max_retries: Maximum number of retries for transient failures (default: 3)
        retry_delay: Base delay between retries in seconds (default: 1.0)
                    Exponential backoff is applied: delay * (attempt + 1)
        base_url: Base URL for the service endpoint (varies by client type)

    Examples:
        >>> config = ClientConfig(timeout=30.0, max_retries=5)
        >>> config_fast = ClientConfig.fast()
        >>> config_reliable = ClientConfig.reliable()
    """

    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    base_url: str = ""

    @classmethod
    def default(cls) -> "ClientConfig":
        """Create a ClientConfig with default settings.

        Returns:
            ClientConfig: Standard settings (60s timeout, 3 retries, 1s base delay)
        """
        return cls(timeout=60.0, max_retries=3, retry_delay=1.0)

    @classmethod
    def fast(cls) -> "ClientConfig":
        """Create a ClientConfig optimized for fast/local operations.

        Returns:
            ClientConfig: Quick settings (30s timeout, 1 retry, 0.5s base delay)
        """
        return cls(timeout=30.0, max_retries=1, retry_delay=0.5)

    @classmethod
    def reliable(cls) -> "ClientConfig":
        """Create a ClientConfig optimized for network resilience.

        Returns:
            ClientConfig: Resilient settings (120s timeout, 5 retries, 1s base delay)
        """
        return cls(timeout=120.0, max_retries=5, retry_delay=1.0)


class BaseDataAccessClient(ABC):
    """Abstract base class for data access clients.

    Provides common functionality for all client types:
    - Async context manager support
    - HTTP client lifecycle management
    - Event loop tracking and safety
    - Shared retry logic framework
    - Error handling patterns

    Subclasses should:
    1. Store client-specific configuration in `_config`
    2. Implement `_create_http_client()` to customize httpx.AsyncClient setup
    3. Override `_handle_error()` for client-specific error mapping
    4. Implement service-specific methods (e.g., `execute()`, `get_stations()`)

    The base class handles:
    - Ensuring HTTP client is initialized (`_ensure_client()`)
    - Detecting event loop changes and reconnecting
    - Cleanup of resources (`close()`)
    - Context manager protocol (`__aenter__`, `__aexit__`)
    """

    def __init__(self, config: ClientConfig):
        """Initialize the base client.

        Args:
            config: ClientConfig instance with timeout, retry, and URL settings
        """
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "BaseDataAccessClient":
        """Async context manager entry.

        Ensures HTTP client is initialized and returns self.

        Returns:
            BaseDataAccessClient: Returns self for use in `async with` block
        """
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: type, exc_val: Exception, exc_tb: Any) -> None:
        """Async context manager exit.

        Ensures HTTP client is closed and resources are cleaned up.

        Args:
            exc_type: Exception type if an exception occurred in the context
            exc_val: Exception instance if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized and ready.

        This method handles:
        1. Detecting if event loop has changed
        2. Recreating client if necessary
        3. Lazy initialization on first call

        Event loop changes can occur when:
        - Client is used across different async contexts
        - Jupyter notebooks restart kernels
        - Tests use different event loop fixtures

        The method detects these changes and automatically recreates
        the HTTP client for the new event loop.

        Raises:
            RuntimeError: If called outside async context (no running event loop)
        """
        current_loop = asyncio.get_running_loop()

        # If we have a client but it's from a different event loop, close and recreate
        if (
            self._client is not None
            and self._event_loop is not None
            and self._event_loop != current_loop
        ):
            try:
                await self._client.aclose()
            except Exception:
                pass  # Ignore errors when closing old client
            self._client = None
            self._event_loop = None

        # Create client if not initialized
        if self._client is None:
            self._client = self._create_http_client()
            self._event_loop = current_loop

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create and configure an httpx.AsyncClient instance.

        This method can be overridden by subclasses to customize:
        - Default headers (User-Agent, Accept, Content-Type)
        - Timeout configuration
        - SSL/TLS settings
        - Proxy configuration
        - Response event hooks

        Returns:
            httpx.AsyncClient: Configured HTTP client instance

        Default implementation creates a basic client with:
        - Timeout from ClientConfig
        - Generic User-Agent header
        - JSON content type headers
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            headers={
                "User-Agent": "soildb-client/0.1.0",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client and clean up resources.

        This method safely closes the HTTP client if it exists.
        It's safe to call multiple times - subsequent calls are no-ops.

        Raises:
            Exception: If an error occurs while closing the client
                      (but generally this is silently handled)
        """
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass  # Silently ignore errors during close
            finally:
                self._client = None
                self._event_loop = None

    @abstractmethod
    async def connect(self) -> bool:
        """Test connection to the service.

        Subclasses should implement this to verify the service is accessible.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionError: If connection fails (specific type depends on client)
        """
        ...

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic and error handling.

        This method implements exponential backoff retry logic for transient failures
        (timeouts and network errors). It's designed to be used by subclasses
        for consistent error handling and retry behavior.

        Retry Strategy:
        - Retried on: httpx.TimeoutException, httpx.NetworkError
        - Not retried on: httpx.HTTPStatusError, other exceptions
        - Backoff: exponential (delay * (attempt + 1))
        - Max attempts: max_retries + 1

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: Full URL for the request
            **kwargs: Additional arguments passed to httpx.AsyncClient method

        Returns:
            httpx.Response: The HTTP response object

        Raises:
            httpx.TimeoutException: If all retry attempts timeout
            httpx.HTTPStatusError: On HTTP error status codes (not retried)
            httpx.NetworkError: If all retry attempts fail with network errors
            httpx.RequestError: For other request errors
        """
        await self._ensure_client()
        assert self._client is not None  # _ensure_client should always set this

        for attempt in range(self._config.max_retries + 1):
            try:
                # Make the request
                if method.upper() == "GET":
                    response = await self._client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await self._client.post(url, **kwargs)
                elif method.upper() == "PUT":
                    response = await self._client.put(url, **kwargs)
                elif method.upper() == "DELETE":
                    response = await self._client.delete(url, **kwargs)
                else:
                    response = await self._client.request(method, url, **kwargs)

                return response

            except httpx.TimeoutException:
                # Timeout is retried
                if attempt < self._config.max_retries:
                    # Exponential backoff
                    delay = self._config.retry_delay * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

            except httpx.NetworkError:
                # Network errors are retried
                if attempt < self._config.max_retries:
                    # Exponential backoff
                    delay = self._config.retry_delay * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

        # Should never reach here
        raise httpx.RequestError("Maximum retries exceeded")
