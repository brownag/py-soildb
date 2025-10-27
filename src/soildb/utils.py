"""
Internal utility functions for soildb.
"""

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    TypeVar,
)

R = TypeVar("R")


def add_sync_version(
    async_fn: Callable[..., Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """
    A decorator that adds a .sync attribute to an async function, allowing it
    to be called synchronously.

    The .sync version runs the async function in a new asyncio event loop.

    Note: This decorator is maintained for backward compatibility. For new code,
    consider using the dedicated sync wrapper functions in soildb.sync module.

    Example:
        >>> @add_sync_version
        ... async def my_async_func(x):
        ...     return x * 2

        >>> # Async usage
        >>> result = await my_async_func(5)

        >>> # Sync usage
        >>> result = my_async_func.sync(5)
    """
    # Import here to avoid circular imports
    from .sync import AsyncSyncBridge

    def sync_wrapper(*args: Any, **kwargs: Any) -> R:
        """Synchronous wrapper for the async function."""
        # Check if the function has a 'client' parameter and extract client class
        sig = inspect.signature(async_fn)
        client_param = sig.parameters.get("client")
        client_class = None

        if client_param and "client" not in kwargs:
            # Extract client class from type annotation
            client_class = AsyncSyncBridge.extract_client_class(client_param.annotation)

        return AsyncSyncBridge.run_async(
            async_fn, args=args, kwargs=kwargs, client_class=client_class
        )

    # Attach the synchronous wrapper to the original async function
    async_fn.sync = sync_wrapper  # type: ignore
    return async_fn
