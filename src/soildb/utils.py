"""
Internal utility functions for soildb.
"""
import asyncio
import inspect
from functools import wraps
from typing import Awaitable, Callable, TypeVar

from .exceptions import SyncUsageError

R = TypeVar("R")

def add_sync_version(async_fn: Callable[..., Awaitable[R]]) -> Callable[..., R]:
    """
    A decorator that adds a .sync attribute to an async function, allowing it
    to be called synchronously.

    The .sync version runs the async function in a new asyncio event loop.
    """

    async def _call_and_cleanup(async_fn, args, kwargs, temp_client):
        try:
            return await async_fn(*args, **kwargs)
        finally:
            if temp_client:
                await temp_client.close()

    def sync_wrapper(*args, **kwargs) -> R:
        """
        Synchronous wrapper for the async function.
        """
        from .client import SDAClient
        
        temp_client = None
        # Check if the function has a 'client' parameter and it's not provided
        sig = inspect.signature(async_fn)
        if 'client' in sig.parameters and 'client' not in kwargs:
            temp_client = SDAClient()
            kwargs['client'] = temp_client
        
        try:
            return asyncio.run(_call_and_cleanup(async_fn, args, kwargs, temp_client))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                raise SyncUsageError(
                    "Cannot use .sync() from within an existing asyncio event loop. "
                    "Use the async version of this function instead."
                ) from e
            else:
                raise

    # Attach the synchronous wrapper to the original async function
    setattr(async_fn, "sync", sync_wrapper)
    return async_fn