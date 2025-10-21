"""
Internal utility functions for soildb.
"""
import asyncio
import inspect
from functools import wraps
from typing import Awaitable, Callable, TypeVar, get_origin, get_args, Union

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
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            raise SyncUsageError(
                "Cannot use .sync() from within an existing asyncio event loop. "
                "Use the async version of this function instead."
            )
        except RuntimeError:
            # No running loop, proceed
            pass
        
        temp_client = None
        # Check if the function has a 'client' parameter and it's not provided
        sig = inspect.signature(async_fn)
        client_param = sig.parameters.get('client')
        if client_param and 'client' not in kwargs:
            # Extract client class from type annotation
            client_class = _extract_client_class(client_param.annotation)
            if client_class:
                temp_client = client_class()
                kwargs['client'] = temp_client
        
        if temp_client:
            coro = _call_and_cleanup(async_fn, args, kwargs, temp_client)
        else:
            coro = async_fn(*args, **kwargs)
        
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            # Try creating a new event loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # Attach the synchronous wrapper to the original async function
    setattr(async_fn, "sync", sync_wrapper)
    return async_fn


def _extract_client_class(annotation) -> type:
    """
    Extract the client class from a type annotation.
    
    Handles cases like:
    - Optional[SDAClient] -> SDAClient
    - SDAClient -> SDAClient
    - Union[SDAClient, None] -> SDAClient
    """
    if annotation is None:
        return None
    
    # Handle Union/Optional types
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # For Optional[T] which is Union[T, None], get the non-None type
        for arg in args:
            if arg is not type(None):  # Skip None
                return arg
    else:
        # Direct type annotation
        return annotation
    
    return None