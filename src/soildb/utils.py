"""
Internal utility functions for soildb.
"""

import asyncio
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from .exceptions import SyncUsageError

R = TypeVar("R")


def add_sync_version(
    async_fn: Callable[..., Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """
    A decorator that adds a .sync attribute to an async function, allowing it
    to be called synchronously.

    The .sync version runs the async function in a new asyncio event loop.
    """

    async def _call_and_cleanup(
        async_fn: Callable[..., Awaitable[R]],
        args: tuple,
        kwargs: dict,
        temp_client: Optional[Any],
    ) -> R:
        try:
            return await async_fn(*args, **kwargs)
        finally:
            if temp_client:
                await temp_client.close()

    def sync_wrapper(*args: Any, **kwargs: Any) -> R:
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
        client_param = sig.parameters.get("client")
        if client_param and "client" not in kwargs:
            # Extract client class from type annotation
            client_class = _extract_client_class(client_param.annotation)
            if client_class:
                temp_client = client_class()
                kwargs["client"] = temp_client

        coro: Awaitable[R]
        if temp_client:
            coro = _call_and_cleanup(async_fn, args, kwargs, temp_client)
        else:
            coro = async_fn(*args, **kwargs)

        try:
            # asyncio.run accepts any Awaitable since Python 3.7
            return asyncio.run(coro)  # type: ignore[arg-type]
        except RuntimeError:
            # Try creating a new event loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # Attach the synchronous wrapper to the original async function
    async_fn.sync = sync_wrapper  # type: ignore
    return async_fn


def _extract_client_class(annotation: Any) -> Optional[type]:
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
        non_none_args = [arg for arg in args if arg != type(None)]
        if non_none_args:
            arg = non_none_args[0]
            # Ensure we return a type, not just any object
            if isinstance(arg, type):
                return arg
    else:
        # Direct type annotation - ensure it's actually a type
        if isinstance(annotation, type):
            return annotation

    return None
