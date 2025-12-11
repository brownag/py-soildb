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

R = TypeVar("R")


class AsyncSyncBridge:
    """Handles conversion of async functions to synchronous versions.

    This class provides utilities for running async code synchronously,
    managing event loops, and handling client instantiation.
    """

    @staticmethod
    def run_async(
        async_fn: Callable[..., Awaitable[R]],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        client_class: Optional[type] = None,
    ) -> R:
        """Run an async function synchronously.

        Args:
            async_fn: Async function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            client_class: Optional client class to instantiate if not provided

        Returns:
            Result of running the async function

        Raises:
            RuntimeError: If called from within an existing event loop
        """
        if kwargs is None:
            kwargs = {}

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - good, we can proceed
            pass
        else:
            # A loop is running - we can't use sync version here
            raise RuntimeError(
                "Cannot use sync version from within an existing asyncio event loop. "
                "Use the async version instead."
            )

        # Handle automatic client instantiation
        temp_client = None
        if client_class:
            sig = inspect.signature(async_fn)
            client_param = sig.parameters.get("client")
            if client_param and "client" not in kwargs:
                temp_client = client_class()
                kwargs["client"] = temp_client

        # Create coroutine
        async def _call_and_cleanup() -> R:
            try:
                return await async_fn(*args, **kwargs)
            finally:
                if temp_client:
                    await temp_client.close()

        # Run the coroutine
        try:
            return asyncio.run(_call_and_cleanup())
        except RuntimeError:
            # Fallback for environments where asyncio.run() doesn't work
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_call_and_cleanup())
            finally:
                loop.close()

    @staticmethod
    def extract_client_class(annotation: Any) -> Optional[type]:
        """Extract client class from type annotation.

        Handles Optional, Union, and direct type annotations.

        Args:
            annotation: Type annotation to extract class from

        Returns:
            Client class if found, None otherwise
        """
        if annotation is None:
            return None

        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                arg = non_none_args[0]
                if isinstance(arg, type):
                    return arg
        else:
            if isinstance(annotation, type):
                return annotation

        return None


def add_sync_version(
    async_fn: Callable[..., Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """
    A decorator that adds a .sync attribute to an async function, allowing it
    to be called synchronously.

    The .sync version runs the async function in a new asyncio event loop.

    Example:
        >>> @add_sync_version
        ... async def my_async_func(x):
        ...     return x * 2

        >>> # Async usage
        >>> result = await my_async_func(5)

        >>> # Sync usage
        >>> result = my_async_func.sync(5)
    """

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
