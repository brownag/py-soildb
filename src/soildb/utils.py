"""
Internal utility functions for soildb.
"""

import asyncio
import functools
import inspect
import types
from collections.abc import Awaitable
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

R = TypeVar("R")
T = TypeVar("T")


def _client_supplied(sig: inspect.Signature, args: tuple, kwargs: dict) -> bool:
    """Check if a client parameter was supplied and is not None.

    Args:
        sig: Function signature.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        True only if client is bound (positionally or by keyword) and not None.
    """
    client_param = sig.parameters.get("client")
    if not client_param:
        return False

    try:
        bound = sig.bind_partial(*args, **kwargs)
        if "client" in bound.arguments:
            return bound.arguments["client"] is not None
    except TypeError:
        # bind_partial failed; check kwargs only
        if "client" in kwargs:
            return kwargs["client"] is not None

    return False


def require_client(client: Optional[T]) -> T:
    """Raise TypeError if client is None, otherwise return it unchanged.

    Args:
        client: The client parameter value (may be None).

    Returns:
        The client if not None.

    Raises:
        TypeError: If client is None.
    """
    if client is None:
        raise TypeError(
            "client is required; call through @add_sync_version or pass a client"
        )
    return client


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
    ) -> R:
        """Run an async function synchronously in a fresh event loop.

        Args:
            async_fn: Async function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

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

        # Create and run coroutine
        async def _call() -> R:
            return await async_fn(*args, **kwargs)

        # Run the coroutine
        try:
            return asyncio.run(_call())
        except RuntimeError:
            # Fallback for environments where asyncio.run() doesn't work
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_call())
            finally:
                loop.close()

    @staticmethod
    def extract_client_class(annotation: Any) -> Optional[type]:
        """Extract client class from type annotation.

        For a Union, returns a class only when exactly one non-None member
        exists and it is a class. Otherwise returns None.

        Args:
            annotation: Type annotation to extract class from

        Returns:
            Client class if found and unambiguous, None otherwise
        """
        if annotation is None:
            return None

        origin = get_origin(annotation)
        is_union = origin is Union or (
            hasattr(types, "UnionType") and origin is types.UnionType
        )
        if is_union:
            args = get_args(annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1 and isinstance(non_none_args[0], type):
                return non_none_args[0]
            return None
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

    The async wrapper owns the client lifecycle: if a function has a `client`
    parameter but no client is provided (positional or keyword), the wrapper
    creates one and closes it after execution.

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

    @functools.wraps(async_fn)
    async def async_wrapper(*args: Any, **kwargs: Any) -> R:
        """Async wrapper that manages client lifecycle."""
        sig = inspect.signature(async_fn)
        client_param = sig.parameters.get("client")

        # Determine if client was supplied (not None)
        client_supplied = _client_supplied(sig, args, kwargs)

        # Create client if needed
        temp_client = None
        if client_param and not client_supplied:
            client_class = AsyncSyncBridge.extract_client_class(client_param.annotation)
            if client_class:
                temp_client = client_class()

                # Bind arguments to handle positional client=None replacement
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    if "client" in bound.arguments:
                        # Client was bound positionally as None; replace it
                        bound.arguments["client"] = temp_client
                        args = bound.args
                        kwargs = bound.kwargs
                    else:
                        # Client not bound; add it as keyword argument
                        kwargs["client"] = temp_client
                except TypeError:
                    # bind_partial failed; add as keyword argument
                    kwargs["client"] = temp_client

        # Execute function with automatic cleanup
        try:
            return await async_fn(*args, **kwargs)
        finally:
            if temp_client:
                await temp_client.close()

    def sync_wrapper(*args: Any, **kwargs: Any) -> R:
        """Synchronous wrapper for the async function."""
        return AsyncSyncBridge.run_async(async_wrapper, args=args, kwargs=kwargs)

    # Attach the synchronous wrapper to the async wrapper
    async_wrapper.sync = sync_wrapper  # type: ignore
    return async_wrapper
