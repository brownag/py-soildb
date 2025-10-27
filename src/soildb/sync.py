"""
Synchronous wrapper functions and utilities for soildb.

This module provides synchronous versions of async functions for users who
prefer blocking calls or cannot use async/await syntax. Under the hood, these
functions use asyncio to run async code synchronously.

Sync wrappers are provided for:
- Convenience functions (get_mapunit_by_*, etc.)
- Fetch functions (fetch_*_by_*, etc.)
- High-level utilities (list_survey_areas, etc.)

Usage:
    # Instead of this async code:
    async with SDAClient() as client:
        response = await client.execute(query)

    # Use this sync code:
    from soildb.sync import get_mapunit_by_areasymbol_sync
    response = get_mapunit_by_areasymbol_sync("IA015")
"""

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, get_args, get_origin

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
            raise RuntimeError(
                "Cannot use sync version from within an existing asyncio event loop. "
                "Use the async version instead."
            )
        except RuntimeError:
            pass

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
        except RuntimeError as e:
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
            non_none_args = [arg for arg in args if arg != type(None)]
            if non_none_args:
                arg = non_none_args[0]
                if isinstance(arg, type):
                    return arg
        else:
            if isinstance(annotation, type):
                return annotation

        return None


# Sync versions of convenience functions
# These provide the primary synchronous API for soildb


def get_mapunit_by_areasymbol_sync(
    areasymbol: str,
    columns: Optional[list[str]] = None,
    client: Optional[Any] = None,
) -> "SDAResponse":
    """Synchronous version of get_mapunit_by_areasymbol.

    Get map unit data by survey area symbol (legend).

    Args:
        areasymbol: Survey area symbol (e.g., 'IA015') to retrieve map units for
        columns: List of columns to return. If None, returns basic map unit columns
        client: SDA client instance. If not provided, creates temporary client

    Returns:
        SDAResponse containing map unit data for the specified survey area

    Examples:
        >>> response = get_mapunit_by_areasymbol_sync("IA015")
        >>> df = response.to_pandas()
    """
    from .convenience import get_mapunit_by_areasymbol
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        get_mapunit_by_areasymbol,
        args=(areasymbol,),
        kwargs={"columns": columns, "client": client} if client else {"columns": columns},
        client_class=SDAClient if not client else None,
    )


def get_mapunit_by_point_sync(
    longitude: float,
    latitude: float,
    columns: Optional[list[str]] = None,
    client: Optional[Any] = None,
) -> "SDAResponse":
    """Synchronous version of get_mapunit_by_point.

    Get map unit data at a specific point location.

    Args:
        longitude: Point longitude coordinate
        latitude: Point latitude coordinate
        columns: List of columns to return. If None, returns basic map unit columns
        client: SDA client instance. If not provided, creates temporary client

    Returns:
        SDAResponse containing map unit data for the specified point

    Examples:
        >>> response = get_mapunit_by_point_sync(-93.6, 42.0)
        >>> df = response.to_pandas()
    """
    from .convenience import get_mapunit_by_point
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        get_mapunit_by_point,
        args=(longitude, latitude),
        kwargs={"columns": columns, "client": client} if client else {"columns": columns},
        client_class=SDAClient if not client else None,
    )


def get_mapunit_by_bbox_sync(
    north: float,
    south: float,
    east: float,
    west: float,
    columns: Optional[list[str]] = None,
    client: Optional[Any] = None,
) -> "SDAResponse":
    """Synchronous version of get_mapunit_by_bbox.

    Get map units within a bounding box.

    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary
        east: Eastern longitude boundary
        west: Western longitude boundary
        columns: List of columns to return. If None, returns basic map unit columns
        client: SDA client instance. If not provided, creates temporary client

    Returns:
        SDAResponse containing map unit data within the bounding box

    Examples:
        >>> response = get_mapunit_by_bbox_sync(43.0, 41.0, -93.0, -95.0)
        >>> df = response.to_pandas()
    """
    from .convenience import get_mapunit_by_bbox
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        get_mapunit_by_bbox,
        args=(north, south, east, west),
        kwargs={"columns": columns, "client": client} if client else {"columns": columns},
        client_class=SDAClient if not client else None,
    )


# Sync versions of fetch functions


def fetch_mapunit_by_areasymbol_sync(
    areasymbols: list[str],
    columns: Optional[list[str]] = None,
    chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Synchronous version of fetch_mapunit_by_areasymbol.

    Fetch map units for multiple survey areas with pagination.

    Args:
        areasymbols: List of survey area symbols
        columns: Columns to fetch. If None, uses defaults
        chunk_size: Number of areas to fetch per request

    Returns:
        List of map unit dictionaries

    Examples:
        >>> results = fetch_mapunit_by_areasymbol_sync(["IA015", "IA109"])
    """
    from .fetch import fetch_mapunit_by_areasymbol
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        fetch_mapunit_by_areasymbol,
        args=(areasymbols,),
        kwargs={"columns": columns, "chunk_size": chunk_size},
        client_class=SDAClient,
    )


def fetch_component_by_mukey_sync(
    mukeys: list[int],
    columns: Optional[list[str]] = None,
    chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Synchronous version of fetch_component_by_mukey.

    Fetch soil components for multiple map units.

    Args:
        mukeys: List of map unit keys
        columns: Columns to fetch. If None, uses defaults
        chunk_size: Number of mukeys to fetch per request

    Returns:
        List of component dictionaries

    Examples:
        >>> results = fetch_component_by_mukey_sync([123456, 123457])
    """
    from .fetch import fetch_component_by_mukey
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        fetch_component_by_mukey,
        args=(mukeys,),
        kwargs={"columns": columns, "chunk_size": chunk_size},
        client_class=SDAClient,
    )


def fetch_component_by_comppct_r_sync(
    comppct_r: float,
    comparison: str = ">=",
    columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Synchronous version of fetch_component_by_comppct_r.

    Fetch components by composition percentage.

    Args:
        comppct_r: Composition percentage threshold
        comparison: Comparison operator ('>=', '>', '<=', '<', '==')
        columns: Columns to fetch. If None, uses defaults

    Returns:
        List of component dictionaries

    Examples:
        >>> results = fetch_component_by_comppct_r_sync(50)
    """
    from .fetch import fetch_component_by_comppct_r
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        fetch_component_by_comppct_r,
        args=(comppct_r,),
        kwargs={"comparison": comparison, "columns": columns},
        client_class=SDAClient,
    )


def fetch_horizon_by_cokey_sync(
    cokeys: list[int],
    columns: Optional[list[str]] = None,
    chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Synchronous version of fetch_horizon_by_cokey.

    Fetch horizons for multiple components.

    Args:
        cokeys: List of component keys
        columns: Columns to fetch. If None, uses defaults
        chunk_size: Number of cokeys to fetch per request

    Returns:
        List of horizon dictionaries

    Examples:
        >>> results = fetch_horizon_by_cokey_sync([1, 2, 3])
    """
    from .fetch import fetch_horizon_by_cokey
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        fetch_horizon_by_cokey,
        args=(cokeys,),
        kwargs={"columns": columns, "chunk_size": chunk_size},
        client_class=SDAClient,
    )


def fetch_pedons_by_bbox_sync(
    north: float,
    south: float,
    east: float,
    west: float,
    columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Synchronous version of fetch_pedons_by_bbox.

    Fetch pedons within a bounding box.

    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary
        east: Eastern longitude boundary
        west: Western longitude boundary
        columns: Columns to fetch. If None, uses defaults

    Returns:
        List of pedon dictionaries

    Examples:
        >>> results = fetch_pedons_by_bbox_sync(43.0, 41.0, -93.0, -95.0)
    """
    from .fetch import fetch_pedons_by_bbox
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        fetch_pedons_by_bbox,
        args=(north, south, east, west),
        kwargs={"columns": columns},
        client_class=SDAClient,
    )


# Sync versions of high-level functions


def list_survey_areas_sync(client: Optional[Any] = None) -> list[dict[str, Any]]:
    """Synchronous version of list_survey_areas.

    Get list of available survey areas.

    Args:
        client: SDA client instance. If not provided, creates temporary client

    Returns:
        List of survey area dictionaries with area_symbol and area_name

    Examples:
        >>> areas = list_survey_areas_sync()
        >>> for area in areas[:5]:
        ...     print(area['areasymbol'], area['areaname'])
    """
    from .high_level import list_survey_areas
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        list_survey_areas,
        kwargs={"client": client},
        client_class=SDAClient if not client else None,
    )


def get_sacatalog_sync(client: Optional[Any] = None) -> list[dict[str, Any]]:
    """Synchronous version of get_sacatalog.

    Get the Soil & Agricultural Commodity Catalog.

    Args:
        client: SDA client instance. If not provided, creates temporary client

    Returns:
        List of catalog entries with soil and crop information

    Examples:
        >>> catalog = get_sacatalog_sync()
        >>> len(catalog)
    """
    from .high_level import get_sacatalog
    from .client import SDAClient

    return AsyncSyncBridge.run_async(
        get_sacatalog,
        kwargs={"client": client},
        client_class=SDAClient if not client else None,
    )


__all__ = [
    "AsyncSyncBridge",
    "get_mapunit_by_areasymbol_sync",
    "get_mapunit_by_point_sync",
    "get_mapunit_by_bbox_sync",
    "fetch_mapunit_by_areasymbol_sync",
    "fetch_component_by_mukey_sync",
    "fetch_component_by_comppct_r_sync",
    "fetch_horizon_by_cokey_sync",
    "fetch_pedons_by_bbox_sync",
    "list_survey_areas_sync",
    "get_sacatalog_sync",
]
