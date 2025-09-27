"""
Utility functions that add value beyond basic QueryBuilder usage.
"""

import threading
from typing import Optional

from .client import SDAClient
from .query import Query, QueryBuilder
from .response import SDAResponse
from .spatial import spatial_query

# Module-level client instance for shared usage
# WARNING: The default client is not thread-safe for concurrent access.
# In multi-threaded or multi-task asyncio applications, provide explicit
# SDAClient instances to convenience functions instead of relying on the
# shared default client to avoid race conditions and connection issues.
_default_client: Optional[SDAClient] = None
_default_client_lock = threading.Lock()


def _get_default_client() -> SDAClient:
    """
    Get or create the default client instance.

    WARNING: This function uses thread-level locking for client initialization,
    but the SDAClient instance itself is not designed for concurrent access
    across multiple threads or asyncio tasks. In production applications with
    concurrent requests, provide explicit SDAClient instances to avoid
    connection state corruption and HTTP client issues.

    For asyncio applications, consider using task-local or context-local
    client instances instead of the global default.
    """
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            # Double-check pattern for thread safety during initialization
            if _default_client is None:
                _default_client = SDAClient()
    return _default_client


async def get_mapunit_by_areasymbol(
    areasymbol: str, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Get map unit data by survey area symbol (legend).

    Args:
        areasymbol: Survey area symbol (e.g., 'IA015') to retrieve map units for
        client: Optional client instance

    Returns:
        SDAResponse containing map unit data for the specified survey area
    """
    if client is None:
        client = _get_default_client()

    query = QueryBuilder.mapunits_by_legend(areasymbol)
    return await client.execute(query)


async def get_mapunit_by_point(
    longitude: float, latitude: float, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Get map unit data at a specific point location.

    Args:
        longitude: Longitude of the point
        latitude: Latitude of the point
        client: Optional client instance

    Returns:
        SDAResponse containing map unit data at the specified point
    """
    wkt_point = f"POINT({longitude} {latitude})"
    return await spatial_query(wkt_point, table="mupolygon", client=client)


async def get_mapunit_by_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get map unit data within a bounding box.

    Args:
        min_x: Western boundary (longitude)
        min_y: Southern boundary (latitude)
        max_x: Eastern boundary (longitude)
        max_y: Northern boundary (latitude)
        client: Optional client instance

    Returns:
        SDAResponse containing map unit data
    """
    if client is None:
        client = _get_default_client()

    query = QueryBuilder.mapunits_intersecting_bbox(min_x, min_y, max_x, max_y)
    return await client.execute(query)


async def get_sacatalog(
    columns: Optional[list[str]] = None, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Get survey area catalog (sacatalog) data.

    Args:
        columns: List of columns to return. If None, returns ['areasymbol', 'areaname', 'saversion']
        client: Optional client instance

    Returns:
        SDAResponse containing sacatalog data

    Examples:
        # Get basic survey area info
        response = await get_sacatalog()
        df = response.to_pandas()  # areasymbol, areaname, saversion

        # Get all available columns
        response = await get_sacatalog(columns=['areasymbol', 'areaname', 'saversion', 'saverest'])
        df = response.to_pandas()

        # Get just survey area symbols (equivalent to old list_survey_areas)
        response = await get_sacatalog(columns=['areasymbol'])
        symbols = response.to_pandas()['areasymbol'].tolist()
    """
    if client is None:
        client = _get_default_client()

    if columns is None:
        columns = ["areasymbol", "areaname", "saversion"]

    query = Query().select(*columns).from_("sacatalog").order_by("areasymbol")

    return await client.execute(query)


async def list_survey_areas(client: Optional[SDAClient] = None) -> list[str]:
    """
    Get a simple list of all survey area symbols.

    .. deprecated::
        Use get_sacatalog() instead. This function is maintained for backward compatibility.

    Returns:
        List of survey area symbols (e.g., ['IA015', 'IA109', ...])
    """
    response = await get_sacatalog(columns=["areasymbol"], client=client)
    df = response.to_pandas()
    return df["areasymbol"].tolist() if not df.empty else []


# Synchronous convenience wrappers for simple operations
def get_sacatalog_sync(
    columns: Optional[list[str]] = None, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Synchronous version of get_sacatalog().

    Get survey area catalog (sacatalog) data.

    Args:
        columns: List of columns to return. If None, returns ['areasymbol', 'areaname', 'saversion']
        client: Optional client instance

    Returns:
        SDAResponse containing sacatalog data

    Note:
        This function creates a new event loop if one doesn't exist.
        For better performance in async applications, use get_sacatalog() instead.
    """
    import asyncio

    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to use run_until_complete
            # This is less efficient but works in Jupyter notebooks, etc.
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(get_sacatalog(columns, client))
        else:
            return loop.run_until_complete(get_sacatalog(columns, client))
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(get_sacatalog(columns, client))


def list_survey_areas_sync(client: Optional[SDAClient] = None) -> list[str]:
    """
    Synchronous version of list_survey_areas().

    Get a simple list of all survey area symbols.

    .. deprecated::
        Use get_sacatalog_sync() instead. This function is maintained for backward compatibility.

    Returns:
        List of survey area symbols (e.g., ['IA015', 'IA109', ...])

    Note:
        This function creates a new event loop if one doesn't exist.
        For better performance in async applications, use list_survey_areas() instead.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(list_survey_areas(client))
        else:
            return loop.run_until_complete(list_survey_areas(client))
    except RuntimeError:
        return asyncio.run(list_survey_areas(client))
