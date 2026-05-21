"""
Utility functions that add value beyond basic query building.
"""

import json
from typing import Any, Optional, Union

from . import query_templates
from .client import SDAClient
from .fetch import fetch_pedons_by_bbox
from .query import ColumnSets, Query
from .response import SDAResponse
from .sanitization import sanitize_sql_string
from .spatial import spatial_query
from .utils import add_sync_version


@add_sync_version
async def get_mapunit_by_areasymbol(
    areasymbol: str,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get map unit data by survey area symbol (legend).

    Args:
        areasymbol: Survey area symbol (e.g., 'IA015') to retrieve map units for
        columns: List of columns to return. If None, returns basic map unit columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing map unit data for the specified survey area

    Examples:
        # Async usage without explicit client (automatic)
        response = await get_mapunit_by_areasymbol("IA015")

        # Sync usage (automatic client management)
        response = get_mapunit_by_areasymbol.sync("IA015")

        # With explicit client
        async with SDAClient() as client:
            response = await get_mapunit_by_areasymbol("IA015", client=client)
    """
    if client is None:
        client = SDAClient()

    query = query_templates.query_mapunits_by_legend(areasymbol, columns)
    response = await client.execute(query)

    return response


@add_sync_version
async def get_mapunit_by_point(
    longitude: float,
    latitude: float,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get map unit data at a specific point location.

    Args:
        longitude: Longitude of the point
        latitude: Latitude of the point
        columns: List of columns to return. If None, returns basic map unit columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing map unit data at the specified point
    """
    if client is None:
        client = SDAClient()

    # Convert columns list to comma-separated string for spatial_query
    what = ", ".join(columns) if columns else None
    wkt_point = f"POINT({longitude} {latitude})"
    return await spatial_query(wkt_point, table="mupolygon", what=what, client=client)


@add_sync_version
async def get_mapunit_by_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get map unit data within a bounding box.

    Args:
        min_x: Western boundary (longitude)
        min_y: Southern boundary (latitude)
        max_x: Eastern boundary (longitude)
        max_y: Northern boundary (latitude)
        columns: List of columns to return. If None, returns basic map unit columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing map unit data
    """
    if client is None:
        client = SDAClient()

    query = query_templates.query_mapunits_intersecting_bbox(
        min_x, min_y, max_x, max_y, columns
    )
    return await client.execute(query)


@add_sync_version
async def get_sacatalog(
    columns: Optional[list[str]] = None, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Get survey area catalog (sacatalog) data.

    Args:
        columns: List of columns to return. If None, returns ['areasymbol', 'areaname', 'saversion']
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing sacatalog data

    Examples:
        # Async usage without explicit client (automatic)
        response = await get_sacatalog()
        df = response.to_pandas()  # areasymbol, areaname, saversion

        # Sync usage (automatic client management)
        response = get_sacatalog.sync()
        df = response.to_pandas()

        # Get specific columns
        response = await get_sacatalog(columns=['areasymbol', 'areaname'])
        df = response.to_pandas()
        symbols = df['areasymbol'].tolist()
    """
    if client is None:
        client = SDAClient()

    query = query_templates.query_available_survey_areas(columns)
    return await client.execute(query)


@add_sync_version
async def get_lab_pedons_by_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get laboratory-analyzed pedon data within a bounding box.

    Args:
        min_x: Western boundary (longitude)
        min_y: Southern boundary (latitude)
        max_x: Eastern boundary (longitude)
        max_y: Northern boundary (latitude)
        columns: List of columns to return. If None, returns basic pedon columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing lab pedon data
    """
    if client is None:
        client = SDAClient()

    bbox = (min_x, min_y, max_x, max_y)
    return await fetch_pedons_by_bbox(bbox, columns, client=client)  # type: ignore


@add_sync_version
async def get_lab_pedon_by_id(
    pedon_id: str,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get a single laboratory-analyzed pedon by its pedon key or user pedon ID.

    Args:
        pedon_id: Pedon key or user pedon ID
        columns: List of columns to return. If None, returns basic pedon columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing lab pedon data
    """
    if client is None:
        client = SDAClient()

    # First try as pedon_key
    query = query_templates.query_pedon_by_pedon_key(pedon_id, columns)
    response = await client.execute(query)

    if not response.is_empty():
        return response

    # If not found, try as user pedon ID
    query = (
        Query()
        .select(*(columns or ColumnSets.PEDON_BASIC))
        .from_("lab_combine_nasis_ncss")
        .where(f"upedonid = {sanitize_sql_string(pedon_id)}")
    )

    return await client.execute(query)


@add_sync_version
async def query_unlimited(
    query: Union[Query, str],
    client: Optional[SDAClient] = None,
) -> list[dict[str, Any]]:
    """
    Execute a query without SDA's 100,000 record limit using FOR JSON AUTO.

    SDA truncates results at 100,000 rows. This function wraps the query in a
    subquery with ``FOR JSON AUTO``, which causes SQL Server to return results
    as concatenated JSON string fragments rather than individual tabular rows.
    SDA's row-count limit applies to tabular rows, so the JSON output bypasses
    it entirely.

    Note: ORDER BY in the inner query is not supported — SQL Server disallows
    ORDER BY inside a subquery unless TOP or OFFSET-FETCH is also present.

    Args:
        query: Query object or raw SQL string to execute.
        client: Optional SDA client instance. If not provided, a temporary
                client is created and closed automatically.

    Returns:
        List of records as dicts, one dict per row.

    Examples:
        # Async
        records = await query_unlimited(
            Query().select("mukey", "muname").from_("mapunit")
        )

        # Sync
        records = query_unlimited.sync(
            Query().select("mukey", "muname").from_("mapunit")
        )
    """
    if client is None:
        client = SDAClient()

    base_sql = query if isinstance(query, str) else query.to_sql()

    # Wrap in FOR JSON AUTO to bypass the 100k row limit.
    # SQL Server streams the JSON output as many short string fragments
    # (~2033 chars each), so SDA returns them as multiple short rows rather
    # than hitting its row-count ceiling.
    json_sql = f"SELECT * FROM ({base_sql}) AS _q FOR JSON AUTO"

    response = await client.execute_sql(json_sql)

    if response.is_empty():
        return []

    # Concatenate all JSON string fragments and parse into a list of dicts.
    fragments = [row[0] for row in response.data if row and row[0] is not None]
    return json.loads("".join(fragments))
