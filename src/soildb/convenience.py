"""
Utility functions that add value beyond basic QueryBuilder usage.
"""

from typing import Optional

from .client import SDAClient
from .fetch import fetch_pedons_by_bbox
from .query import ColumnSets, Query, QueryBuilder
from .response import SDAResponse
from .spatial import spatial_query


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
        client: SDA client instance (required)

    Returns:
        SDAResponse containing map unit data for the specified survey area

    Raises:
        TypeError: If client is not provided
    """
    if client is None:
        raise TypeError("client parameter is required")

    query = QueryBuilder.mapunits_by_legend(areasymbol, columns)
    return await client.execute(query)


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
        client: SDA client instance (required)

    Returns:
        SDAResponse containing map unit data at the specified point

    Raises:
        TypeError: If client is not provided
    """
    if client is None:
        raise TypeError("client parameter is required")

    # Convert columns list to comma-separated string for spatial_query
    what = ", ".join(columns) if columns else None
    wkt_point = f"POINT({longitude} {latitude})"
    return await spatial_query(wkt_point, table="mupolygon", what=what, client=client)


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
        client: SDA client instance (required)

    Returns:
        SDAResponse containing map unit data

    Raises:
        TypeError: If client is not provided
    """
    if client is None:
        raise TypeError("client parameter is required")

    query = QueryBuilder.mapunits_intersecting_bbox(min_x, min_y, max_x, max_y, columns)
    return await client.execute(query)


async def get_sacatalog(
    columns: Optional[list[str]] = None, client: Optional[SDAClient] = None
) -> "SDAResponse":
    """
    Get survey area catalog (sacatalog) data.

    Args:
        columns: List of columns to return. If None, returns ['areasymbol', 'areaname', 'saversion']
        client: SDA client instance (required)

    Returns:
        SDAResponse containing sacatalog data

    Raises:
        TypeError: If client is not provided

    Examples:
        # Get basic survey area info
        client = SDAClient()
        response = await get_sacatalog(client=client)
        df = response.to_pandas()  # areasymbol, areaname, saversion

        # Get all available columns
        response = await get_sacatalog(columns=['areasymbol', 'areaname', 'saversion', 'saverest'], client=client)
        df = response.to_pandas()

        # Get just survey area symbols
        response = await get_sacatalog(columns=['areasymbol'], client=client)
        df = response.to_pandas()
        symbols = df['areasymbol'].tolist()
    """
    if client is None:
        raise TypeError("client parameter is required")

    query = QueryBuilder.available_survey_areas(columns)
    return await client.execute(query)


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
        client: SDA client instance (required)

    Returns:
        SDAResponse containing lab pedon data

    Raises:
        TypeError: If client is not provided
    """
    if client is None:
        raise TypeError("client parameter is required")

    bbox = (min_x, min_y, max_x, max_y)
    return await fetch_pedons_by_bbox(bbox, columns, client=client)  # type: ignore


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
        client: SDA client instance (required)

    Returns:
        SDAResponse containing lab pedon data

    Raises:
        TypeError: If client is not provided
    """
    if client is None:
        raise TypeError("client parameter is required")

    # First try as pedon_key
    query = QueryBuilder.pedon_by_pedon_key(pedon_id, columns)
    response = await client.execute(query)

    if not response.is_empty():
        return response

    # If not found, try as user pedon ID
    query = (
        Query()
        .select(*(columns or ColumnSets.PEDON_BASIC))
        .from_("lab_combine_nasis_ncss")
        .where(f"upedonid = '{pedon_id}'")
    )

    return await client.execute(query)
