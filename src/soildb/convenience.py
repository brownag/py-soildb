"""
Utility functions that add value beyond basic query building.
"""

import json
import warnings
from typing import Any, Literal, Optional, Union, cast

from . import query_templates
from .client import SDAClient
from .fetch import fetch_pedons_by_bbox
from .query import ColumnSets, Query
from .response import SDAResponse
from .sanitization import sanitize_sql_string
from .spatial import spatial_query
from .utils import add_sync_version, require_client


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
    query = query_templates.query_mapunits_by_legend(areasymbol, columns)
    response = await require_client(client).execute(query)

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
    query = query_templates.query_mapunits_intersecting_bbox(
        min_x, min_y, max_x, max_y, columns
    )
    return await require_client(client).execute(query)


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
    query = query_templates.query_available_survey_areas(columns)
    return await require_client(client).execute(query)


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
    bbox = (min_x, min_y, max_x, max_y)
    return await fetch_pedons_by_bbox(bbox, columns, client=client)  # type: ignore


LAB_PEDON_ID_COLUMNS = ("pedon_key", "pedoniid", "upedonid", "pedlabsampnum")
LabPedonIdColumn = Literal["pedon_key", "pedoniid", "upedonid", "pedlabsampnum"]


@add_sync_version
async def get_lab_pedon(
    x: Union[str, int],
    what: LabPedonIdColumn = "pedon_key",
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get laboratory-analyzed pedon data by one kind of pedon identifier.

    ``pedon_key`` and ``pedoniid`` are unique. A ``upedonid`` lookup may return
    several pedons.

    Args:
        x: Identifier value to look up
        what: Which identifier ``x`` is. One of:
            - 'pedon_key': Pedon key (numeric, unique record in lab data)
            - 'pedoniid': NASIS pedon record ID, i.e. NASIS ``peiid`` (unique)
            - 'upedonid': Pedon ID assigned by the describer (not unique)
            - 'pedlabsampnum': Lab pedon number (e.g. '85P0234')
        columns: List of columns to return. If None, returns basic pedon columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing lab pedon data

    Raises:
        ValueError: If ``what`` is not a supported identifier column

    Examples:
        >>> response = await get_lab_pedon("S1999NY061001", what="upedonid")
        >>> response = get_lab_pedon.sync("85P0234", what="pedlabsampnum")
    """
    if what not in LAB_PEDON_ID_COLUMNS:
        raise ValueError(f"what must be one of {LAB_PEDON_ID_COLUMNS}, got {what!r}")

    if what == "pedon_key":
        query = query_templates.query_pedon_by_pedon_key(str(x), columns)
    else:
        query = (
            Query()
            .select(*(columns or ColumnSets.PEDON_BASIC))
            .from_("lab_combine_nasis_ncss")
            .where(f"{what} = {sanitize_sql_string(str(x))}")
        )

    return await require_client(client).execute(query)


@add_sync_version
async def get_lab_pedon_by_id(
    pedon_id: str,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """
    Get a laboratory-analyzed pedon by pedon key, falling back to pedon ID.

    .. deprecated:: 0.9.0
        The fallback is ambiguous when a value is both a pedon key and a
        pedon ID. Use ``get_lab_pedon(x, what=...)`` instead.

    Args:
        pedon_id: Pedon key or pedon ID (``upedonid``)
        columns: List of columns to return. If None, returns basic pedon columns
        client: Optional SDA client instance. If not provided, a temporary client is created and closed automatically.

    Returns:
        SDAResponse containing lab pedon data
    """
    warnings.warn(
        "get_lab_pedon_by_id() is deprecated; use "
        'get_lab_pedon(x, what="pedon_key" | "pedoniid" | "upedonid" | "pedlabsampnum")',
        DeprecationWarning,
        stacklevel=2,
    )
    return await _get_lab_pedon_key_then_id(pedon_id, columns, client)


async def _get_lab_pedon_key_then_id(
    pedon_id: str,
    columns: Optional[list[str]] = None,
    client: Optional[SDAClient] = None,
) -> "SDAResponse":
    """Legacy lookup: try ``pedon_id`` as a pedon key, then as a ``upedonid``."""
    client = require_client(client)

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
async def _query_json_auto(
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

    Args:
        query: Query object (preferred, safely built with Query builder) or raw
               SQL string. If passing a raw string, ensure all values are
               properly escaped using sanitization helpers (sanitize_sql_string,
               etc.) to prevent SQL injection.
        client: Optional SDA client instance. If not provided, a temporary
                client is created and closed automatically.

    Returns:
        List of records as dicts, one dict per row.

    Security Note:
        Query objects are inherently safe (built with the Query builder).
        Raw SQL strings are treated as-is and the caller is responsible for
        ensuring they are properly escaped.

    Examples:
        records = _query_json_auto.sync(query)
        df = pd.DataFrame(records)
    """
    base_sql = query if isinstance(query, str) else query.to_sql()

    json_sql = f"~DeclareVarchar(@json,max)~;WITH src (n) AS ({base_sql} FOR JSON AUTO) SELECT @json = src.n FROM src SELECT @json, LEN(@json);"
    response = await require_client(client).execute_sql(json_sql)

    if response.is_empty():
        return []

    return cast(list[dict[str, Any]], json.loads(response.data[0][0]))
