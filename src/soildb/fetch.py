"""
Bulk data fetching with automatic pagination and abstraction levels.

This module provides a hierarchical API for fetching large SSURGO datasets:

TIER 1 - PRIMARY INTERFACE (Use for most cases):
  fetch_by_keys() - Universal key-based fetcher with pagination support
    - Flexible: works with any SSURGO table and key column
    - Recommended: Use this unless you need specialized behavior
    - Performance: Automatic chunking, concurrent requests

TIER 2 - SPECIALIZED CONVENIENCE FUNCTIONS (Use for specific tables):
  fetch_mapunit_polygon() - Map unit polygons (mukey)
  fetch_component_by_mukey() - Components (mukey)
  fetch_chorizon_by_cokey() - Horizons (cokey)
  fetch_survey_area_polygon() - Survey area boundaries (areasymbol)
    - Deprecated: These wrap fetch_by_keys() for specific tables
    - Migration: Use fetch_by_keys() with appropriate table/key_column
    - Rationale: Pre-fetch convenience, but fetch_by_keys() is simpler

TIER 3 - COMPLEX MULTI-STEP FETCHES:
  fetch_pedons_by_bbox() - Lab pedons with optional site+horizon data
  fetch_pedon_horizons() - Horizon data for pedon sites
    - Complex: Multi-table joins, optional geometry, custom return types
    - Keep: Significant value over raw queries

TIER 4 - KEY LOOKUP HELPERS (For planning complex fetches):
  get_mukey_by_areasymbol() - Discover all mukeys in survey areas
  get_cokey_by_mukey() - Discover all cokeys in map units
    - Use before multi-step fetches to plan key lists
    - Small results: Immediate execution (no chunking)

ARCHITECTURE DIAGRAM:

    User Query
        ↓
    ┌─────────────────────────────────────┐
    │ fetch_by_keys()                     │ ← PRIMARY (use this)
    │ (handles all SSURGO tables)         │
    └─────────────────────────────────────┘
        ↑                    ↑
        │                    └── _fetch_chunk() [internal]
        │                         ↑
        ├── fetch_mapunit_polygon()     │
        ├── fetch_component_by_mukey()  │ ← TIER 2 (deprecated, wrap
        ├── fetch_chorizon_by_cokey()   │   fetch_by_keys)
        └── fetch_survey_area_polygon() │
    
    ┌─────────────────────────────────────┐
    │ fetch_pedons_by_bbox()              │ ← TIER 3 (complex)
    │ fetch_pedon_horizons()              │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │ get_mukey_by_areasymbol()           │ ← TIER 4 (helpers)
    │ get_cokey_by_mukey()                │
    └─────────────────────────────────────┘

RECOMMENDED USAGE PATTERNS:

1. Simple fetch by keys (MOST COMMON):
   >>> response = await fetch_by_keys([123, 456], "component")
   
2. For common tables with specific column needs:
   >>> response = await fetch_by_keys(mukeys, "mapunit", columns=["mukey", "muname"])
   
3. Discover keys for multi-step operations:
   >>> mukeys = await get_mukey_by_areasymbol(["IA001"])
   >>> components = await fetch_by_keys(mukeys, "component")
   
4. Complex operations with relationships:
   >>> spc = await fetch_pedons_by_bbox(bbox, return_type="soilprofilecollection")
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
import warnings

from .client import SDAClient
from .exceptions import SoilDBError
from .query import Query, QueryBuilder
from .response import SDAResponse
from .sanitization import sanitize_sql_numeric, sanitize_sql_string_list
from .schema_system import SCHEMAS, get_schema
from .utils import add_sync_version

logger = logging.getLogger(__name__)

# Common SSURGO tables and their typical key columns
TABLE_KEY_MAPPING = {
    # Core tables
    "legend": "lkey",
    "mapunit": "mukey",
    "component": "cokey",
    "chorizon": "chkey",
    "chfrags": "chfragkey",
    "chtexturegrp": "chtgkey",
    "chtexture": "chtkey",
    # Spatial tables
    "mupolygon": "mukey",
    "sapolygon": "areasymbol",  # or lkey
    "mupoint": "mukey",
    "muline": "mukey",
    "featpoint": "featkey",
    "featline": "featkey",
    # Interpretation tables
    "cointerp": "cokey",
    "chinterp": "chkey",
    "copmgrp": "copmgrpkey",
    "corestrictions": "reskeyid",
    # Administrative
    "sacatalog": "areasymbol",
    "laoverlap": "lkey",
    "legendtext": "lkey",
}


class FetchError(SoilDBError):
    """Raised when key-based fetching fails."""

    def __str__(self) -> str:
        """Return helpful fetch error message."""
        if "Unknown table" in self.message:
            return f"{self.message} Supported tables include: {', '.join(TABLE_KEY_MAPPING.keys())}"
        elif "No responses to combine" in self.message:
            return "No data was returned from the fetch operation. This may indicate invalid keys or an empty result set."
        return self.message


class QueryPresets:
    """
    Predefined query configurations for common SSURGO fetching patterns.

    This class provides convenient preset configurations for frequently-used queries,
    eliminating the need for separate functions like fetch_component_by_mukey().
    Use these presets to configure fetch_by_keys() with optimal defaults.

    **DESIGN RATIONALE**:
    Instead of having many similar functions (fetch_component_by_mukey,
    fetch_chorizon_by_cokey, etc.), QueryPresets provides named configurations
    that can be passed to fetch_by_keys(). This reduces code duplication while
    providing the same convenience.

    **USAGE EXAMPLES**:
        # Use preset configuration
        >>> preset = QueryPresets.COMPONENT
        >>> response = await fetch_by_keys(
        ...     mukeys, preset.table, preset.key_column,
        ...     columns=preset.columns, chunk_size=preset.chunk_size,
        ...     include_geometry=preset.include_geometry
        ... )

        # Or unpack preset as kwargs
        >>> response = await fetch_by_keys(mukeys, **preset.as_kwargs())

    **AVAILABLE PRESETS**:
    - MAPUNIT: Map unit core data
    - COMPONENT: Component data (keyed by mukey)
    - CHORIZON: Component horizon data (keyed by cokey)
    - MUPOLYGON: Map unit polygons with geometry
    - SAPOLYGON: Survey area boundaries with geometry
    - COINTERP: Component interpretations
    - CHINTERP: Horizon interpretations

    See Also:
        fetch_by_keys() - Main function these presets configure
    """

    class _Preset:
        """Internal preset configuration container."""

        def __init__(
            self,
            table: str,
            key_column: str,
            columns: Optional[List[str]] = None,
            chunk_size: int = 1000,
            include_geometry: bool = False,
            description: str = "",
        ):
            self.table = table
            self.key_column = key_column
            self.columns = columns
            self.chunk_size = chunk_size
            self.include_geometry = include_geometry
            self.description = description

        def as_kwargs(self) -> Dict[str, Any]:
            """Return preset as kwargs dict for fetch_by_keys()."""
            return {
                "table": self.table,
                "key_column": self.key_column,
                "columns": self.columns,
                "chunk_size": self.chunk_size,
                "include_geometry": self.include_geometry,
            }

        def __repr__(self) -> str:
            return f"QueryPreset(table={self.table}, key_column={self.key_column}, chunk_size={self.chunk_size})"

    # MAPUNIT core data (all key columns + basic metadata)
    MAPUNIT = _Preset(
        table="mapunit",
        key_column="mukey",
        columns=["mukey", "muname", "mustatus", "muacres", "mucomppct_r"],
        chunk_size=1000,
        description="Map unit core data (name, status, acres, composition %)",
    )

    # COMPONENT data (core component properties, keyed by mukey)
    COMPONENT = _Preset(
        table="component",
        key_column="mukey",
        columns=["cokey", "mukey", "compname", "comppct_r", "majcompflag"],
        chunk_size=1000,
        description="Component data (name, percent, major flag)",
    )

    # COMPONENT with detailed taxonomic/chemical data
    COMPONENT_DETAILED = _Preset(
        table="component",
        key_column="mukey",
        columns=[
            "cokey",
            "mukey",
            "compname",
            "comppct_r",
            "majcompflag",
            "taxclname",
            "hydgrp",
        ],
        chunk_size=800,
        description="Component data with taxonomic and hydrologic group",
    )

    # CHORIZON data (horizon properties, keyed by cokey)
    CHORIZON = _Preset(
        table="chorizon",
        key_column="cokey",
        columns=[
            "chkey",
            "cokey",
            "hzname",
            "hzdept_r",
            "hzdepb_r",
            "texture",
        ],
        chunk_size=500,
        description="Horizon data (depth, texture)",
    )

    # CHORIZON with detailed chemical/physical properties
    CHORIZON_DETAILED = _Preset(
        table="chorizon",
        key_column="cokey",
        columns=[
            "chkey",
            "cokey",
            "hzname",
            "hzdept_r",
            "hzdepb_r",
            "texture",
            "claytotal_r",
            "sandtotal_r",
            "silttotal_r",
            "om_r",
            "ph1to1h2o_r",
        ],
        chunk_size=300,
        description="Horizon data with texture, clay%, sand%, silt%, OM, pH",
    )

    # MUPOLYGON (map unit boundaries with geometry)
    MUPOLYGON = _Preset(
        table="mupolygon",
        key_column="mukey",
        include_geometry=True,
        chunk_size=200,
        description="Map unit boundaries with WKT polygon geometry",
    )

    # SAPOLYGON (survey area boundaries with geometry)
    SAPOLYGON = _Preset(
        table="sapolygon",
        key_column="areasymbol",
        include_geometry=True,
        chunk_size=50,
        description="Survey area boundaries with WKT polygon geometry",
    )

    # COINTERP (component interpretations)
    COINTERP = _Preset(
        table="cointerp",
        key_column="cokey",
        columns=["cokey", "cointerpiid", "interpname", "interphr"],
        chunk_size=500,
        description="Component interpretations (land use ratings, suitability)",
    )

    # CHINTERP (horizon interpretations)
    CHINTERP = _Preset(
        table="chinterp",
        key_column="chkey",
        columns=["chkey", "chinterpiid", "interpname", "interphr"],
        chunk_size=500,
        description="Horizon interpretations",
    )

    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """
        Get all available presets with descriptions.

        Returns:
            Dict mapping preset name to description

        Example:
            >>> presets = QueryPresets.list_presets()
            >>> for name, desc in presets.items():
            ...     print(f"{name}: {desc}")
        """
        presets = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, cls._Preset):
                presets[attr_name] = attr.description
        return presets


@add_sync_version
async def fetch_by_keys(
    keys: Union[Sequence[Union[str, int]], str, int],
    table: str,
    key_column: Optional[str] = None,
    columns: Optional[Union[str, List[str]]] = None,
    chunk_size: int = 1000,
    include_geometry: bool = False,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch data from a table using a list of key values with pagination (PRIMARY INTERFACE).

    This is the canonical function for bulk key-based fetching from SSURGO. It handles
    all table types, automatic pagination, and concurrent requests. Use this for most
    data fetching operations unless you need specialized behavior.

    **WHEN TO USE THIS (Primary Interface)**:
    - You have a list of database keys (mukeys, cokeys, areasymbols, etc.)
    - You want to fetch data from any SSURGO table
    - You need customizable column selection
    - Standard use case for bulk operations

    **DESIGN - Abstraction Levels**:
    - TIER 1: fetch_by_keys() - Universal interface (RECOMMENDED)
    - TIER 2: fetch_component_by_mukey(), etc. - Deprecated wrappers
    - Migration: These Tier 2 functions wrap fetch_by_keys() for backward compatibility

    **WHEN NOT TO USE**:
    - For single records: Use Query + client.execute() directly
    - For spatial queries: Use spatial_query()
    - For complex multi-table operations: Use fetch_pedons_by_bbox() or fetch_pedon_horizons()

    **PERFORMANCE NOTES**:
    - Uses concurrent requests for chunked fetches (chunk_size < total_keys)
    - Recommended chunk_size: 500-2000 keys depending on key length and network
    - For very large datasets (>10,000 keys), consider processing in batches
    - Geometry inclusion increases response size significantly (~3-5x larger)
    - Optimization: Smaller chunk_size for long keys or slow network

    **PARAMETER GUIDE**:
    - keys: Single key (string/int) or list of keys
    - table: SSURGO table name (mapunit, component, chorizon, mupolygon, sapolygon, etc.)
    - key_column: Column to match keys against (auto-detected from table if None)
    - columns: Specific columns to retrieve (all columns if None)
    - chunk_size: Keys per query (default 1000, try 500-2000)
    - include_geometry: Add WKT geometry for spatial tables
    - client: Optional SDAClient instance (creates one if None)

    **TABLE KEY MAPPING** (auto-detected):
    - mapunit → mukey
    - component → cokey
    - chorizon → chkey
    - mupolygon → mukey
    - sapolygon → areasymbol
    - featpoint → featkey
    - And many others (see TABLE_KEY_MAPPING)

    **COLUMN SELECTION STRATEGIES**:
    - Default (None): Uses schema-defined default columns for table
    - List: ["mukey", "muname", "mustatus"] - explicit columns
    - String: "mukey, muname, mustatus" - comma-separated columns

    Args:
        keys: Key value(s) to fetch (single key or list of keys, e.g., mukeys, cokeys, areasymbols)
        table: Target SSURGO table name
        key_column: Column name for the key (auto-detected if None)
        columns: Columns to select (default: all columns from schema, or key columns if no schema)
        chunk_size: Number of keys to process per query (default: 1000, recommended: 500-2000)
        include_geometry: Whether to include geometry as WKT for spatial tables
        client: Optional SDA client instance (creates temporary client if None)

    Returns:
        SDAResponse: Combined query results with all matching rows

    Raises:
        FetchError: If keys list is empty, unknown table, or network error
        TypeError: If keys/table parameters are invalid

    Examples:
        # Fetch map unit data for specific mukeys (RECOMMENDED)
        >>> mukeys = [123456, 123457, 123458]
        >>> response = await fetch_by_keys(mukeys, "mapunit")
        >>> df = response.to_pandas()

        # With custom columns
        >>> response = await fetch_by_keys(
        ...     mukeys, "mapunit",
        ...     columns=["mukey", "muname", "muacres"]
        ... )

        # Fetch components with map unit information
        >>> response = await fetch_by_keys(
        ...     mukeys, "component",
        ...     key_column="mukey",
        ...     columns=["cokey", "compname", "comppct_r"]
        ... )

        # Large dataset with optimization
        >>> large_keys = list(range(100000, 110000))  # 10,000 keys
        >>> response = await fetch_by_keys(
        ...     large_keys, "chorizon",
        ...     key_column="cokey",
        ...     chunk_size=500,  # Smaller chunks for large lists
        ...     client=my_client
        ... )
        >>> df = response.to_pandas()
        >>> print(f"Fetched {len(df)} horizon records")

        # Fetch polygons with geometry for mapping
        >>> response = await fetch_by_keys(
        ...     ["IA001", "IA002"], "sapolygon",
        ...     key_column="areasymbol",
        ...     include_geometry=True
        ... )
        >>> gdf = response.to_geodataframe()  # Convert to GeoDataFrame
        >>> gdf.plot()  # Map the survey area boundaries

    **MIGRATION FROM DEPRECATED FUNCTIONS**:
    Instead of: Use:
        fetch_mapunit_polygon(mukeys) → fetch_by_keys(mukeys, "mupolygon")
        fetch_component_by_mukey(mukeys) → fetch_by_keys(mukeys, "component", "mukey")
        fetch_chorizon_by_cokey(cokeys) → fetch_by_keys(cokeys, "chorizon", "cokey")
        fetch_survey_area_polygon(areas) → fetch_by_keys(areas, "sapolygon", "areasymbol")

    **ADVANCED USAGE**:
    For complex workflows combining multiple queries, consider:
    - Using get_cokey_by_mukey() to discover keys before fetching
    - Using fetch_pedons_by_bbox() for multi-table operations
    - Custom Query building for non-key-based filtering

    See Also:
        fetch_by_keys_sync() - Synchronous version
        fetch_pedons_by_bbox() - For complex multi-table operations
        fetch_pedon_horizons() - For pedon horizon data
        get_cokey_by_mukey() - Discover keys before fetching
        get_mukey_by_areasymbol() - Discover keys before fetching
    """
    if isinstance(keys, (str, int)):
        keys = cast(List[Union[str, int]], [keys])

    keys_list = cast(List[Union[str, int]], keys)

    if not keys_list:
        raise FetchError("The 'keys' parameter cannot be an empty list.")

    if client is None:
        raise TypeError("client parameter is required")

    # Auto-detect key column if not provided
    if key_column is None:
        key_column = TABLE_KEY_MAPPING.get(table.lower())
        if key_column is None:
            raise FetchError(
                f"Unknown table '{table}'. Please specify key_column parameter."
            )

    if columns is None:
        select_columns = "*"
    elif isinstance(columns, list):
        select_columns = ", ".join(columns)
    else:
        select_columns = columns

    # Add geometry column for spatial tables if requested
    if include_geometry:
        geom_column = _get_geometry_column_for_table(table)
        if geom_column:
            if select_columns == "*":
                select_columns = f"*, {geom_column}.STAsText() as geometry"
            else:
                select_columns = (
                    f"{select_columns}, {geom_column}.STAsText() as geometry"
                )

    key_strings = [_format_key_for_sql(key) for key in keys_list]

    num_chunks = math.ceil(len(key_strings) / chunk_size)

    if num_chunks == 1:
        # Single query for small key lists
        return await _fetch_chunk(
            key_strings, table, key_column, select_columns, client
        )
    else:
        # Multiple queries for large key lists
        logger.debug(
            f"Fetching {len(keys_list)} keys in {num_chunks} chunks of {chunk_size}"
        )

        # Create chunks
        chunks = [
            key_strings[i : (i + chunk_size)]
            for i in range(0, len(key_strings), chunk_size)
        ]

        # Execute all chunks concurrently
        chunk_tasks = [
            _fetch_chunk(chunk_keys, table, key_column, select_columns, client)
            for chunk_keys in chunks
        ]

        chunk_responses = await asyncio.gather(*chunk_tasks)

        # Combine all responses
        return _combine_responses(chunk_responses)


async def _fetch_chunk(
    key_strings: List[str],
    table: str,
    key_column: str,
    select_columns: str,
    client: SDAClient,
) -> SDAResponse:
    """Fetch a single chunk of keys."""
    # Build IN clause
    keys_in_clause = ", ".join(key_strings)
    where_clause = f"{key_column} IN ({keys_in_clause})"

    # Build and execute query
    query = (
        Query()
        .select(*[col.strip() for col in select_columns.split(",")])
        .from_(table)
        .where(where_clause)
    )

    return await client.execute(query)


def _combine_responses(responses: List[SDAResponse]) -> SDAResponse:
    """Combine multiple SDAResponse objects into one."""
    if not responses:
        raise FetchError("No responses to combine")

    if len(responses) == 1:
        return responses[0]

    # Combine data from all responses
    combined_data = []
    for response in responses:
        combined_data.extend(response.data)

    # Create new response with combined data
    # Reconstruct the raw data format that SDAResponse expects
    first_response = responses[0]

    # Build the combined table in SDA format
    combined_table = []

    # Add the header row (column names)
    combined_table.append(first_response.columns)

    # Add the metadata row
    combined_table.append(first_response.metadata)

    # Add all the combined data rows
    combined_table.extend(combined_data)

    # Create new raw data structure
    combined_raw_data = {"Table": combined_table}

    # Create and return new SDAResponse
    return SDAResponse(combined_raw_data)


def _format_key_for_sql(key: Union[str, int]) -> str:
    """Format a key value for use in SQL IN clause."""
    if isinstance(key, str):
        # Escape single quotes and wrap in quotes
        escaped_key = key.replace("'", "''")
        return f"'{escaped_key}'"
    else:
        # Numeric keys don't need quotes
        return str(key)


def _get_geometry_column_for_table(table: str) -> Optional[str]:
    """Get the geometry column name for a spatial table."""
    geometry_columns = {
        "mupolygon": "mupolygongeo",
        "sapolygon": "sapolygongeo",
        "mupoint": "mupointgeo",
        "muline": "mulinegeo",
        "featpoint": "featpointgeo",
        "featline": "featlinegeo",
    }
    return geometry_columns.get(table.lower())


# ============================================================================
# TIER 2 - SPECIALIZED FUNCTIONS (DEPRECATED - Use fetch_by_keys instead)
# ============================================================================
# These functions wrap fetch_by_keys() for specific tables.
# They are deprecated but maintained for backward compatibility.
# All will be removed in v0.4.0 - migrate to fetch_by_keys().
# ============================================================================

@add_sync_version
async def fetch_mapunit_polygon(
    mukeys: Union[List[Union[str, int]], Union[str, int]],
    columns: Optional[Union[str, List[str]]] = None,
    include_geometry: bool = True,
    chunk_size: int = 1000,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch map unit polygon data for a list of mukeys (DEPRECATED - Use fetch_by_keys).

    .. deprecated:: 0.3.0
        Use :func:`fetch_by_keys` instead

    This function wraps :func:`fetch_by_keys` for the mupolygon table.
    It will be removed in v0.4.0.

    Performance Notes:
    - Geometry data significantly increases response size and processing time
    - For large areas, consider using smaller chunk_size (500-1000)
    - Polygon geometries can be very large; consider bbox filtering first

    Args:
        mukeys: Map unit key(s) (single key or list of keys)
        columns: Columns to select (default: key columns from schema + geometry)
        include_geometry: Whether to include polygon geometry as WKT
        chunk_size: Chunk size for pagination (recommended: 500-1000 for geometry)
        client: Optional SDA client

    Returns:
        SDAResponse with map unit polygon data
    """
    warnings.warn(
        "fetch_mapunit_polygon() is deprecated and will be removed in v0.4.0. "
        "Use fetch_by_keys(mukeys, 'mupolygon', include_geometry=True, client=client) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Handle single mukey values for convenience
    if not isinstance(mukeys, list):
        mukeys = [mukeys]

    if columns is None:
        # Use schema-based default columns for mupolygon table
        schema = get_schema("mupolygon")
        columns = schema.get_default_columns() if schema else []

    return await fetch_by_keys(
        mukeys, "mupolygon", "mukey", columns, chunk_size, include_geometry, client
    )


@add_sync_version
async def fetch_component_by_mukey(
    mukeys: Union[List[Union[str, int]], Union[str, int]],
    columns: Optional[Union[str, List[str]]] = None,
    chunk_size: int = 1000,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch component data for a list of mukeys (DEPRECATED - Use fetch_by_keys).

    .. deprecated:: 0.3.0
        Use :func:`fetch_by_keys` instead

    This function wraps :func:`fetch_by_keys` for the component table.
    It will be removed in v0.4.0.

    Performance Notes:
    - Components are the most numerous SSURGO entities (often 1000s per survey area)
    - Use chunk_size of 500-1000 for large mukey lists
    - Consider filtering for major components only if not needed

    Args:
        mukeys: Map unit key(s) (single key or list of keys)
        columns: Columns to select (default: key component columns from schema)
        chunk_size: Chunk size for pagination (recommended: 500-1000)
        client: Optional SDA client

    Returns:
        SDAResponse with component data
    """
    warnings.warn(
        "fetch_component_by_mukey() is deprecated and will be removed in v0.4.0. "
        "Use fetch_by_keys(mukeys, 'component', key_column='mukey', client=client) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Handle single mukey values for convenience
    if not isinstance(mukeys, list):
        mukeys = [mukeys]

    if columns is None:
        # Use schema-based default columns for component table
        schema = get_schema("component")
        if schema:
            columns = schema.get_default_columns() + ["mukey"]
        else:
            columns = ["mukey"]

    response = await fetch_by_keys(
        mukeys, "component", "mukey", columns, chunk_size, False, client
    )

    return response


@add_sync_version
async def fetch_chorizon_by_cokey(
    cokeys: Union[List[Union[str, int]], Union[str, int]],
    columns: Optional[Union[str, List[str]]] = None,
    chunk_size: int = 1000,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch chorizon data for a list of cokeys (DEPRECATED - Use fetch_by_keys).

    .. deprecated:: 0.3.0
        Use :func:`fetch_by_keys` instead

    This function wraps :func:`fetch_by_keys` for the chorizon table.
    It will be removed in v0.4.0.

    Performance Notes:
    - Horizon data includes detailed soil properties (texture, chemistry, etc.)
    - Each component typically has 3-7 horizons; expect large result sets
    - Use chunk_size of 200-500 for large cokey lists

    Args:
        cokeys: Component key(s) (single key or list of keys)
        columns: Columns to select (default: key chorizon columns from schema)
        chunk_size: Chunk size for pagination (recommended: 200-500)
        client: Optional SDA client

    Returns:
        SDAResponse with chorizon data
    """
    warnings.warn(
        "fetch_chorizon_by_cokey() is deprecated and will be removed in v0.4.0. "
        "Use fetch_by_keys(cokeys, 'chorizon', key_column='cokey', client=client) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Handle single cokey values for convenience
    if not isinstance(cokeys, list):
        cokeys = [cokeys]

    if columns is None:
        # Use schema-based default columns for chorizon table
        schema = get_schema("chorizon")
        columns = schema.get_default_columns() if schema else []

    return await fetch_by_keys(
        cokeys, "chorizon", "cokey", columns, chunk_size, False, client
    )


@add_sync_version
async def fetch_survey_area_polygon(
    areasymbols: Union[List[str], str],
    columns: Optional[Union[str, List[str]]] = None,
    include_geometry: bool = True,
    chunk_size: int = 1000,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch survey area polygon data for a list of area symbols (DEPRECATED - Use fetch_by_keys).

    .. deprecated:: 0.3.0
        Use :func:`fetch_by_keys` instead

    This function wraps :func:`fetch_by_keys` for the sapolygon table.
    It will be removed in v0.4.0.

    Performance Notes:
    - Survey area boundaries are large polygons; geometry data is substantial
    - Most use cases only need a few survey areas at a time
    - Consider tabular queries first, then fetch geometry only when needed

    Args:
        areasymbols: Survey area symbol(s) (single symbol or list of symbols)
        columns: Columns to select (default: key columns + geometry)
        include_geometry: Whether to include polygon geometry as WKT
        chunk_size: Chunk size for pagination (usually not needed for survey areas)
        client: Optional SDA client

    Returns:
        SDAResponse with survey area polygon data
    """
    warnings.warn(
        "fetch_survey_area_polygon() is deprecated and will be removed in v0.4.0. "
        "Use fetch_by_keys(areasymbols, 'sapolygon', key_column='areasymbol', "
        "include_geometry=True, client=client) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Handle single areasymbol values for convenience
    if not isinstance(areasymbols, list):
        areasymbols = [areasymbols]

    if columns is None:
        columns = "areasymbol, spatialversion, lkey"

    return await fetch_by_keys(
        areasymbols,
        "sapolygon",
        "areasymbol",
        columns,
        chunk_size,
        include_geometry,
        client,
    )


@add_sync_version
async def fetch_pedons_by_bbox(
    bbox: Tuple[float, float, float, float],
    columns: Optional[List[str]] = None,
    chunk_size: int = 1000,
    return_type: Literal["sitedata", "combined", "soilprofilecollection"] = "sitedata",
    client: Optional[SDAClient] = None,
) -> Union[SDAResponse, Dict[str, Any], Any]:
    """
    Fetch pedon site data within a geographic bounding box with flexible return types.

    Similar to fetchLDM() in R soilDB, this function retrieves laboratory-analyzed
    soil profiles (pedons) within a specified geographic area. The return type
    can be customized to return site data only, combined site and horizon data,
    or a SoilProfileCollection object.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        columns: List of columns to return for site data. If None, returns basic pedon columns
        chunk_size: Number of pedons to process per query (for pagination when fetching horizons)
        return_type: Type of return value (default: "sitedata")
            - "sitedata": Returns only site data as SDAResponse
            - "combined": Returns dict with keys "site" (SDAResponse) and "horizons" (SDAResponse)
            - "soilprofilecollection": Returns a SoilProfileCollection object with site and horizon data
        client: Optional SDA client instance

    Returns:
        Depending on return_type:
        - "sitedata": SDAResponse containing pedon site data only
        - "combined": Dict with keys "site" (SDAResponse) and "horizons" (SDAResponse)
        - "soilprofilecollection": SoilProfileCollection object

    Raises:
        TypeError: If client parameter is required but not provided
        ImportError: If soilprofilecollection is requested but not installed
        ValueError: If return_type is invalid

    Examples:
        # Fetch pedons in California's Central Valley - site data only (default)
        >>> bbox = (-122.0, 36.0, -118.0, 38.0)
        >>> response = await fetch_pedons_by_bbox(bbox)
        >>> df = response.to_pandas()

        # Fetch site and horizon data separately
        >>> result = await fetch_pedons_by_bbox(bbox, return_type="combined")
        >>> site_df = result["site"].to_pandas()
        >>> horizons_df = result["horizons"].to_pandas()

        # Fetch complete pedon profiles as SoilProfileCollection
        >>> spc = await fetch_pedons_by_bbox(bbox, return_type="soilprofilecollection")
        >>> # spc is now a soilprofilecollection.SoilProfileCollection object

        # Get horizon data for returned pedons (manual approach)
        >>> site_response = await fetch_pedons_by_bbox(bbox)
        >>> pedon_keys = site_response.to_pandas()["pedon_key"].unique().tolist()
        >>> horizons = await fetch_pedon_horizons(pedon_keys, client=client)
    """
    if return_type not in ["sitedata", "combined", "soilprofilecollection"]:
        raise ValueError(
            f"Invalid return_type: {return_type!r}. Must be one of: "
            "'sitedata', 'combined', 'soilprofilecollection'"
        )

    min_lon, min_lat, max_lon, max_lat = bbox

    if client is None:
        raise TypeError("client parameter is required")

    # Fetch site data
    query = QueryBuilder.pedons_intersecting_bbox(
        min_lon, min_lat, max_lon, max_lat, columns
    )
    site_response = await client.execute(query)

    # If only site data is requested or response is empty, return early
    if return_type == "sitedata" or site_response.is_empty():
        return site_response

    # For "combined" or "soilprofilecollection", we need horizon data
    # Get pedon keys for horizon fetching
    site_df = site_response.to_pandas()
    pedon_keys = site_df["pedon_key"].unique().tolist()

    # Fetch horizons in chunks if needed
    all_horizons = []
    sample_cols = None
    sample_meta = None
    if len(pedon_keys) <= chunk_size:
        # Single query for small pedon lists
        horizons_response = await fetch_pedon_horizons(pedon_keys, client=client)
        if not horizons_response.is_empty():
            # Capture columns and metadata from the response
            sample_cols = horizons_response.columns
            sample_meta = horizons_response.metadata
            all_horizons.extend(horizons_response.data)
    else:
        # Multiple queries for large pedon lists
        logger.debug(
            f"Fetching horizons for {len(pedon_keys)} pedons in chunks of {chunk_size}"
        )
        for i in range(0, len(pedon_keys), chunk_size):
            chunk_keys = pedon_keys[i : i + chunk_size]
            chunk_response = await fetch_pedon_horizons(chunk_keys, client=client)
            if not chunk_response.is_empty():
                # Capture columns and metadata from first non-empty chunk
                if sample_cols is None:
                    sample_cols = chunk_response.columns
                    sample_meta = chunk_response.metadata
                all_horizons.extend(chunk_response.data)

    # Build horizons response object from combined data
    if all_horizons:
        # Reconstruct the raw data format that SDAResponse expects
        horizons_table = []
        horizons_table.append(sample_cols)
        horizons_table.append(sample_meta)
        horizons_table.extend(all_horizons)
        horizons_raw_data = {"Table": horizons_table}
        horizons_response = SDAResponse(horizons_raw_data)
    else:
        # Empty horizons response
        horizons_response = SDAResponse({})

    if return_type == "combined":
        return {"site": site_response, "horizons": horizons_response}

    elif return_type == "soilprofilecollection":
        # Convert to SoilProfileCollection
        if horizons_response.is_empty():
            raise ValueError(
                "No horizon data found. Cannot create SoilProfileCollection without horizons."
            )

        return horizons_response.to_soilprofilecollection(
            site_data=site_df,
            site_id_col="pedon_key",
            hz_id_col="layer_key",
            hz_top_col="hzn_top",
            hz_bot_col="hzn_bot",
        )

    # Fallback (shouldn't reach here due to validation above)
    return site_response


@add_sync_version
async def fetch_pedon_horizons(
    pedon_keys: Union[List[str], str],
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch horizon data for specified pedon keys.

    Args:
        pedon_keys: Single pedon key or list of pedon keys
        client: Optional SDA client instance

    Returns:
        SDAResponse containing horizon data
    """
    if isinstance(pedon_keys, str):
        pedon_keys = [pedon_keys]

    if client is None:
        raise TypeError("client parameter is required")

    query = QueryBuilder.pedon_horizons_by_pedon_keys(pedon_keys)
    return await client.execute(query)


# ============================================================================
# TIER 4 - KEY LOOKUP HELPERS (For planning multi-step fetches)
# ============================================================================
# These functions discover database keys for use in subsequent fetches.
# Use before complex multi-step operations to plan key lists.
# Small results: Immediate execution (no chunking).
# ============================================================================

@add_sync_version
async def get_mukey_by_areasymbol(
    areasymbols: List[str], client: Optional[SDAClient] = None
) -> List[int]:
    """
    Get all mukeys for given area symbols (TIER 4 - Helper).

    **WHEN TO USE THIS**:
    - You know the survey area(s) but need to discover all map units
    - Planning multi-step fetch operations
    - Building key lists for fetch_by_keys()

    **DESIGN - Why this helper exists**:
    - Convenience: Discovers all mukeys in survey areas
    - Use before: fetch_by_keys(..., "component", key_column="mukey")
    - Performance: Small result (quick execution)

    Args:
        areasymbols: List of survey area symbols (e.g., ["IA001", "IA002"])
        client: Required SDA client instance

    Returns:
        List of all mukeys found in specified survey areas

    Examples:
        # Discover mukeys in survey areas
        >>> mukeys = await get_mukey_by_areasymbol(["IA001", "IA002"])
        >>> print(f"Found {len(mukeys)} map units")
        
        # Then fetch components for those map units
        >>> components = await fetch_by_keys(mukeys, "component", key_column="mukey")
        >>> df = components.to_pandas()

    See Also:
        get_cokey_by_mukey() - Discover cokeys from mukeys
        fetch_by_keys() - Use discovered keys to fetch data
    """
    if client is None:
        raise TypeError("client parameter is required")

    # Use the existing get_mapunits_by_legend pattern but for multiple areas
    key_strings = sanitize_sql_string_list(areasymbols)
    where_clause = f"l.areasymbol IN ({', '.join(key_strings)})"

    query = (
        Query()
        .select("m.mukey")
        .from_("mapunit m")
        .inner_join("legend l", "m.lkey = l.lkey")
        .where(where_clause)
    )

    response = await client.execute(query)
    df = response.to_pandas()

    return df["mukey"].tolist() if not df.empty else []


@add_sync_version
async def get_cokey_by_mukey(
    mukeys: Union[List[Union[str, int]], Union[str, int]],
    major_components_only: bool = True,
    client: Optional[SDAClient] = None,
) -> List[str]:
    """
    Get all cokeys for given mukeys (TIER 4 - Helper).

    **WHEN TO USE THIS**:
    - You know the map units but need to discover all components
    - Planning multi-step fetch operations to get horizons
    - Building key lists for fetch_by_keys()

    **DESIGN - Why this helper exists**:
    - Convenience: Discovers all cokeys in map units
    - Use before: fetch_by_keys(..., "chorizon", key_column="cokey")
    - Performance: Small result (quick execution)
    - Option: major_components_only to filter

    Args:
        mukeys: Map unit key(s) (single key or list of keys)
        major_components_only: If True, only return major components (default: True)
        client: Required SDA client instance

    Returns:
        List of all component keys found in specified map units

    Examples:
        # Discover cokeys in map units
        >>> cokeys = await get_cokey_by_mukey([123456, 123457])
        >>> print(f"Found {len(cokeys)} components")
        
        # Then fetch horizons for those components
        >>> horizons = await fetch_by_keys(cokeys, "chorizon", key_column="cokey")
        >>> df = horizons.to_pandas()
        
        # Include minor components
        >>> all_cokeys = await get_cokey_by_mukey([123456], major_components_only=False)

    See Also:
        get_mukey_by_areasymbol() - Discover mukeys from survey areas
        fetch_by_keys() - Use discovered keys to fetch data
    """
    # Handle single mukey values for convenience
    if not isinstance(mukeys, list):
        mukeys = [mukeys]

    # At this point mukeys is guaranteed to be a list
    mukeys_list: List[Union[str, int]] = mukeys

    sanitized_keys = [sanitize_sql_numeric(k) for k in mukeys_list]
    where_clause = f"mukey IN ({', '.join(sanitized_keys)})"
    if major_components_only:
        where_clause += " AND majcompflag = 'Yes'"

    response = await fetch_by_keys(
        mukeys_list, "component", "mukey", "cokey", client=client
    )
    df = response.to_pandas()

    return df["cokey"].tolist() if not df.empty else []
