"""
Bulk data fetching with automatic pagination and abstraction levels.

This module provides a hierarchical API for fetching large SSURGO datasets:

TIER 1 - PRIMARY INTERFACE (Use for most cases):
  fetch_by_keys() - Universal key-based fetcher with pagination support
    - Flexible: works with any SSURGO table and key column
    - Recommended: Use this unless you need specialized behavior
    - Performance: Automatic chunking, concurrent requests

TIER 2 - COMPLEX MULTI-STEP FETCHES:
  fetch_pedons_by_bbox() - Lab pedons with optional site+horizon data
  fetch_pedon_horizons() - Horizon data for pedon sites
  fetch_ldm() - Laboratory Data Mart queries (SDA or SQLite)
    - Complex: Multi-table joins, optional geometry, custom return types
    - Keep: Significant value over raw queries

TIER 3 - KEY LOOKUP HELPERS (For planning complex fetches):
  get_mukey_by_areasymbol() - Discover all mukeys in survey areas
  get_cokey_by_mukey() - Discover all cokeys in map units
    - Use before multi-step fetches to plan key lists
    - Small results: Immediate execution (no chunking)

ARCHITECTURE DIAGRAM:

    User Query
        ↓
    ┌─────────────────────────────────────┐
    │ fetch_by_keys()                     │ ← PRIMARY (use this)
    │ (handles all SSURGO tables)         │ ↓ uses fetch_chunked internally
    └─────────────────────────────────────┘
        ↑
        ├── fetch_mapunit_polygon()     │ ← TIER 2 (deprecated, wrap
        ├── fetch_component_by_mukey()  │   fetch_by_keys)
        ├── fetch_chorizon_by_cokey()   │
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
   >>> result = await fetch_pedons_by_bbox(bbox, return_type="combined")
   >>> site_df = result["site"].to_pandas()
   >>> horizons_df = result["horizons"].to_pandas()
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

from .chunked import fetch_chunked
from .client import SDAClient
from .exceptions import SoilDBError
from .ldm.client import LDMClient
from .query import Query
from .query_templates import (
    query_pedon_horizons_by_pedon_keys,
    query_pedons_intersecting_bbox,
)
from .response import SDAResponse
from .sanitization import sanitize_sql_numeric, sanitize_sql_string_list
from .ssurgo_tables import KEY_COLUMNS, geometry_column
from .ssurgo_tables import key_column as get_key_column
from .utils import add_sync_version, require_client

logger = logging.getLogger(__name__)


class FetchError(SoilDBError):
    """Raised when key-based fetching fails."""

    def __str__(self) -> str:
        """Return helpful fetch error message."""
        if "Unknown table" in self.message:
            return f"{self.message} Supported tables include: {', '.join(KEY_COLUMNS.keys())}"
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
            columns: Optional[list[str]] = None,
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

        def as_kwargs(self) -> dict[str, Any]:
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
    def list_presets(cls) -> dict[str, str]:
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
    columns: Optional[Union[str, list[str]]] = None,
    chunk_size: int = 1000,
    max_concurrency: int = 4,
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
    - max_concurrency: Maximum concurrent queries (default 4, 1-16 typical)
    - include_geometry: Add WKT geometry for spatial tables
    - client: Optional SDAClient instance (creates one if None)

    **TABLE KEY MAPPING** (auto-detected):
    - mapunit → mukey
    - component → cokey
    - chorizon → chkey
    - mupolygon → mukey
    - sapolygon → areasymbol
    - featpoint → featkey
    - And many others (see ssurgo_tables.KEY_COLUMNS)

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
        max_concurrency: Maximum concurrent queries to execute (default: 4)
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
        >>> gdf = response.to_geopandas()  # Convert to GeoDataFrame
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
    client = require_client(client)
    if isinstance(keys, (str, int)):
        keys = cast(list[Union[str, int]], [keys])

    keys_list = cast(list[Union[str, int]], keys)

    if not keys_list:
        raise FetchError("The 'keys' parameter cannot be an empty list.")

    # Auto-detect key column if not provided
    if key_column is None:
        key_column = get_key_column(table)
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
        geom_column = geometry_column(table)
        if geom_column:
            if select_columns == "*":
                select_columns = f"*, {geom_column}.STAsText() as geometry"
            else:
                select_columns = (
                    f"{select_columns}, {geom_column}.STAsText() as geometry"
                )

    def build_query(chunk_keys: Sequence[Union[str, int]]) -> Query:
        """Build a Query for a chunk of keys."""
        return (
            Query()
            .select(*[col.strip() for col in select_columns.split(",")])
            .from_(table)
            .where_in(key_column, chunk_keys)
        )

    logger.debug(
        f"Fetching {len(keys_list)} keys with chunk_size={chunk_size}, "
        f"max_concurrency={max_concurrency}"
    )

    return await fetch_chunked(
        keys_list,
        build_query,
        client.execute,
        chunk_size=chunk_size,
        max_concurrency=max_concurrency,
    )


@add_sync_version
async def fetch_pedons_by_bbox(
    bbox: tuple[float, float, float, float],
    columns: Optional[list[str]] = None,
    chunk_size: int = 1000,
    return_type: Literal["sitedata", "combined"] = "sitedata",
    client: Optional[SDAClient] = None,
) -> Union[SDAResponse, dict[str, Any]]:
    """
    Fetch pedon site data within a geographic bounding box with flexible return types.

    Similar to fetchLDM() in R soilDB, this function retrieves laboratory-analyzed
    soil profiles (pedons) within a specified geographic area. The return type
    can be customized to return site data only or combined site and horizon data.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        columns: List of columns to return for site data. If None, returns basic pedon columns
        chunk_size: Number of pedons to process per query (for pagination when fetching horizons)
        return_type: Type of return value (default: "sitedata")
            - "sitedata": Returns only site data as SDAResponse
            - "combined": Returns dict with keys "site" (SDAResponse) and "horizons" (SDAResponse)
        client: Optional SDA client instance

    Returns:
        Depending on return_type:
        - "sitedata": SDAResponse containing pedon site data only
        - "combined": Dict with keys "site" (SDAResponse) and "horizons" (SDAResponse)

    Raises:
        TypeError: If client parameter is required but not provided
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

        # Get horizon data for returned pedons (manual approach)
        >>> site_response = await fetch_pedons_by_bbox(bbox)
        >>> pedon_keys = site_response.to_pandas()["pedon_key"].unique().tolist()
        >>> horizons = await fetch_pedon_horizons(pedon_keys, client=client)
    """
    client = require_client(client)
    if return_type not in ["sitedata", "combined"]:
        raise ValueError(
            f"Invalid return_type: {return_type!r}. Must be one of: "
            "'sitedata', 'combined'"
        )

    min_lon, min_lat, max_lon, max_lat = bbox

    # Fetch site data
    query = query_pedons_intersecting_bbox(min_lon, min_lat, max_lon, max_lat, columns)
    site_response = await client.execute(query)

    # If only site data is requested or response is empty, return early
    if return_type == "sitedata" or site_response.is_empty():
        return site_response

    # For "combined" or "soilprofilecollection", we need horizon data
    # Get pedon keys for horizon fetching
    site_df = site_response.to_pandas()
    pedon_keys = site_df["pedon_key"].unique().tolist()

    # Fetch horizons in chunks with bounded concurrency
    def build_horizon_query(chunk_keys: Sequence[str]) -> Query:
        """Build a horizon query for a chunk of pedon keys."""
        return query_pedon_horizons_by_pedon_keys(list(chunk_keys))

    logger.debug(
        f"Fetching horizons for {len(pedon_keys)} pedons in chunks of {chunk_size}"
    )

    horizons_response = await fetch_chunked(
        pedon_keys,
        build_horizon_query,
        client.execute,
        chunk_size=chunk_size,
    )

    # return_type == "combined"
    return {"site": site_response, "horizons": horizons_response}


@add_sync_version
async def fetch_pedon_horizons(
    pedon_keys: Union[list[str], str],
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
    client = require_client(client)
    if isinstance(pedon_keys, str):
        pedon_keys = [pedon_keys]

    query = query_pedon_horizons_by_pedon_keys(pedon_keys)
    return await client.execute(query)


@add_sync_version
async def fetch_ldm(
    x: Optional[Union[list[Union[str, int]], str, int]] = None,
    what: str = "pedlabsampnum",
    bycol: str = "pedon_key",
    tables: Optional[list[str]] = None,
    WHERE: Optional[str] = None,
    chunk_size: int = 1000,
    max_retries: int = 3,
    layer_type: Union[str, Sequence[str], None] = (
        "horizon",
        "layer",
        "reporting layer",
    ),
    area_type: Optional[str] = "ssa",
    prep_code: Union[str, Sequence[str], None] = ("S", ""),
    analyzed_size_frac: Union[str, Sequence[str], None] = ("<2 mm", ""),
    dsn: Optional[Union[str, Path]] = None,
    client: Optional[Union[LDMClient, SDAClient]] = None,
) -> SDAResponse:
    """
    Query Kellogg Soil Survey Laboratory Data Mart via SDA or SQLite snapshot.

    This function provides access to laboratory soil characterization data from
    the NCSS Kellogg Soil Survey Laboratory (KSSL). Data can be queried from the
    Soil Data Access (SDA) web service or from a local SQLite snapshot.

    This is a high-level convenience function that wraps LDMClient. For advanced
    use cases or reusing connections, use LDMClient directly.

    Args:
        x: Values to search for in column specified by 'what'. Can be single value
           or list. If both 'x' and 'WHERE' are None, returns empty result.
        what: Column name for filtering. Common values:
            - 'pedlabsampnum': Lab pedon number
            - 'upedonid': Pedon ID (assigned by describer, not unique)
            - 'corr_name': Correlated Taxon Name
            - 'samp_name': Sampled As Taxon Name
            - 'pedon_key': Pedon internal key
        bycol: Column name for chunking operations (default: 'pedon_key').
               Used when 'x' contains more records than chunk_size.
        tables: List of LDM tables to retrieve. If None, returns default tables:
                - lab_physical_properties
                - lab_chemical_properties
                - lab_calculations_including_estimates_and_default_values
                - lab_rosetta_Key
                Optional tables available:
                - lab_major_and_trace_elements_and_oxides
                - lab_mineralogy_glass_count_and_optical_properties
                - lab_mir
                - lab_xrd_and_thermal
        WHERE: Custom SQL WHERE clause. Cannot be combined with 'x' parameter.
               Example: "corr_name LIKE 'Miami%' AND area_code = 'US'"
        chunk_size: Number of records per query chunk (default: 1000).
                    When 'x' exceeds this size, queries are split into chunks
                    and executed concurrently for better performance.
        max_retries: Maximum retry attempts with halved chunk_size (default: 3).
                     If a chunked query fails, the chunk size is halved and
                     the query is retried up to max_retries times.
        layer_type: Filter by horizon type. Options:
                - ('horizon', 'layer', 'reporting layer') (default)
                    - 'horizon': Standard horizon
                    - 'layer': Custom layer
                    - 'reporting layer': Reporting layer
        area_type: Filter by geographic classification. Options:
               - 'ssa': Soil Survey Area (default)
                   - 'ssa': Soil Survey Area
                   - 'state': State
                   - 'county': County
                   - 'mlra': Major Land Resource Area
                   - 'nforest': National Forest
                   - 'npark': National Park
        prep_code: Sample preparation code(s) (default: ('S', '')).
                   Options:
               - 'S', 'F', 'HM', 'HM_SK', 'GP', 'M', 'N', ''
        analyzed_size_frac: Analyzed soil particle size (default: ('<2 mm', '')).
                            Options:
                    - '<2 mm', '<0.002 mm', '0.02-0.05 mm', '0.05-0.1 mm'
                    - '0.1-0.25 mm', '0.25-0.5 mm', '0.5-1 mm', '1-2 mm'
                    - '0.02-2 mm', '0.05-2 mm', ''
        dsn: Path to SQLite database. If None, queries Soil Data Access web service.
             If provided alongside an SDAClient, 'dsn' takes precedence for the
             SQLite backend. Download SQLite snapshots from:
             https://ncsslabdatamart.sc.egov.usda.gov/database_download.aspx
        client: Optional LDMClient or SDAClient instance.
                - If None, an ephemeral LDMClient is created as an async context
                  manager (using dsn or SDA web service) and closed on completion.
                - If an LDMClient is provided, it is used directly without closing,
                  and 'dsn' is ignored (the client's configured backend is used).
                - If an SDAClient is provided, its underlying HTTP connection is
                  reused in an ephemeral LDMClient without closing the caller's
                  client. If 'dsn' is also provided alongside an SDAClient, 'dsn'
                  takes precedence and the local SQLite backend is used instead.

    Returns:
        SDAResponse: Query results with laboratory data

    Raises:
        LDMParameterError: If invalid parameters provided
        LDMQueryError: If query execution fails
        LDMBackendSelectionError: If data source unavailable
        FileNotFoundError: If dsn path doesn't exist

    Examples:
        Query by laboratory pedon ID via Soil Data Access::

            >>> response = await fetch_ldm(
            ...     x=['85P0234', '40A3306'],
            ...     what='pedlabsampnum'
            ... )
            >>> df = response.to_pandas()

        Query from local SQLite snapshot::

            >>> response = await fetch_ldm(
            ...     x=['85P0234'],
            ...     what='pedlabsampnum',
            ...     dsn='path/to/LDM_FY2025.sqlite'
            ... )
            >>> df = response.to_pandas()

        Query by correlated taxon name via custom WHERE clause::

            >>> response = await fetch_ldm(
            ...     WHERE="corr_name LIKE 'Miami%' AND area_code = 'US'",
            ...     tables=['lab_physical_properties', 'lab_chemical_properties']
            ... )
            >>> df = response.to_pandas()

        Query with specific filtering for sieved samples::

            >>> response = await fetch_ldm(
            ...     x=['85P0234'],
            ...     what='pedlabsampnum',
            ...     prep_code='S',
            ...     analyzed_size_frac='<2 mm',
            ...     layer_type='horizon'
            ... )

        Synchronous API (automatic .sync() method)::

            >>> response = fetch_ldm.sync(x=['85P0234'], what='pedlabsampnum')
            >>> df = response.to_pandas()

        Convert to other formats::

            >>> response = await fetch_ldm(x=['85P0234'], what='pedlabsampnum')
            >>> df = response.to_pandas()  # pandas DataFrame
            >>> gdf = response.to_geopandas()  # geopandas GeoDataFrame

    See Also:
        - LDMClient: Lower-level client for advanced use cases
        - fetch_by_keys: Universal key-based fetcher for any SSURGO table
        - fetch_pedons_by_bbox: Fetch pedons in geographic area
        - R fetchLDM docs: https://ncss-tech.github.io/soilDB/reference/fetchLDM.html
        - Lab Data Mart: https://ncsslabdatamart.sc.egov.usda.gov
    """

    if isinstance(client, LDMClient):
        return await client.query(
            x=x,
            what=what,
            bycol=bycol,
            tables=tables,
            WHERE=WHERE,
            chunk_size=chunk_size,
            max_retries=max_retries,
            layer_type=layer_type,
            area_type=area_type,
            prep_code=prep_code,
            analyzed_size_frac=analyzed_size_frac,
        )

    if dsn is not None and client is not None:
        logger.warning(
            "Both 'dsn' and 'client' (SDAClient) were provided to fetch_ldm; "
            "'dsn' takes precedence and the local SQLite backend will be used."
        )

    if client is None:
        ctx = LDMClient(dsn=dsn)
    else:
        ctx = LDMClient(dsn=dsn, sda_client=client)

    async with ctx:
        return await ctx.query(
            x=x,
            what=what,
            bycol=bycol,
            tables=tables,
            WHERE=WHERE,
            chunk_size=chunk_size,
            max_retries=max_retries,
            layer_type=layer_type,
            area_type=area_type,
            prep_code=prep_code,
            analyzed_size_frac=analyzed_size_frac,
        )


# ============================================================================
# TIER 4 - KEY LOOKUP HELPERS (For planning multi-step fetches)
# ============================================================================
# These functions discover database keys for use in subsequent fetches.
# Use before complex multi-step operations to plan key lists.
# Small results: Immediate execution (no chunking).
# ============================================================================


@add_sync_version
async def get_mukey_by_areasymbol(
    areasymbols: list[str], client: Optional[SDAClient] = None
) -> list[int]:
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
    client = require_client(client)
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
    mukeys: Union[list[Union[str, int]], Union[str, int]],
    major_components_only: bool = True,
    client: Optional[SDAClient] = None,
) -> list[str]:
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
    mukeys_list: list[Union[str, int]] = mukeys

    sanitized_keys = [sanitize_sql_numeric(k) for k in mukeys_list]
    where_clause = f"mukey IN ({', '.join(sanitized_keys)})"
    if major_components_only:
        where_clause += " AND majcompflag = 'Yes'"

    response = await fetch_by_keys(
        mukeys_list, "component", "mukey", "cokey", client=client
    )
    df = response.to_pandas()

    return df["cokey"].tolist() if not df.empty else []
