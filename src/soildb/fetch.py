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
from .query import Query
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


def _combine_responses(
    responses: List[SDAResponse], deduplicate: bool = False
) -> SDAResponse:
    """
    Combine multiple SDAResponse objects into a single unified response.

    This function consolidates paginated query results from concurrent requests
    into a single response object. It handles schema consistency, deduplication,
    and validation to ensure data integrity.

    **HOW RESPONSES ARE COMBINED**:

    Responses are merged by concatenating data rows while preserving column order
    and metadata from the first response. The process assumes all responses share
    the same schema (same columns in same order). The combined response maintains
    the SDA table format with header row, metadata row, and data rows.

    Structure:
    ```
    Combined Response:
    - Row 0: Column names (e.g., ["mukey", "muname", "clay"])
    - Row 1: Column metadata/types (e.g., ["Int", "NVarChar", "Float"])
    - Rows 2+: Data rows from all input responses (combined and optionally deduped)
    ```

    **METADATA HANDLING**:

    - Column definitions taken from first response (assumed consistent)
    - All validation states from input responses are combined:
      - Errors: If any response has errors, combined response includes them
      - Warnings: All warnings from all responses are collected
      - Data quality score: Average of all response quality scores
    - Response timestamps and request IDs are preserved from first response

    **DEDUPLICATION LOGIC**:

    When deduplicate=True, duplicate rows are detected and removed based on the
    primary key column (first column, typically). Behavior:

    - First occurrence of each key value is preserved
    - Subsequent occurrences are marked as duplicates and removed
    - Deduplication occurs BEFORE validation
    - Statistics logged: "Deduped K rows from N total rows"
    - Use case: When fetching overlapping key ranges, some rows appear in multiple chunks

    Note: Deduplication is based on row equality, not just key columns. If the
    same key has different values in other columns, both are kept (data conflict).

    **CONFLICT RESOLUTION**:

    Conflicts occur when the same key appears with different values in other
    columns. Behavior:

    - No automatic conflict resolution (data is kept as-is)
    - Conflict detection during validation (logged as warning)
    - User must decide: merge manually or reject response
    - Consider: How did conflicting data originate? (data quality issue)
    - Typical cause: Concurrent fetches overlapped, or source data inconsistency

    **DECISION TREE - COMBINING RESPONSES**:

    When combining responses, assume:
    1. All responses are from the same SSURGO table (same schema)
    2. All responses have identical column definitions (order and types)
    3. Keys come from sequential chunks (no intentional overlap unless using deduplicate=True)
    4. Metadata (column types) are consistent across all responses
    5. Validation state can be merged (errors accumulated, score averaged)

    Combining responses will FAIL if:
    - Responses have different column counts or names
    - Responses have different metadata/type information
    - Responses are None or empty (internal handling only)

    **VALIDATION AFTER COMBINING**:

    After combining, the response is validated:
    1. Schema consistency check: All columns match first response
    2. Type consistency check: Data types match declared types
    3. Row integrity check: All rows have same number of columns
    4. Deduplication check: Report any duplicates detected
    5. Conflict detection: Report any key-value conflicts

    Validation errors block combining (exception raised).
    Validation warnings are logged but don't block combining.

    **PERFORMANCE NOTES**:

    - Time complexity: O(n) where n = total rows across all responses
    - Space complexity: O(n) for combined data storage
    - Deduplication: O(n) with hash table for seen keys
    - Validation: O(n) for full data check
    - For 1M+ rows: Consider streaming or incremental processing

    Args:
        responses: List of SDAResponse objects to combine.
                   Must contain at least one response.
                   All responses should be from the same query/table.
        deduplicate: If True, remove duplicate rows (default: False).
                     Uses first column as deduplication key.
                     Preserves first occurrence of each key value.

    Returns:
        SDAResponse: Combined response with all data merged.
                     Validation state includes all input responses.

    Raises:
        FetchError: If responses list is empty
        FetchError: If schema mismatch detected (different columns/types)
        FetchError: If row integrity check fails (inconsistent column counts)
        FetchError: If response format is invalid (missing headers/metadata)

    Examples:
        # Basic combination of two responses
        >>> response1 = await fetch_by_keys([1, 2, 3], "mapunit", client=client)
        >>> response2 = await fetch_by_keys([4, 5, 6], "mapunit", client=client)
        >>> combined = _combine_responses([response1, response2])
        >>> print(f"Combined {len([response1, response2])} responses, "
        ...       f"{len(combined.data)} rows total")

        # Combine with deduplication (handles overlapping key ranges)
        >>> overlapping_responses = [...]  # Multiple responses with possible overlaps
        >>> combined = _combine_responses(overlapping_responses, deduplicate=True)
        >>> df = combined.to_pandas()

        # Access combined validation state
        >>> combined = _combine_responses([r1, r2, r3])
        >>> validation_result = combined.validation_result
        >>> if validation_result.has_errors:
        ...     print(f"Validation errors: {validation_result.errors}")

    See Also:
        fetch_by_keys() - Public function that uses this internally
        SDAResponse - Response object format and structure
        _validate_schema_consistency() - Helper for schema validation
        _validate_row_integrity() - Helper for data validation
    """
    import time

    start_time = time.time()

    # Validate inputs
    if not responses:
        raise FetchError("No responses to combine")

    if len(responses) == 1:
        logger.debug("Single response, returning as-is")
        return responses[0]

    logger.debug(
        f"Combining {len(responses)} responses, deduplicate={deduplicate}"
    )

    # Validate schema consistency across all responses
    try:
        _validate_schema_consistency(responses)
    except FetchError as e:
        logger.error(f"Schema validation failed: {e}")
        raise

    # Collect data from all responses with deduplication if requested
    combined_data = []
    seen_keys: Dict[Any, bool] = {}  # Track seen keys for deduplication
    deduped_count = 0

    first_response = responses[0]

    for response_idx, response in enumerate(responses):
        if response.is_empty():
            logger.debug(f"Response {response_idx} is empty, skipping")
            continue

        for row_idx, row in enumerate(response.data):
            # Extract first column value as key for deduplication
            if deduplicate and row:
                row_key = next(iter(row.values())) if isinstance(row, dict) else row[0]

                if row_key in seen_keys:
                    deduped_count += 1
                    logger.debug(
                        f"Deduplicating: {row_key} (seen before in earlier chunk)"
                    )
                    continue  # Skip duplicate

                seen_keys[row_key] = True

            combined_data.append(row)

    # Validate row integrity
    try:
        _validate_row_integrity(combined_data, first_response.columns)
    except FetchError as e:
        logger.warning(f"Row integrity warning (continuing anyway): {e}")

    # Build the combined table in SDA format
    combined_table: List[Any] = []

    # Add the header row (column names)
    combined_table.append(first_response.columns)

    # Add the metadata row (column types)
    combined_table.append(first_response.metadata)

    # Add all the combined data rows
    combined_table.extend(combined_data)

    # Create new raw data structure
    combined_raw_data: Dict[str, Any] = {"Table": combined_table}

    # Create new SDAResponse
    combined_response = SDAResponse(combined_raw_data)

    # Combine validation state from all responses
    try:
        _merge_validation_state(combined_response, responses)
    except Exception as e:
        logger.warning(f"Could not merge validation state: {e}")

    # Log combining statistics
    elapsed_time = time.time() - start_time
    total_input_rows = sum(len(r.data) for r in responses)
    logger.info(
        f"Combined {len(responses)} responses: "
        f"{len(combined_data)} rows total (deduped: {deduped_count}), "
        f"elapsed: {elapsed_time:.3f}s"
    )

    if deduplicate and deduped_count > 0:
        logger.warning(
            f"Deduplication removed {deduped_count} duplicate rows "
            f"({100*deduped_count/total_input_rows:.1f}% reduction). "
            f"Check if chunking strategy is causing overlaps."
        )

    return combined_response


def _validate_schema_consistency(responses: List[SDAResponse]) -> None:
    """
    Validate that all responses have consistent schemas.

    Checks that all responses have the same columns in the same order
    and same metadata/type information.

    Args:
        responses: List of responses to validate

    Raises:
        FetchError: If schema mismatch detected
    """
    if not responses:
        return

    first_columns = responses[0].columns
    first_metadata = responses[0].metadata

    for idx, response in enumerate(responses[1:], start=1):
        if response.columns != first_columns:
            raise FetchError(
                f"Schema mismatch: Response {idx} has different columns. "
                f"Expected: {first_columns}, Got: {response.columns}"
            )

        if response.metadata != first_metadata:
            logger.warning(
                f"Metadata mismatch in response {idx}: "
                f"Expected: {first_metadata}, Got: {response.metadata}. "
                f"Using first response metadata."
            )


def _validate_row_integrity(rows: List[Any], expected_columns: List[str]) -> None:
    """
    Validate that all rows have consistent structure.

    Checks that all rows have the same number of columns as the schema,
    and that columns are in the expected order.

    Args:
        rows: List of data rows
        expected_columns: Expected column list from schema

    Raises:
        FetchError: If row integrity issues detected
    """
    if not rows:
        return

    expected_col_count = len(expected_columns)

    for row_idx, row in enumerate(rows):
        if isinstance(row, dict):
            if len(row) != expected_col_count:
                raise FetchError(
                    f"Row {row_idx} has {len(row)} columns, "
                    f"expected {expected_col_count}. "
                    f"Expected columns: {expected_columns}"
                )
        elif isinstance(row, (list, tuple)):
            if len(row) != expected_col_count:
                raise FetchError(
                    f"Row {row_idx} has {len(row)} columns, "
                    f"expected {expected_col_count}"
                )
        else:
            raise FetchError(f"Row {row_idx} has unexpected type: {type(row)}")


def _merge_validation_state(
    combined_response: SDAResponse, input_responses: List[SDAResponse]
) -> None:
    """
    Merge validation state from all input responses into combined response.

    Combines validation state by:
    1. Collecting all errors and warnings
    2. Averaging data quality scores
    3. Recording merge timestamp

    Args:
        combined_response: The newly combined response object
        input_responses: List of original responses being combined

    Note:
        This function modifies combined_response in-place if validation
        state attributes exist. If attributes don't exist, continues silently.
    """
    try:
        # Check if responses have validation_result attribute
        validation_results = [
            r.validation_result
            for r in input_responses
            if hasattr(r, "validation_result") and r.validation_result is not None
        ]

        if not validation_results:
            logger.debug("No validation state to merge")
            return

        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        quality_scores = []

        for vr in validation_results:
            if hasattr(vr, "errors") and vr.errors:
                all_errors.extend(vr.errors)
            if hasattr(vr, "warnings") and vr.warnings:
                all_warnings.extend(vr.warnings)
            if hasattr(vr, "data_quality_score"):
                quality_scores.append(vr.data_quality_score)

        # Update combined response validation state
        if hasattr(combined_response, "validation_result"):
            vr = combined_response.validation_result
            if vr and hasattr(vr, "errors"):
                vr.errors = all_errors
            if vr and hasattr(vr, "warnings"):
                vr.warnings = all_warnings

            # Average quality score
            if quality_scores and hasattr(vr, "data_quality_score"):
                avg_score = sum(quality_scores) / len(quality_scores)
                vr.data_quality_score = avg_score

            logger.debug(
                f"Merged validation state: {len(all_errors)} errors, "
                f"{len(all_warnings)} warnings, "
                f"avg quality score: {avg_score:.2f}" if quality_scores else ""
            )

    except Exception as e:
        logger.debug(f"Could not merge validation state (non-critical): {e}")


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
    from . import query_templates
    query = query_templates.query_pedons_intersecting_bbox(
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

    from . import query_templates
    query = query_templates.query_pedon_horizons_by_pedon_keys(pedon_keys)
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
