"""
Python client for the USDA Soil Data Access web service.

Query soil survey data and export to DataFrames.
"""

try:
    from importlib import metadata

    __version__ = metadata.version(__name__)
except Exception:
    __version__ = "unknown"

from . import fetch
from .awdb import (
    AWDBClient,
    AWDBConnectionError,
    AWDBError,
    AWDBQueryError,
    ForecastData,
    ReferenceData,
    StationInfo,
    StationTimeSeries,
    TimeSeriesDataPoint,
    find_stations_by_criteria,
    get_monitoring_station_data,
    get_nearby_stations,
    get_soil_moisture_data,
    list_available_variables,
)
from .base_client import BaseDataAccessClient, ClientConfig
from .client import SDAClient
from .convenience import (
    get_lab_pedon_by_id,
    get_lab_pedons_by_bbox,
    get_mapunit_by_areasymbol,
    get_mapunit_by_bbox,
    get_mapunit_by_point,
    get_sacatalog,
)
from .exceptions import (
    SDAConnectionError,
    SDAMaintenanceError,
    SDANetworkError,
    SDAQueryError,
    SDAResponseError,
    SDATimeoutError,
    SoilDBError,
)
from .fetch import (
    fetch_by_keys,
    fetch_chorizon_by_cokey,
    fetch_component_by_mukey,
    fetch_mapunit_polygon,
    fetch_pedon_horizons,
    fetch_pedons_by_bbox,
    fetch_survey_area_polygon,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
    QueryPresets,
)
from .high_level import (
    fetch_mapunit_struct_by_point,
    fetch_pedon_struct_by_bbox,
    fetch_pedon_struct_by_id,
)
from .metadata import (
    MetadataParseError,
    SurveyMetadata,
    extract_metadata_summary,
    filter_metadata_by_bbox,
    get_metadata_statistics,
    parse_survey_metadata,
    search_metadata_by_keywords,
)
from .query import Query, QueryBuilder, SpatialQuery
from .query_templates import (
    query_available_survey_areas,
    query_component_horizons_by_legend,
    query_components_at_point,
    query_components_by_legend,
    query_from_sql,
    query_mapunits_by_legend,
    query_mapunits_intersecting_bbox,
    query_pedon_by_pedon_key,
    query_pedon_horizons_by_pedon_keys,
    query_pedons_intersecting_bbox,
    query_spatial_by_legend,
    query_survey_area_boundaries,
)
from .response import SDAResponse
from .spc_presets import (
    ColumnConfig,
    CustomColumnConfig,
    LabPedonHorizonColumns,
    MapunitComponentHorizonColumns,
    PedonSiteHorizonColumns,
    StandardSDAHorizonColumns,
    get_preset,
    list_presets,
)
from .spc_validator import (
    SPCColumnValidator,
    SPCDepthValidator,
    SPCValidationError,
    SPCWarnings,
    create_spc_validation_report,
)
from .spatial import (
    SpatialQueryBuilder,
    bbox_query,
    mupolygon_in_bbox,
    point_query,
    query_featline,
    query_featpoint,
    query_mupolygon,
    query_sapolygon,
    sapolygon_in_bbox,
    spatial_query,
)
from .sync import (
    AsyncSyncBridge,
    fetch_component_by_comppct_r_sync,
    fetch_component_by_mukey_sync,
    fetch_horizon_by_cokey_sync,
    fetch_mapunit_by_areasymbol_sync,
    fetch_pedons_by_bbox_sync,
    get_mapunit_by_areasymbol_sync,
    get_mapunit_by_bbox_sync,
    get_mapunit_by_point_sync,
    get_sacatalog_sync,
    list_survey_areas_sync,
)

__all__ = [
    # Core classes and base classes
    "BaseDataAccessClient",
    "ClientConfig",
    "SDAClient",
    "Query",
    "SpatialQuery",
    "QueryBuilder",
    "SDAResponse",
    # SoilProfileCollection integration (NEW)
    "ColumnConfig",
    "StandardSDAHorizonColumns",
    "LabPedonHorizonColumns",
    "PedonSiteHorizonColumns",
    "MapunitComponentHorizonColumns",
    "CustomColumnConfig",
    "get_preset",
    "list_presets",
    "SPCValidationError",
    "SPCColumnValidator",
    "SPCDepthValidator",
    "SPCWarnings",
    "create_spc_validation_report",
    # Query template functions (replaces QueryBuilder)
    "query_mapunits_by_legend",
    "query_components_by_legend",
    "query_component_horizons_by_legend",
    "query_components_at_point",
    "query_mapunits_intersecting_bbox",
    "query_spatial_by_legend",
    "query_available_survey_areas",
    "query_survey_area_boundaries",
    "query_from_sql",
    "query_pedons_intersecting_bbox",
    "query_pedon_horizons_by_pedon_keys",
    "query_pedon_by_pedon_key",
    # AWDB (SCAN/SNOTEL) classes and functions
    "AWDBClient",
    "find_stations_by_criteria",
    "get_monitoring_station_data",
    "get_nearby_stations",
    "get_soil_moisture_data",
    "list_available_variables",
    "AWDBError",
    "AWDBConnectionError",
    "AWDBQueryError",
    "ForecastData",
    "ReferenceData",
    "StationInfo",
    "TimeSeriesDataPoint",
    "StationTimeSeries",
    # Exceptions
    "SoilDBError",
    "SDANetworkError",
    "SDAConnectionError",
    "SDATimeoutError",
    "SDAMaintenanceError",
    "SDAQueryError",
    "SDAResponseError",
    "MetadataParseError",
    # Metadata parsing
    "SurveyMetadata",
    "parse_survey_metadata",
    "extract_metadata_summary",
    "search_metadata_by_keywords",
    "filter_metadata_by_bbox",
    "get_metadata_statistics",
    # Async convenience functions
    "get_mapunit_by_areasymbol",
    "get_mapunit_by_point",
    "get_mapunit_by_bbox",
    "get_lab_pedons_by_bbox",
    "get_lab_pedon_by_id",
    "get_sacatalog",
    # Sync convenience functions (primary API for blocking calls)
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
    # High-level functions
    "fetch_mapunit_struct_by_point",
    "fetch_pedon_struct_by_bbox",
    "fetch_pedon_struct_by_id",
    # Spatial query functions
    "spatial_query",
    "point_query",
    "bbox_query",
    "query_mupolygon",
    "query_sapolygon",
    "query_featpoint",
    "query_featline",
    "mupolygon_in_bbox",
    "sapolygon_in_bbox",
    "SpatialQueryBuilder",
    # Bulk/paginated fetching - FETCH FUNCTION HIERARCHY
    # TIER 1 - PRIMARY INTERFACE (Use for most cases)
    "fetch_by_keys",  # Universal key-based fetcher - RECOMMENDED
    "QueryPresets",  # Preset configurations for common queries - NEW
    # TIER 2 - SPECIALIZED FUNCTIONS (deprecated, use fetch_by_keys)
    "fetch_mapunit_polygon",  # DEPRECATED - use fetch_by_keys(..., "mupolygon")
    "fetch_component_by_mukey",  # DEPRECATED - use fetch_by_keys(..., "component", "mukey")
    "fetch_chorizon_by_cokey",  # DEPRECATED - use fetch_by_keys(..., "chorizon", "cokey")
    "fetch_survey_area_polygon",  # DEPRECATED - use fetch_by_keys(..., "sapolygon", "areasymbol")
    # TIER 3 - COMPLEX MULTI-STEP FETCHES
    "fetch_pedons_by_bbox",  # Lab pedons with flexible return types
    "fetch_pedon_horizons",  # Horizon data for pedon sites
    # TIER 4 - KEY LOOKUP HELPERS (for planning multi-step operations)
    "get_mukey_by_areasymbol",  # Discover mukeys from survey areas
    "get_cokey_by_mukey",  # Discover cokeys from map units
    # Module
    "fetch",  # fetch module
]
