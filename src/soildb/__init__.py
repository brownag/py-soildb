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
    # New names (recommended)
    discover_stations,
    discover_stations_nearby,
    # Deprecated names (for backward compatibility)
    find_stations_by_criteria,
    get_monitoring_station_data,
    get_nearby_stations,
    get_property_data_near,
    get_soil_moisture_by_depth,
    get_soil_moisture_data,
    get_station_sensor_heights,
    get_station_sensor_metadata,
    list_available_variables,
    station_available_properties,
    station_sensor_depths,
    station_sensors,
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
    QueryPresets,
    fetch_by_keys,
    fetch_pedon_horizons,
    fetch_pedons_by_bbox,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
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
from .query import Query
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
from .spatial import (
    SpatialQueryBuilder,
    bbox_query,
    mupolygon_in_bbox,
    point_query,
    sapolygon_in_bbox,
    spatial_query,
)
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
from .type_conversion import (
    TypeMap,
    TypeProcessor,
    convert_value,
    get_default_type_map,
)

# Sync wrappers use @add_sync_version decorator (see utils.py)
# Access via function_name.sync() instead of function_name_sync()

__all__ = [
    # Core classes and base classes
    "BaseDataAccessClient",
    "ClientConfig",
    "SDAClient",
    "Query",
    "SDAResponse",
    # Type conversion system (UNIFIED - consolidates scattered logic)
    "TypeMap",
    "TypeProcessor",
    "get_default_type_map",
    "convert_value",
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
    # Query template functions
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
    # New names (recommended)
    "discover_stations",
    "discover_stations_nearby",
    "get_property_data_near",
    "get_soil_moisture_by_depth",
    "station_available_properties",
    "station_sensor_depths",
    "station_sensors",
    # Deprecated names (for backward compatibility)
    "find_stations_by_criteria",
    "get_monitoring_station_data",
    "get_nearby_stations",
    "get_soil_moisture_data",
    "get_station_sensor_heights",
    "get_station_sensor_metadata",
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
    # Sync versions available via .sync() decorator on all async functions
    # Example: get_mapunit_by_areasymbol.sync("IA109")
    # High-level functions
    "fetch_mapunit_struct_by_point",
    "fetch_pedon_struct_by_bbox",
    "fetch_pedon_struct_by_id",
    # Spatial query functions
    "spatial_query",
    "point_query",
    "bbox_query",
    "mupolygon_in_bbox",
    "sapolygon_in_bbox",
    "SpatialQueryBuilder",
    # Bulk/paginated fetching - FETCH FUNCTION HIERARCHY
    "fetch_by_keys",  # Universal key-based fetcher - RECOMMENDED
    "fetch_pedons_by_bbox",  # Lab pedons with flexible return types
    "fetch_pedon_horizons",  # Horizon data for pedon sites
    "get_mukey_by_areasymbol",  # Discover mukeys from survey areas
    "get_cokey_by_mukey",  # Discover cokeys from map units
    "QueryPresets",  # Preset configurations for common queries
    # Module
    "fetch",  # fetch module
]
