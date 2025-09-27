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
from .client import SDAClient
from .convenience import (
    get_mapunit_by_areasymbol,
    get_mapunit_by_bbox,
    get_mapunit_by_point,
    get_sacatalog,
    get_sacatalog_sync,
    list_survey_areas,  # Backward compatibility
    list_survey_areas_sync,  # Backward compatibility
)
from .exceptions import (
    SDAConnectionError,
    SDAMaintenanceError,
    SDAQueryError,
    SoilDBError,
)
from .fetch import (
    fetch_by_keys,
    fetch_chorizon_by_cokey,
    fetch_component_by_mukey,
    fetch_mapunit_polygon,
    fetch_survey_area_polygon,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
)
from .metadata import (
    MetadataParseError,
    SurveyMetadata,
    extract_metadata_summary,
    parse_survey_metadata,
)
from .query import Query, QueryBuilder, SpatialQuery
from .response import SDAResponse
from .spatial import (
    SpatialQueryBuilder,
    mupolygon_in_bbox,
    query_featline,
    query_featpoint,
    query_mupolygon,
    query_sapolygon,
    sapolygon_in_bbox,
    spatial_query,
)

__all__ = [
    # Core classes
    "SDAClient",
    "Query",
    "SpatialQuery",
    "QueryBuilder",
    "SDAResponse",
    # Exceptions
    "SoilDBError",
    "SDAConnectionError",
    "SDAQueryError",
    "SDAMaintenanceError",
    "MetadataParseError",
    # Metadata parsing
    "SurveyMetadata",
    "parse_survey_metadata",
    "extract_metadata_summary",
    # Convenience functions
    "get_mapunit_by_areasymbol",
    "get_mapunit_by_point",
    "get_mapunit_by_bbox",
    "get_sacatalog",
    "get_sacatalog_sync",
    "list_survey_areas",  # Backward compatibility
    "list_survey_areas_sync",  # Backward compatibility
    # Spatial query functions
    "spatial_query",
    "query_mupolygon",
    "query_sapolygon",
    "query_featpoint",
    "query_featline",
    "mupolygon_in_bbox",
    "sapolygon_in_bbox",
    "SpatialQueryBuilder",
    # Bulk/paginated fetching
    "fetch",
    "fetch_by_keys",
    "fetch_mapunit_polygon",
    "fetch_component_by_mukey",
    "fetch_chorizon_by_cokey",
    "fetch_survey_area_polygon",
    "get_mukey_by_areasymbol",
    "get_cokey_by_mukey",
]
