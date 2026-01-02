"""
Air and Water Database (AWDB) module for SCAN/SNOTEL data access.

This module provides access to real-time monitoring data from SCAN (Soil Climate
Analysis Network) and SNOTEL (SNOwpack TELemetry) stations operated by the USDA
Natural Resources Conservation Service.

SCAN/SNOTEL stations provide real-time monitoring of:
- Soil moisture and temperature at multiple depths
- Precipitation (rainfall and snowfall)
- Air temperature and humidity
- Snow water equivalent
- Wind speed and direction

API Documentation:
- SCAN: https://www.wcc.nrcs.usda.gov/scan/
- SNOTEL: https://www.wcc.nrcs.usda.gov/snotel/
- Data API: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/
"""

from .client import AWDBClient
from .convenience import (
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
from .exceptions import AWDBConnectionError, AWDBError, AWDBQueryError
from .models import (
    ForecastData,
    ReferenceData,
    StationInfo,
    StationTimeSeries,
    TimeSeriesDataPoint,
)

__all__ = [
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
    # Exceptions
    "AWDBError",
    "AWDBConnectionError",
    "AWDBQueryError",
    # Models
    "ForecastData",
    "ReferenceData",
    "StationInfo",
    "TimeSeriesDataPoint",
    "StationTimeSeries",
]
