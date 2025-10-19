"""
Air and Water Database (AWDB) module for SCAN/SNOTEL data access.

This module provides access to real-time monitoring data from SCAN (Soil Climate Analysis Network)
and SNOTEL (SNOwpack TELemetry) stations operated by the USDA Natural Resources Conservation Service.

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
    get_monitoring_station_data,
    get_nearby_stations,
    list_available_variables,
)
from .exceptions import AWDBError
from .models import StationInfo, StationTimeSeries, TimeSeriesDataPoint

__all__ = [
    "AWDBClient",
    "get_monitoring_station_data",
    "get_nearby_stations",
    "list_available_variables",
    "AWDBError",
    "StationInfo",
    "TimeSeriesDataPoint",
    "StationTimeSeries",
]
