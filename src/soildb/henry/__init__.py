"""
Henry Mount Soil Climate Database module for soildb.

This module provides access to the Henry Mount Soil Climate Database
hosted by UC Davis California Soil Resource Lab, which contains
NRCS-collected portable sensor data organized by projects and
SSO (Soil Survey Office) codes.

Unlike SCAN/SNOTEL which represents permanent infrastructure, Henry
includes temporary and portable sensor deployments for research and
monitoring projects.

Key Features:
- Access to NRCS portable sensor networks
- Project-based organization of stations
- Support for multiple data types: soil temp, soil moisture, air temp, water level
- Gzipped JSON API responses
- Integration with standardized AWDB element codes

Data Characteristics:
- Total sensors: ~1,600+ across various projects and SSO codes
- Accessible subset: ~216 sensors in "2-SON" SSO (California)
- Temporal coverage: Varies by project (2020s onward typically)
- Granularities: Daily and hourly data available
- Depths: Multiple soil depths for subsurface measurements

Basic Usage:
    # High-level convenience functions
    from soildb.henry import find_henry_stations, fetch_henry_data

    # Find stations in California SSO
    stations = await find_henry_stations(sso_code='2-SON')

    # Fetch data for a station
    data = await fetch_henry_data(
        station_id='CA_SITE_001',
        variable_name='soiltemp',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

    # Or use sync versions
    stations = find_henry_stations.sync(sso_code='2-SON')
    data = fetch_henry_data.sync(
        station_id='CA_SITE_001',
        variable_name='soiltemp',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

Advanced Usage:
    # Direct client instantiation for batch operations
    from soildb.henry import HenryClient

    async with HenryClient() as client:
        # Get stations
        stations = await client.get_stations(sso_code='2-SON')

        # For each station, fetch data
        for station in stations:
            data = await client.get_station_data(
                station_id=station.station_id,
                variable_name='soilVWC',
                start_date='2024-01-01',
                end_date='2024-12-31'
            )

API Documentation:
    Endpoint: http://soilmap2-1.lawr.ucdavis.edu/henry/query.php

    Query Parameters:
    - what: Data type (sensors, soiltemp, soilVWC, airtemp, waterlevel, all)
    - usersiteid: Specific station ID (e.g., 'CA_SITE_001')
    - project: Project code (comma-separated for multiple)
    - sso: SSO office code (e.g., '2-SON', 'CA', 'TX')
    - gran: Granularity (hour, day, week, month, year)
    - start: Start date (YYYY-MM-DD format)
    - stop: End date (YYYY-MM-DD format)

    Response Format:
    - Gzipped JSON with multiple arrays: sensors, soiltemp, soilVWC, airtemp, waterlevel
    - Timestamps: YYYY-MM-DD HH:MM:SS (space-separated)
    - Coordinates: WGS84 (EPSG:4326), fields: wgs84_latitude, wgs84_longitude
"""

from .client import HenryClient
from .convenience import (
    fetch_henry_data,
    find_henry_stations,
    get_henry_variables,
    list_henry_projects,
)
from .exceptions import HenryAPIError, HenryDataError, HenryError, HenryNetworkError
from .models import (
    HenryDataCoverage,
    HenrySensor,
    HenryStation,
    HenryStationStatus,
    HenryTimeSeriesDataPoint,
)
from .utils import (
    cm_to_inches,
    construct_element_code,
    henry_variable_to_base_code,
    inches_to_cm,
    parse_element_code,
    parse_henry_timestamp,
)

__all__ = [
    # Client
    "HenryClient",
    # Convenience functions
    "find_henry_stations",
    "fetch_henry_data",
    "get_henry_variables",
    "list_henry_projects",
    # Exceptions
    "HenryError",
    "HenryAPIError",
    "HenryNetworkError",
    "HenryDataError",
    # Data models
    "HenryStation",
    "HenrySensor",
    "HenryTimeSeriesDataPoint",
    "HenryDataCoverage",
    "HenryStationStatus",
    # Utilities
    "cm_to_inches",
    "inches_to_cm",
    "construct_element_code",
    "parse_element_code",
    "henry_variable_to_base_code",
    "parse_henry_timestamp",
]
