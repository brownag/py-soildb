"""
High-level convenience functions for Henry Mount Soil Climate Database access.

These functions provide simple async/sync interfaces for common Henry database
queries without needing to instantiate the client directly.
"""

from typing import Any, Dict, List, Optional

from ..utils import add_sync_version
from .client import HenryClient
from .exceptions import HenryError


@add_sync_version
async def find_henry_stations(
    project_code: Optional[str] = None,
    sso_code: Optional[str] = None,
    state_code: Optional[str] = None,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find Henry stations matching specified criteria.

    Simple high-level function for discovering Henry stations without
    directly instantiating a client.

    Args:
        project_code: Filter by project code
        sso_code: Filter by SSO office code (e.g., '2-SON', 'CA')
        state_code: Filter by state code (e.g., 'CA')
        active_only: Only return currently active stations (no removal_date)

    Returns:
        List of station dictionaries with metadata

    Raises:
        HenryError: If query fails

    Examples:
        # Async version
        stations = await find_henry_stations(sso_code='2-SON')

        # Sync version
        stations = find_henry_stations.sync(sso_code='2-SON')

        # Get active stations in California
        stations = find_henry_stations.sync(state_code='CA', active_only=True)
    """
    if not any([project_code, sso_code, state_code]):
        raise HenryError(
            "At least one filter parameter (project_code, sso_code, or state_code) is required"
        )

    async with HenryClient() as client:
        stations = await client.get_stations(
            project_code=project_code,
            sso_code=sso_code,
            state_code=state_code,
        )

        # Convert to dictionaries
        result = []
        for station in stations:
            station_dict = {
                "station_id": station.station_id,
                "station_name": station.station_name,
                "project_code": station.project_code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation_m": station.elevation_m,
                "state": station.state,
                "county": station.county,
                "sso_code": station.sso_code,
                "huc": station.huc,
            }

            # Filter for active only if requested
            if active_only:
                if station.removal_date is not None:
                    continue

            result.append(station_dict)

        return result


@add_sync_version
async def fetch_henry_data(
    station_id: str,
    variable_name: str,
    start_date: str,
    end_date: str,
    duration: str = "DAILY",
) -> List[Dict[str, Any]]:
    """
    Fetch time series data from a Henry station.

    Simple high-level function for fetching data without directly
    instantiating a client.

    Args:
        station_id: Station identifier (usersiteid)
        variable_name: Variable to fetch (e.g., 'soiltemp', 'soilVWC', 'airtemp', 'waterlevel')
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        duration: Data granularity ('DAILY' or 'HOURLY', default: 'DAILY')

    Returns:
        List of data point dictionaries with timestamp, value, flags

    Raises:
        HenryError: If query fails

    Examples:
        # Async version
        data = await fetch_henry_data(
            station_id='CA_SITE_001',
            variable_name='soiltemp',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Sync version
        data = fetch_henry_data.sync(
            station_id='CA_SITE_001',
            variable_name='soiltemp',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Get hourly data for 7 days
        data = fetch_henry_data.sync(
            station_id='CA_SITE_001',
            variable_name='soilVWC',
            start_date='2024-01-01',
            end_date='2024-01-07',
            duration='HOURLY'
        )
    """
    async with HenryClient() as client:
        data_points = await client.get_station_data(
            station_id=station_id,
            variable_name=variable_name,
            start_date=start_date,
            end_date=end_date,
            duration=duration,
        )

        # Convert to dictionaries
        result = []
        for dp in data_points:
            dp_dict = {
                "station_id": dp.station_id,
                "element_code": dp.element_code,
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "duration": dp.duration,
            }

            if dp.qc_flag:
                dp_dict["qc_flag"] = dp.qc_flag
            if dp.qa_flag:
                dp_dict["qa_flag"] = dp.qa_flag

            result.append(dp_dict)

        return result


@add_sync_version
async def get_henry_variables(
    station_id: str,
) -> List[Dict[str, Any]]:
    """
    Get available variables/sensors for a Henry station.

    Args:
        station_id: Station identifier (usersiteid)

    Returns:
        List of variable dictionaries describing available measurements

    Raises:
        HenryError: If station not found or query fails

    Examples:
        # Async version
        variables = await get_henry_variables(station_id='CA_SITE_001')

        # Sync version
        variables = get_henry_variables.sync(station_id='CA_SITE_001')
    """
    async with HenryClient() as client:
        sensors = await client.get_available_variables(station_id=station_id)

        # Convert to dictionaries
        result = []
        for sensor in sensors:
            sensor_dict = {
                "station_id": sensor.station_id,
                "variable_name": sensor.variable_name,
                "element_code": sensor.element_code,
                "sensor_description": sensor.sensor_description,
            }

            if sensor.depth_cm is not None:
                sensor_dict["depth_cm"] = sensor.depth_cm  # type: ignore

            result.append(sensor_dict)

        return result


@add_sync_version
async def list_henry_projects(
    sso_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List unique projects available in Henry database.

    This function discovers available projects by fetching stations
    and extracting unique project codes.

    Args:
        sso_code: Filter by SSO office code (optional)

    Returns:
        List of project information dictionaries

    Raises:
        HenryError: If query fails

    Examples:
        # Get all projects
        projects = await list_henry_projects()

        # Get projects for a specific SSO
        projects = await list_henry_projects(sso_code='2-SON')

        # Sync version
        projects = list_henry_projects.sync(sso_code='2-SON')
    """
    try:
        # Default to accessible SSO if none specified
        if sso_code is None:
            sso_code = "2-SON"

        async with HenryClient() as client:
            stations = await client.get_stations(sso_code=sso_code)

            # Extract unique projects
            projects_dict = {}
            for station in stations:
                project = station.project_code
                if project not in projects_dict:
                    projects_dict[project] = {
                        "project_code": project,
                        "station_count": 0,
                        "sso_code": station.sso_code,
                        "state": station.state,
                    }

                projects_dict[project]["station_count"] += 1  # type: ignore

            return list(projects_dict.values())

    except Exception as e:
        raise HenryError(f"Error listing projects: {e}") from e


__all__ = [
    "find_henry_stations",
    "fetch_henry_data",
    "get_henry_variables",
    "list_henry_projects",
]
