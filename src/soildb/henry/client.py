"""
Henry Mount Soil Climate Database client for asynchronous data access.

This client provides access to the Henry database hosted by UC Davis California
Soil Resource Lab, which contains NRCS-collected portable sensor data organized
by projects and SSO offices.

API Documentation:
- Endpoint: http://soilmap2-1.lawr.ucdavis.edu/henry/query.php
- Response format: Gzipped JSON
- Date format: YYYY-MM-DD HH:MM:SS (space-separated, not ISO T)
"""

import gzip
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from ..base_client import BaseDataAccessClient, ClientConfig
from .exceptions import HenryAPIError, HenryDataError, HenryNetworkError
from .models import HenrySensor, HenryStation, HenryTimeSeriesDataPoint
from .utils import (
    henry_variable_to_base_code,
)


class HenryClient(BaseDataAccessClient):
    """
    Async client for accessing data from the Henry Mount Soil Climate Database.

    The Henry database provides NRCS-collected portable sensor data organized by projects
    and SSO offices. Unlike SCAN/SNOTEL (permanent infrastructure), Henry includes
    temporary and portable sensor deployments.

    Usage:
        async with HenryClient() as client:
            stations = await client.get_stations(sso_code="2-SON")
            data = await client.get_station_data(
                station_id="CA_SITE_001",
                variable_name="soiltemp",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
    """

    BASE_URL = "http://soilmap2-1.lawr.ucdavis.edu/henry"

    def __init__(self, timeout: int = 60, config: Optional[ClientConfig] = None):
        """
        Initialize Henry client.

        Args:
            timeout: Request timeout in seconds (default: 60)
            config: ClientConfig instance with timeout and retry settings.
                   If provided, takes precedence over timeout parameter.

        Examples:
            >>> client = HenryClient(timeout=120)
            >>> client_with_config = HenryClient(config=ClientConfig.reliable())
        """
        if config is None:
            config = ClientConfig(timeout=float(timeout))
        else:
            if timeout != 60:
                config.timeout = float(timeout)

        super().__init__(config)

    async def __aenter__(self) -> "HenryClient":
        """Async context manager entry."""
        await super().__aenter__()
        return self

    @property
    def timeout(self) -> float:
        """Get the timeout in seconds (for backward compatibility)."""
        return self._config.timeout

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client with Henry-specific configuration."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            headers={
                "User-Agent": "soildb-henry-client/0.1.0",
                "Accept": "application/json",
            },
        )

    async def connect(self) -> bool:
        """
        Test connection to Henry API.

        Returns:
            True if connection successful

        Raises:
            HenryNetworkError: If connection fails
        """
        try:
            # Try to fetch sensors with SSO filter to test connectivity
            # Using small SSO with manageable data size
            await self._make_henry_request(
                "query.php", params={"what": "sensors", "sso": "2-SON"}
            )
            return True
        except Exception as e:
            raise HenryNetworkError(f"Connection test failed: {e}") from e

    async def _make_henry_request(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Make an async request to the Henry API with error handling.

        Henry API returns gzipped JSON responses that need decompression.

        Args:
            endpoint: API endpoint relative to BASE_URL (e.g., "query.php")
            params: Query parameters to include in the request

        Returns:
            Parsed JSON response (dict)

        Raises:
            HenryConnectionError: For network/timeout/server issues
            HenryAPIError: For API format or bad parameter errors
            HenryDataError: For invalid or unparseable data
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            # Make request with raw bytes response to handle gzip
            response = await super()._make_request("GET", url, params=params)
            response.raise_for_status()

            # Henry returns gzipped JSON responses
            content = response.content

            # Try to decompress if gzipped
            try:
                decompressed = gzip.decompress(content)
                data = json.loads(decompressed.decode("utf-8"))
            except gzip.BadGzipFile:
                # Response is not gzipped, try parsing directly
                data = response.json()

            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise HenryAPIError("Bad request - invalid parameters") from e
            elif e.response.status_code == 404:
                raise HenryAPIError("Resource not found") from e
            elif e.response.status_code == 413:
                raise HenryAPIError(
                    "Request too large - reduce data range or add more filters"
                ) from e
            elif e.response.status_code == 429:
                raise HenryNetworkError("Rate limit exceeded") from e
            elif e.response.status_code >= 500:
                raise HenryNetworkError("Henry service temporarily unavailable") from e
            else:
                raise HenryNetworkError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e

        except httpx.TimeoutException as e:
            raise HenryNetworkError(
                f"Request timeout after {self._config.timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise HenryNetworkError(f"Network error: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HenryDataError(f"Invalid response data: {e}") from e

    async def get_stations(
        self,
        project_code: Optional[str] = None,
        sso_code: Optional[str] = None,
        state_code: Optional[str] = None,
        station_ids: Optional[List[str]] = None,
    ) -> List[HenryStation]:
        """
        Fetch Henry station metadata.

        At least one filter parameter is required. The API accepts:
        - project: Project code(s)
        - sso: Soil Survey Office code(s) (e.g., '2-SON', 'CA')
        - usersiteid: Specific station ID(s)

        Args:
            project_code: Filter by project code
            sso_code: Filter by SSO office code (e.g., '2-SON' for accessible stations)
            state_code: Filter by state code (e.g., 'CA')
            station_ids: List of specific station IDs (usersiteid values)

        Returns:
            List of HenryStation objects

        Raises:
            HenryAPIError: If API returns error or no stations found
            HenryNetworkError: If network request fails
        """
        if not any([project_code, sso_code, state_code, station_ids]):
            raise HenryAPIError(
                "At least one filter parameter (project, sso, state, or station_ids) is required"
            )

        params = {"what": "sensors"}

        if project_code:
            params["project"] = project_code
        if sso_code:
            params["sso"] = sso_code
        if state_code:
            params["state"] = state_code
        if station_ids:
            params["usersiteid"] = ",".join(station_ids)

        try:
            response = await self._make_henry_request("query.php", params=params)

            if "sensors" not in response:
                raise HenryDataError("Missing 'sensors' key in API response")

            sensors = response.get("sensors", [])

            if not sensors:
                return []

            # Parse sensors into HenryStation objects
            stations = []
            seen_ids = set()

            for sensor in sensors:
                station_id = sensor.get("sid")
                if not station_id or station_id in seen_ids:
                    continue

                seen_ids.add(station_id)

                # Parse installation dates if present
                install_date = None
                if "installation_date" in sensor and sensor["installation_date"]:
                    try:
                        install_date = datetime.fromisoformat(
                            sensor["installation_date"]
                        )
                    except (ValueError, TypeError):
                        pass

                removal_date = None
                if "removal_date" in sensor and sensor["removal_date"]:
                    try:
                        removal_date = datetime.fromisoformat(sensor["removal_date"])
                    except (ValueError, TypeError):
                        pass

                station = HenryStation(
                    station_id=station_id,
                    station_name=sensor.get("name", "Unknown"),
                    project_code=sensor.get("project", ""),
                    sso_code=sensor.get("sso"),
                    latitude=float(sensor.get("wgs84_latitude", 0)),
                    longitude=float(sensor.get("wgs84_longitude", 0)),
                    elevation_m=int(sensor.get("elevation_m"))
                    if sensor.get("elevation_m")
                    else None,
                    state=sensor.get("state"),
                    county=sensor.get("county"),
                    huc=sensor.get("huc"),
                    installation_date=install_date,
                    removal_date=removal_date,
                )

                stations.append(station)

            return stations

        except HenryDataError:
            raise
        except HenryAPIError:
            raise
        except Exception as e:
            raise HenryAPIError(f"Error fetching stations: {e}") from e

    async def get_station_data(
        self,
        station_id: str,
        variable_name: str,
        start_date: str,
        end_date: str,
        duration: str = "DAILY",
    ) -> List[HenryTimeSeriesDataPoint]:
        """
        Fetch time series data for a Henry station.

        Args:
            station_id: Station identifier (usersiteid)
            variable_name: Variable to fetch ('soiltemp', 'soilVWC', 'airtemp', 'waterlevel', 'all')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            duration: Data granularity ('DAILY' or 'HOURLY', default: 'DAILY')

        Returns:
            List of HenryTimeSeriesDataPoint objects

        Raises:
            HenryAPIError: If API returns error or invalid parameters
            HenryNetworkError: If network request fails
        """
        params = {
            "what": variable_name,
            "usersiteid": station_id,
            "start": start_date,
            "stop": end_date,
            "gran": "day" if duration == "DAILY" else "hour",
        }

        try:
            response = await self._make_henry_request("query.php", params=params)

            # Response structure depends on variable_name
            # If 'all', response has keys: soiltemp, soilVWC, airtemp, waterlevel
            # If specific variable, response has that key

            data_points = []

            # Determine which keys to look for
            if variable_name == "all":
                data_keys = ["soiltemp", "soilVWC", "airtemp", "waterlevel"]
            else:
                data_keys = [variable_name]

            # Extract element code from variable
            base_code = henry_variable_to_base_code(
                variable_name if variable_name != "all" else "soiltemp"
            )

            for data_key in data_keys:
                if data_key not in response:
                    continue

                records = response.get(data_key, [])
                if not records:
                    continue

                for record in records:
                    timestamp_str = record.get("date_time")
                    if not timestamp_str:
                        continue

                    # Parse Henry timestamp format (YYYY-MM-DD HH:MM:SS)
                    try:
                        # Handle both space-separated and ISO formats
                        if " " in timestamp_str:
                            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        else:
                            dt = datetime.fromisoformat(timestamp_str)
                    except (ValueError, TypeError):
                        continue

                    # Get element code for this variable/depth
                    element_code = base_code

                    # Extract sensor value
                    value = record.get("sensor_value")
                    if value is not None:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = None

                    data_point = HenryTimeSeriesDataPoint(
                        station_id=station_id,
                        element_code=element_code,
                        timestamp=dt,
                        value=value,
                        duration=duration,
                        qc_flag=record.get("qc_flag"),
                        qa_flag=record.get("qa_flag"),
                    )

                    data_points.append(data_point)

            return data_points

        except (HenryDataError, HenryAPIError):
            raise
        except Exception as e:
            raise HenryAPIError(f"Error fetching station data: {e}") from e

    async def get_available_variables(
        self,
        station_id: str,
    ) -> List[HenrySensor]:
        """
        Get available variables/sensors for a Henry station.

        This fetches sensor metadata to understand what variables and depths
        are available at a specific station.

        Args:
            station_id: Station identifier (usersiteid)

        Returns:
            List of HenrySensor objects describing available variables

        Raises:
            HenryAPIError: If station not found or API error
            HenryNetworkError: If network request fails
        """
        try:
            # Fetch station sensors to get metadata
            stations = await self.get_stations(station_ids=[station_id])

            if not stations:
                raise HenryAPIError(f"Station {station_id} not found")

            # For now, return basic sensors based on common Henry variables
            # In a real implementation, we might parse depth from sensor descriptions
            # station = stations[0]

            sensors = []
            for variable_name in ["soiltemp", "soilVWC", "airtemp", "waterlevel"]:
                base_code = henry_variable_to_base_code(variable_name)

                sensor = HenrySensor(
                    station_id=station_id,
                    variable_name=variable_name,
                    element_code=base_code,
                    sensor_description=f"{variable_name} sensor",
                )

                sensors.append(sensor)

            return sensors

        except HenryAPIError:
            raise
        except Exception as e:
            raise HenryAPIError(f"Error fetching variables: {e}") from e


__all__ = ["HenryClient"]
