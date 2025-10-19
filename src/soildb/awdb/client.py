"""
AWDB (Air and Water Database) client for soildb.
"""

import json
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .exceptions import AWDBConnectionError, AWDBError, AWDBQueryError
from .models import StationInfo


class AWDBClient:
    """
    Client for accessing data via the NRCS AWDB REST API.

    The AWDB (Air-Water Database) API provides access to real-time and historical
    monitoring data from networks such as SCAN and SNOTEL.
    """

    BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": "soildb-awdb-client/0.1.0",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "AWDBClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Any:
        """Make a request to the AWDB API with error handling."""
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            raise AWDBConnectionError(f"Request timeout after {self.timeout}s") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise AWDBQueryError("Station or data not found") from e
            elif e.response.status_code == 429:
                raise AWDBConnectionError("Rate limit exceeded") from e
            elif e.response.status_code >= 500:
                raise AWDBConnectionError("AWDB service temporarily unavailable") from e
            else:
                raise AWDBConnectionError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e
        except httpx.RequestError as e:
            raise AWDBConnectionError(f"Network error: {e}") from e
        except json.JSONDecodeError as e:
            raise AWDBQueryError(f"Invalid JSON response: {e}") from e

    def get_stations(
        self,
        network_codes: Optional[List[str]] = None,
        state_codes: Optional[List[str]] = None,
    ) -> List[StationInfo]:
        """
        Get list of available stations.

        Args:
            network_codes: Optional list of network codes (e.g., ['SCAN', 'SNOTEL']).
                           If None, stations from all networks are returned.
            state_codes: List of state codes to filter by

        Returns:
            List of StationInfo objects
        """
        params = {"logicalAnd": "true"}

        if network_codes:
            params["networkCodes"] = ",".join(network_codes)

        if state_codes:
            params["stateCodes"] = ",".join(state_codes)

        try:
            data = self._make_request("stations", params)

            stations = []
            for station_data in data:
                try:
                    station = StationInfo(
                        station_triplet=station_data.get("stationTriplet", ""),
                        name=station_data.get("name", "Unknown"),
                        latitude=float(station_data.get("latitude", 0)),
                        longitude=float(station_data.get("longitude", 0)),
                        elevation=station_data.get("elevation"),
                        network_code=station_data.get("networkCode", "UNKNOWN"),
                        state=station_data.get("state"),
                        county=station_data.get("county"),
                    )
                    stations.append(station)
                except (ValueError, TypeError):
                    # Skip invalid station data but don't fail completely
                    continue

            return stations

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve station list: {e}") from e

    def find_nearby_stations(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 50.0,
        network_codes: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Tuple[StationInfo, float]]:
        """
        Find stations near a given location.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            max_distance_km: Maximum distance to search
            network_codes: Network codes to include
            limit: Maximum number of stations to return

        Returns:
            List of (StationInfo, distance_km) tuples, sorted by distance
        """
        stations = self.get_stations(network_codes=network_codes)

        nearby_stations = []
        for station in stations:
            distance = self._haversine_distance(
                latitude, longitude, station.latitude, station.longitude
            )

            if distance <= max_distance_km:
                nearby_stations.append((station, distance))

        # Sort by distance
        nearby_stations.sort(key=lambda x: x[1])

        return nearby_stations[:limit]

    def get_station_data(
        self,
        station_triplet: str,
        elements: str,
        start_date: str,
        end_date: str,
        duration: str = "DAILY",
        ordinal: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for a specific station and element.

        Args:
            station_triplet: Station identifier (e.g., '1234:UT:SNTL')
            elements: Element code (e.g., 'SMS', 'STO', 'PREC')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            duration: Data duration ('DAILY', 'HOURLY', etc.)
            ordinal: Sensor ordinal (1, 2, 3, etc. for multiple sensors)

        Returns:
            List of data points with timestamp, value, and flags
        """
        params = {
            "stationTriplets": station_triplet,
            "elements": elements,  # Fixed: use 'elements' not 'element_code'
            "ordinal": str(ordinal),
            "duration": duration,
            "getFlags": "true",
            "alwaysReturnDailyFeb29": "false",
            "beginDate": start_date,
            "endDate": end_date,
        }

        try:
            data = self._make_request("data", params)

            # Process the response data - handle nested structure properly
            processed_data: List[Dict[str, Any]] = []
            if not data:
                return processed_data

            for station_data in data:
                if "data" not in station_data or not station_data["data"]:
                    continue

                # Get the first element's data (should be the one we requested)
                element_data = station_data["data"][0]
                if "values" not in element_data:
                    continue

                for value_item in element_data["values"]:
                    try:
                        # Parse timestamp - handle different formats
                        date_str = value_item.get("date", "")
                        if not date_str:
                            continue

                        # Handle ISO format with timezone
                        if "T" in date_str:
                            # Remove Z suffix if present and add UTC
                            date_str = date_str.replace("Z", "+00:00")
                            timestamp = datetime.fromisoformat(date_str)
                        else:
                            # Assume YYYY-MM-DD format
                            timestamp = datetime.strptime(date_str, "%Y-%m-%d")

                        data_point = {
                            "timestamp": timestamp,
                            "value": value_item.get("value"),
                            "flags": [],  # Flags not available in current API response structure
                        }
                        processed_data.append(data_point)

                    except (ValueError, TypeError):
                        # Skip invalid data points
                        continue

            return processed_data

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve station data: {e}") from e

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth's radius in kilometers

        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c
