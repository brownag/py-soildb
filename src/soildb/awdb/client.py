"""
AWDB (Air and Water Database) client for soildb.
"""

import json
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..base_client import BaseDataAccessClient, ClientConfig
from .exceptions import AWDBConnectionError, AWDBError, AWDBQueryError
from .models import ForecastData, ReferenceData, StationInfo, TimeSeriesDataPoint


class AWDBClient(BaseDataAccessClient):
    """
    Async client for accessing data via the NRCS AWDB REST API.

    The AWDB (Air-Water Database) API provides access to real-time and historical
    monitoring data from networks such as SCAN and SNOTEL.
    """

    BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

    def __init__(self, timeout: int = 60, config: Optional[ClientConfig] = None):
        """
        Initialize AWDB client.

        Can be initialized either with a timeout parameter or with a ClientConfig object.
        If config is provided, it takes precedence over the timeout parameter.

        Args:
            timeout: Request timeout in seconds (default: 60)
            config: ClientConfig instance with timeout and retry settings.
                   If provided, takes precedence over timeout parameter.

        Examples:
            >>> # Using timeout parameter
            >>> client = AWDBClient(timeout=120)

            >>> # Using ClientConfig with presets
            >>> config = ClientConfig.reliable()
            >>> client = AWDBClient(config=config)
        """
        if config is None:
            config = ClientConfig(timeout=float(timeout))
        else:
            # If timeout is explicitly different from default, use it
            if timeout != 60:
                config.timeout = float(timeout)

        super().__init__(config)

    @property
    def timeout(self) -> float:
        """Get the timeout in seconds (for backward compatibility).

        Returns:
            float: Timeout value from config
        """
        return self._config.timeout

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client with AWDB-specific configuration.

        Returns:
            httpx.AsyncClient: Configured client with AWDB headers
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            headers={
                "User-Agent": "soildb-awdb-client/0.1.0",
                "Accept": "application/json",
            },
        )

    async def connect(self) -> bool:
        """
        Test connection to AWDB service.

        Returns:
            True if connection successful

        Raises:
            AWDBConnectionError: If connection fails
        """
        try:
            # Try to get a simple endpoint that requires no parameters
            await self._make_request("reference-data")
            return True
        except Exception as e:
            raise AWDBConnectionError(f"Connection test failed: {e}") from e

    async def _make_request(  # type: ignore[override]
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Any:
        """Make an async request to the AWDB API with error handling.

        This method wraps the base class retry logic with AWDB-specific error handling.

        Args:
            endpoint: API endpoint relative to BASE_URL (e.g., "stations", "data")
            params: Query parameters to include in the request

        Returns:
            Parsed JSON response

        Raises:
            AWDBQueryError: If response status indicates bad parameters or not found
            AWDBConnectionError: If there are network, timeout, or server issues
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = await super()._make_request("GET", url, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise AWDBQueryError("Bad request - invalid parameters") from e
            elif e.response.status_code == 404:
                raise AWDBQueryError("Station or data not found") from e
            elif e.response.status_code == 413:
                raise AWDBQueryError(
                    "Request too large - reduce data range or parameters"
                ) from e
            elif e.response.status_code == 429:
                raise AWDBConnectionError("Rate limit exceeded") from e
            elif e.response.status_code >= 500:
                raise AWDBConnectionError("AWDB service temporarily unavailable") from e
            else:
                raise AWDBConnectionError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e
        except httpx.TimeoutException as e:
            raise AWDBConnectionError(
                f"Request timeout after {self._config.timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise AWDBConnectionError(f"Network error: {e}") from e
        except json.JSONDecodeError as e:
            raise AWDBQueryError(f"Invalid JSON response: {e}") from e

    async def get_stations(
        self,
        network_codes: Optional[List[str]] = None,
        state_codes: Optional[List[str]] = None,
        station_triplets: Optional[List[str]] = None,
        station_names: Optional[List[str]] = None,
        dco_codes: Optional[List[str]] = None,
        county_names: Optional[List[str]] = None,
        elements: Optional[List[str]] = None,
        durations: Optional[List[str]] = None,
        hucs: Optional[List[str]] = None,
        return_forecast_point_metadata: bool = False,
        return_reservoir_metadata: bool = False,
        return_station_elements: bool = False,
        active_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[StationInfo]:
        """
        Get list of available stations with comprehensive filtering options.

        This method leverages the full AWDB API filtering capabilities, including wildcard support
        for station triplets, names, counties, and HUCs. All filtering is performed server-side
        for optimal performance.

        Args:
            network_codes: Filter by network codes (e.g., ['SCAN', 'SNTL']). Supports wildcards like ['*:OR:*'].
            state_codes: Filter by state codes (e.g., ['OR', 'WA'])
            station_triplets: Filter by station triplets with wildcards (e.g., ['*:OR:SNTL', '302:*:*'])
            station_names: Filter by station names with wildcards (e.g., ['*Lake*', 'Alab*'])
            dco_codes: Filter by DCO codes
            county_names: Filter by county names with wildcards (e.g., ['*County', 'Wall*'])
            elements: Filter stations with specific elements (format: elementCode:heightDepth:ordinal, supports wildcards)
            durations: Filter stations by data durations (HOURLY, DAILY, SEMIMONTHLY, MONTHLY, CALENDAR_YEAR, WATER_YEAR, SEASONAL)
            hucs: Filter by HUC codes with wildcards (e.g., ['170601*', '*050101'])
            return_forecast_point_metadata: Include forecast point metadata for applicable stations
            return_reservoir_metadata: Include reservoir metadata for applicable stations
            return_station_elements: Include detailed station elements information
            active_only: Return only active stations (default: True)

        Returns:
            List of StationInfo objects with enhanced metadata based on request parameters

        Raises:
            AWDBQueryError: If the request parameters are invalid
            AWDBConnectionError: If there are network or server issues

        Examples:
            # Get all SNOTEL stations in Oregon or Washington
            stations = client.get_stations(station_triplets=['*:OR:SNTL', '*:WA:SNTL'])

            # Get stations with soil moisture sensors in California
            stations = client.get_stations(elements=['SMS:*'], state_codes=['CA'])

            # Get stations in specific HUC areas
            stations = client.get_stations(hucs=['170601*'])
        """
        params = {}

        # Build parameters for API request
        triplet_patterns = []
        if station_triplets:
            triplet_patterns.extend(station_triplets)

        if network_codes and state_codes:
            for network in network_codes:
                for state in state_codes:
                    triplet_patterns.append(f"*:{state}:{network}")
        elif network_codes:
            # Network-only patterns - use wildcard triplets since API doesn't have networkCodes param
            for network in network_codes:
                triplet_patterns.append(f"*:*:{network}")
        elif state_codes:
            # State-only patterns (all networks in state)
            for state in state_codes:
                triplet_patterns.append(f"*:*:*{state}")

        if triplet_patterns:
            params["stationTriplets"] = ",".join(triplet_patterns)
        if station_names:
            params["stationNames"] = ",".join(station_names)
        if dco_codes:
            params["dcoCodes"] = ",".join(dco_codes)
        if county_names:
            params["countyNames"] = ",".join(county_names)
        if elements:
            params["elements"] = ",".join(elements)
        if durations:
            params["durations"] = ",".join(durations)
        if hucs:
            params["hucs"] = ",".join(hucs)

        params["returnForecastPointMetadata"] = str(
            return_forecast_point_metadata
        ).lower()
        params["returnReservoirMetadata"] = str(return_reservoir_metadata).lower()
        params["returnStationElements"] = str(return_station_elements).lower()
        params["activeOnly"] = str(active_only).lower()

        try:
            data = await self._make_request("stations", params)

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
                        # Additional metadata fields
                        station_id=station_data.get("stationId"),
                        dco_code=station_data.get("dcoCode"),
                        huc=station_data.get("huc"),
                        data_time_zone=station_data.get("dataTimeZone"),
                        pedon_code=station_data.get("pedonCode"),
                        shef_id=station_data.get("shefId"),
                        operator=station_data.get("operator"),
                        begin_date=station_data.get("beginDate"),
                        end_date=station_data.get("endDate"),
                        forecast_point=station_data.get("forecastPoint"),
                        reservoir_metadata=station_data.get("reservoirMetadata"),
                        station_elements=station_data.get("stationElements"),
                    )

                    stations.append(station)
                except (ValueError, TypeError):
                    # Skip invalid station data but don't fail completely
                    continue

            return stations[:limit] if limit else stations

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve station list: {e}") from e

    async def find_nearby_stations(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 50.0,
        network_codes: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Tuple[StationInfo, float]]:
        """
        Find stations near a given location.

        Note: This method fetches all stations and filters client-side for distance.
        For large-scale applications, consider using the get_stations method with
        geographic filtering if available in future API versions.

        Args:
            latitude: Target latitude (-90 to 90)
            longitude: Target longitude (-180 to 180)
            max_distance_km: Maximum search distance in kilometers
            network_codes: Optional network codes to filter by before distance calculation
            limit: Maximum number of nearest stations to return

        Returns:
            List of (StationInfo, distance_km) tuples, sorted by ascending distance

        Raises:
            ValueError: If latitude/longitude are outside valid ranges
            AWDBQueryError: If station retrieval fails
        """
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")

        stations = await self.get_stations(network_codes=network_codes)

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

    async def get_station_data(
        self,
        station_triplet: str,
        elements: str,
        start_date: str,
        end_date: str,
        duration: str = "DAILY",
        ordinal: int = 1,
        period_ref: str = "END",
        central_tendency_type: str = "NONE",
        return_flags: bool = True,
        return_original_values: bool = False,
        return_suspect_data: bool = False,
        insert_or_update_begin_date: Optional[str] = None,
    ) -> List[TimeSeriesDataPoint]:
        """
        Get time series data for a specific station and element with enhanced options.

        This method provides access to all AWDB data retrieval parameters, including
        quality flags, central tendency calculations, and incremental data updates.

        Args:
            station_triplet: Station identifier (e.g., '1234:UT:SNTL')
            elements: Element code (e.g., 'SMS', 'STO', 'PREC')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            duration: Data duration ('DAILY', 'HOURLY', 'MONTHLY', etc.)
            ordinal: Sensor ordinal (1, 2, 3, etc. for multiple sensors)
            period_ref: Timestamp reference ('START' or 'END' of period)
            central_tendency_type: Include averages/medians ('NONE', 'ALL', 'MEDIAN', 'AVERAGE')
            return_flags: Include quality control and assurance flags
            return_original_values: Include original values before processing
            return_suspect_data: Include data marked as suspect (normally filtered)
            insert_or_update_begin_date: For incremental updates (YYYY-MM-DD format)

        Returns:
            List of TimeSeriesDataPoint objects with comprehensive data and quality information

        Raises:
            AWDBQueryError: If station/element not found or invalid parameters
            AWDBConnectionError: If there are network or server issues
        """
        params = {
            "stationTriplets": station_triplet,
            "elements": elements,
            "ordinal": str(ordinal),
            "duration": duration,
            "returnFlags": str(return_flags).lower(),
            "alwaysReturnDailyFeb29": "false",
            "beginDate": start_date,
            "endDate": end_date,
            "periodRef": period_ref,
            "centralTendencyType": central_tendency_type,
            "returnOriginalValues": str(return_original_values).lower(),
            "returnSuspectData": str(return_suspect_data).lower(),
        }

        if insert_or_update_begin_date:
            params["insertOrUpdateBeginDate"] = insert_or_update_begin_date

        try:
            data = await self._make_request("data", params)

            # Process the response data - handle nested structure properly
            processed_data: List[TimeSeriesDataPoint] = []
            if not data:
                return processed_data

            for station_data in data:
                if "data" not in station_data or not station_data["data"]:
                    continue

                # Check for error messages
                if "error" in station_data:
                    raise AWDBQueryError(f"API error: {station_data['error']}")

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

                        # Build flags list from available flag fields
                        flags = []
                        if value_item.get("qcFlag"):
                            flags.append(f"QC:{value_item['qcFlag']}")
                        if value_item.get("qaFlag"):
                            flags.append(f"QA:{value_item['qaFlag']}")
                        if value_item.get("origQcFlag"):
                            flags.append(f"ORIG_QC:{value_item['origQcFlag']}")

                        data_point = TimeSeriesDataPoint(
                            timestamp=timestamp,
                            value=value_item.get("value"),
                            flags=flags,
                            qc_flag=value_item.get("qcFlag"),
                            qa_flag=value_item.get("qaFlag"),
                            orig_value=value_item.get("origValue"),
                            orig_qc_flag=value_item.get("origQcFlag"),
                            average=value_item.get("average"),
                            median=value_item.get("median"),
                            month=value_item.get("month"),
                            month_part=value_item.get("monthPart"),
                            year=value_item.get("year"),
                            collection_date=value_item.get("collectionDate"),
                        )
                        processed_data.append(data_point)

                    except (ValueError, TypeError):
                        # Skip invalid data points
                        continue

            return processed_data

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve station data: {e}") from e

    async def check_station_data_availability(
        self,
        station_triplet: str,
        elements: str,
        start_date: str,
        end_date: str,
        duration: str = "MONTHLY",
        ordinal: int = 1,
    ) -> Dict[str, Any]:
        """
        Check data availability for a station by querying coarser time granularity.

        This method helps determine if a station has any data for a given element
        by using monthly aggregation, which is more likely to return results than
        daily data for sparse or intermittent measurements.

        Args:
            station_triplet: Station identifier (e.g., '1234:UT:SNTL')
            elements: Element code (e.g., 'SMS', 'STO', 'PREC')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            duration: Data duration to check ('MONTHLY', 'YEARLY', etc.)
            ordinal: Sensor ordinal (1, 2, 3, etc. for multiple sensors)

        Returns:
            Dictionary with availability information:
            - 'has_data': Boolean indicating if any data exists
            - 'data_points': Number of data points found
            - 'date_range': Date range checked
            - 'sample_values': List of up to 5 sample values with dates

        Raises:
            AWDBQueryError: If station/element not found or invalid parameters
            AWDBConnectionError: If there are network or server issues
        """
        try:
            # Try to get data with coarser granularity first
            data = await self.get_station_data(
                station_triplet=station_triplet,
                elements=elements,
                start_date=start_date,
                end_date=end_date,
                duration=duration,
                ordinal=ordinal,
                return_flags=False,  # Simplify response
            )

            # Extract sample values
            sample_values = []
            for point in data[:5]:  # Limit to 5 samples
                sample_values.append(
                    {
                        "date": point.timestamp.date().isoformat(),
                        "value": point.value,
                    }
                )

            return {
                "has_data": len(data) > 0,
                "data_points": len(data),
                "date_range": {"start": start_date, "end": end_date},
                "duration": duration,
                "sample_values": sample_values,
            }

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to check data availability: {e}") from e

    async def get_forecasts(
        self,
        station_triplets: List[str],
        element_codes: Optional[List[str]] = None,
        begin_publication_date: Optional[str] = None,
        end_publication_date: Optional[str] = None,
        exceedence_probabilities: Optional[List[int]] = None,
        forecast_periods: Optional[List[str]] = None,
    ) -> List[ForecastData]:
        """
        Get forecast data for one or more stations.

        Args:
            station_triplets: List of station triplets (e.g., ['302:OR:SNTL'])
            element_codes: List of element codes (e.g., ['RESC', 'SRVO'])
            begin_publication_date: Start date for publication period (YYYY-MM-DD)
            end_publication_date: End date for publication period (YYYY-MM-DD)
            exceedence_probabilities: List of exceedence probabilities (e.g., [10, 30, 50])
            forecast_periods: List of forecast periods (e.g., ['03-01', '07-31'])

        Returns:
            List of ForecastData objects
        """
        params = {
            "stationTriplets": ",".join(station_triplets),
        }

        if element_codes:
            params["elementCodes"] = ",".join(element_codes)
        if begin_publication_date:
            params["beginPublicationDate"] = begin_publication_date
        if end_publication_date:
            params["endPublicationDate"] = end_publication_date
        if exceedence_probabilities:
            params["exceedenceProbabilities"] = ",".join(
                map(str, exceedence_probabilities)
            )
        if forecast_periods:
            params["forecastPeriods"] = ",".join(forecast_periods)

        try:
            data = await self._make_request("forecasts", params)

            forecasts = []
            for forecast_data in data:
                try:
                    forecast = ForecastData(
                        station_triplet=forecast_data.get("stationTriplet", ""),
                        forecast_point_name=forecast_data.get("forecastPointName"),
                        data=forecast_data.get("data", []),
                    )
                    forecasts.append(forecast)
                except (ValueError, TypeError):
                    continue

            return forecasts

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve forecast data: {e}") from e

    async def get_reference_data(
        self,
        reference_lists: Optional[List[str]] = None,
    ) -> ReferenceData:
        """
        Get reference data from AWDB.

        Retrieves lookup tables and reference information used by the AWDB system,
        including elements, networks, units, states, and other metadata.

        Args:
            reference_lists: List of reference data types to retrieve. Available options:
                            'dcos', 'durations', 'elements', 'forecastPeriods', 'functions',
                            'instruments', 'networks', 'physicalElements', 'states', 'units'.
                            If None, returns all reference data.

        Returns:
            ReferenceData object containing requested reference information

        Raises:
            AWDBQueryError: If invalid reference list specified
            AWDBConnectionError: If there are network or server issues
        """
        params = {}
        if reference_lists:
            params["referenceLists"] = ",".join(reference_lists)

        try:
            data = await self._make_request("reference-data", params)

            # The API returns a dict directly (not a list)
            if data and isinstance(data, dict):
                return ReferenceData(
                    dcos=data.get("dcos"),
                    durations=data.get("durations"),
                    elements=data.get("elements"),
                    forecast_periods=data.get("forecastPeriods"),
                    functions=data.get("functions"),
                    instruments=data.get("instruments"),
                    networks=data.get("networks"),
                    physical_elements=data.get("physicalElements"),
                    states=data.get("states"),
                    units=data.get("units"),
                )
            else:
                return ReferenceData()

        except AWDBError:
            raise
        except Exception as e:
            raise AWDBQueryError(f"Failed to retrieve reference data: {e}") from e

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
