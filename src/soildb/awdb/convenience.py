"""
High-level convenience functions for AWDB data access.

This module provides a clean, intuitive API organized by function family:

1. **Discovery Functions** (finding stations)
   - discover_stations_nearby(lat, lon, ...) - geographic discovery
   - discover_stations(network_codes, state_codes, ...) - attribute-based discovery

2. **Data Retrieval** (getting measurements)
   - get_property_data_near(lat, lon, property, ...) - time series near location
   - get_soil_moisture_by_depth(station_triplet, depths, ...) - multi-depth soil data

3. **Metadata Functions** (sensor inventory)
   - station_sensors(station_triplet) - comprehensive sensor info
   - station_available_properties(station_triplet) - available variables
   - station_sensor_depths(station_triplet, property) - available depths/heights

Legacy function names are supported for backward compatibility but are deprecated.
See convenience_api_migration() for usage examples and migration guide.
"""
# mypy: disable-error-code="attr-defined"

import warnings
from typing import Any, Dict, List, Optional

from ..utils import add_sync_version
from .client import AWDBClient
from .exceptions import AWDBError

# Mapping from property names to AWDB element codes
# Note: Some properties can have height/depth specifications in format elementCode:heightDepth:ordinal
PROPERTY_ELEMENT_MAP = {
    "soil_moisture": "SMS",  # Soil Moisture Percent (requires depth, negative inches)
    "soil_temp": "STO",  # Soil Temperature (requires depth, negative inches)
    "precipitation": "PREC",  # Precipitation (no height/depth)
    "air_temp": "TOBS",  # Air Temperature Observed (can have height, positive inches)
    "snow_water_equivalent": "WTEQ",  # Snow Water Equivalent (no height/depth)
    "snow_depth": "SNWD",  # Snow Depth (no height/depth)
    "wind_speed": "WSPD",  # Wind Speed (can have height, positive inches)
    "wind_direction": "WDIR",  # Wind Direction (can have height, positive inches)
    "relative_humidity": "RHUM",  # Relative Humidity (can have height, positive inches)
    "solar_radiation": "SRAD",  # Solar Radiation (can have height, positive inches)
    # Additional mappings based on AWDB reference data
    "battery": "BATT",  # Battery voltage
    "dew_point_temp": "DPTP",  # Dew Point Temperature
    "real_dielectric_constant": "RDC",  # Real Dielectric Constant (soil moisture proxy)
    "salinity": "SAL",  # Salinity
    "snow_density": "SNDN",  # Snow Density
    "snow_temp": "STEMP",  # Snow Temperature
    "barometric_pressure": "PRES",  # Barometric Pressure
    "photosynthetic_radiation": "PARV",  # Photosynthetically Active Radiation
    "stream_stage": "SRMO",  # Stream Stage (gauge height)
    "stream_discharge": "SRDOX",  # Stream Discharge
    "evaporation": "EVAP",  # Evaporation
    "fuel_moisture": "FUEL",  # Fuel Moisture
    "net_solar_radiation": "NTRDV",  # Net Solar Radiation
    "vapor_pressure": "SVPV",  # Vapor Pressure - Saturated
    # Additional mappings for unknown elements we discovered
    "precipitation_increment": "PRCP",  # Precipitation Increment
    "vapor_pressure_partial": "PVPV",  # Vapor Pressure - Partial
    "relative_humidity_enclosure": "RHENC",  # Relative Humidity Enclosure
    "precipitation_month_to_date": "PRCPMTD",  # Precipitation Month-to-Date
    "relative_humidity_min": "RHUMN",  # Relative Humidity Minimum
    "relative_humidity_max": "RHUMX",  # Relative Humidity Maximum
    "soil_moisture_min": "SMN",  # Soil Moisture Minimum
    "soil_moisture_avg": "SMV",  # Soil Moisture Average
    "soil_moisture_max": "SMX",  # Soil Moisture Maximum
    "solar_radiation_total": "SRADT",  # Solar Radiation Total
    "solar_radiation_avg": "SRADV",  # Solar Radiation Average
    "soil_temp_min": "STN",  # Soil Temperature Minimum
    "soil_temp_avg": "STV",  # Soil Temperature Average
    "soil_temp_max": "STX",  # Soil Temperature Maximum
    "air_temp_avg": "TAVG",  # Air Temperature Average
    "air_temp_max": "TMAX",  # Air Temperature Maximum
    "air_temp_min": "TMIN",  # Air Temperature Minimum
    "wind_direction_avg": "WDIRV",  # Wind Direction Average
    "wind_speed_avg": "WSPDV",  # Wind Speed Average
    "wind_speed_max": "WSPDX",  # Wind Speed Maximum
    # Newly discovered unknown elements with definitions
    "precipitation_increment_snow_adjusted": "PRCPSA",  # PRECIPITATION INCREMENT - SNOW-ADJ
    "snow_rain_ratio": "SNRR",  # Snow Rain Ratio
    "snow_water_equivalent_maximum": "WTEQX",  # SNOW WATER EQUIVALENT MAXIMUM
    # ZDUM remains unmapped - no definition found in API reference data
}

# Properties that can have height/depth specification
HEIGHT_DEPTH_PROPERTIES = {
    "soil_moisture",
    "soil_temp",  # Below surface (negative heights)
    "air_temp",
    "wind_speed",
    "wind_direction",
    "relative_humidity",
    "solar_radiation",  # Above surface (positive heights)
}

# Soil properties that require depth specification (subset of HEIGHT_DEPTH_PROPERTIES)
REQUIRED_DEPTH_PROPERTIES = {"soil_moisture", "soil_temp"}

# Properties that can optionally have height specification (above surface)
OPTIONAL_HEIGHT_PROPERTIES = {
    "air_temp",
    "wind_speed",
    "wind_direction",
    "relative_humidity",
    "solar_radiation",
}

# Properties that don't support height/depth specification
FIXED_PROPERTIES = {
    "precipitation",
    "snow_water_equivalent",
    "snow_depth",
    "battery",
    "dew_point_temp",
    "real_dielectric_constant",
    "salinity",
    "snow_density",
    "snow_temp",
    "barometric_pressure",
    "photosynthetic_radiation",
    "stream_stage",
    "stream_discharge",
    "evaporation",
    "fuel_moisture",
    "net_solar_radiation",
    "vapor_pressure",
}

# Units for each property (will be dynamically updated from API)
PROPERTY_UNITS = {
    "soil_moisture": "pct",  # volumetric %
    "soil_temp": "degF",  # Fahrenheit
    "precipitation": "inch",  # inches
    "air_temp": "degF",  # Fahrenheit
    "snow_water_equivalent": "inch",  # inches
    "snow_depth": "inch",  # inches
    "wind_speed": "mph",  # miles per hour
    "wind_direction": "degrees",
    "relative_humidity": "%",
    "solar_radiation": "watt/m2",
    # Additional units for new properties
    "battery": "volt",
    "dew_point_temp": "degF",
    "real_dielectric_constant": "unitless",
    "salinity": "gram/l",
    "snow_density": "kg/m3",
    "snow_temp": "degF",
    "barometric_pressure": "kPa",
    "photosynthetic_radiation": "watt/m2",
    "stream_stage": "ft",
    "stream_discharge": "cfs",
    "evaporation": "inch",
    "fuel_moisture": "pct",
    "net_solar_radiation": "watt/m2",
    "vapor_pressure": "kPa",
    # Units for additional discovered elements
    "precipitation_increment": "inch",
    "vapor_pressure_partial": "kPa",
    "relative_humidity_enclosure": "pct",
    "precipitation_month_to_date": "inch",
    "relative_humidity_min": "pct",
    "relative_humidity_max": "pct",
    "soil_moisture_min": "pct",
    "soil_moisture_avg": "pct",
    "soil_moisture_max": "pct",
    "solar_radiation_total": "watt/m2",
    "solar_radiation_avg": "watt/m2",
    "soil_temp_min": "degF",
    "soil_temp_avg": "degF",
    "soil_temp_max": "degF",
    "air_temp_avg": "degF",
    "air_temp_max": "degF",
    "air_temp_min": "degF",
    "wind_direction_avg": "degrees",
    "wind_speed_avg": "mph",
    "wind_speed_max": "mph",
    # Units for newly mapped unknown elements
    "precipitation_increment_snow_adjusted": "in",  # Snow Adjusted Total Precipitation
    "snow_rain_ratio": "unitless",  # Snow Rain Ratio (dimensionless)
    "snow_water_equivalent_maximum": "in",  # Maximum Snow Water Equivalent
}


@add_sync_version
async def discover_stations_nearby(
    latitude: float,
    longitude: float,
    max_distance_km: float = 50.0,
    network_codes: Optional[List[str]] = None,
    limit: int = 10,
    include_sensor_metadata: bool = False,
) -> List[Dict]:
    """
    Discover AWDB stations near a geographic location.

    Args:
        latitude: Target latitude (WGS84)
        longitude: Target longitude (WGS84)
        max_distance_km: Maximum search distance in kilometers
        network_codes: Network codes to include (e.g., 'SCAN', 'SNTL')
        limit: Maximum number of stations to return
        include_sensor_metadata: Include detailed sensor information for each station

    Returns:
        List of station dictionaries with distance information and optional sensor metadata

    Examples:
        >>> stations = await discover_stations_nearby(42.0, -93.6, max_distance_km=25)
        >>> nearby_snotel = await discover_stations_nearby(
        ...     42.0, -93.6, network_codes=['SNTL'], limit=5, include_sensor_metadata=True
        ... )
    """
    async with AWDBClient() as client:
        stations_with_distance = await client.find_nearby_stations(
            latitude, longitude, max_distance_km, network_codes, limit
        )

        result = []
        for station, distance in stations_with_distance:
            station_dict = {
                "station_triplet": station.station_triplet,
                "name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation": station.elevation,
                "network_code": station.network_code,
                "state": station.state,
                "county": station.county,
                "distance_km": round(distance, 2),
            }

            # Add sensor metadata if requested
            if include_sensor_metadata:
                try:
                    sensor_metadata = await station_sensors(station.station_triplet)
                    station_dict["sensor_metadata"] = sensor_metadata["sensors"]
                except Exception as e:
                    station_dict["sensor_metadata"] = {"error": str(e)}

            result.append(station_dict)

        return result


@add_sync_version
async def discover_stations(
    network_codes: Optional[List[str]] = None,
    state_codes: Optional[List[str]] = None,
    station_triplets: Optional[List[str]] = None,
    station_names: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    active_only: bool = True,
    limit: Optional[int] = None,
    include_sensor_metadata: bool = False,
) -> List[Dict]:
    """
    Discover stations using advanced filtering criteria with wildcard support.

    This function leverages the full AWDB API filtering capabilities for efficient
    server-side filtering using wildcards. When include_sensor_metadata=True, it
    automatically queries station element details to provide complete sensor inventories.

    Args:
        network_codes: Network codes with wildcards (e.g., ['*:OR:*'] for all networks in OR)
        state_codes: State codes (e.g., ['OR', 'WA'])
        station_triplets: Station triplets with wildcards (e.g., ['*:OR:SNTL'])
        station_names: Station names with wildcards (e.g., ['*Lake*'])
        elements: Element codes with wildcards (e.g., ['SMS:*', 'STO:-20:*'])
        active_only: Return only active stations
        limit: Maximum number of stations to return
        include_sensor_metadata: Include detailed sensor information for each station

    Returns:
        List of station dictionaries with optional sensor metadata

    Examples:
        >>> # Find all SNOTEL stations in Oregon
        >>> stations = await discover_stations(station_triplets=['*:OR:SNTL'])
        >>>
        >>> # Find stations with soil moisture sensors in California
        >>> stations = await discover_stations(elements=['SMS:*'], state_codes=['CA'])
        >>>
        >>> # Find stations by name pattern
        >>> stations = await discover_stations(station_names=['*River*'])
        >>>
        >>> # Get sensor metadata for found stations
        >>> stations = await discover_stations(
        ...     station_triplets=['*:OR:SNTL'],
        ...     include_sensor_metadata=True
        ... )
    """
    async with AWDBClient() as client:
        stations = await client.get_stations(
            network_codes=network_codes,
            state_codes=state_codes,
            station_triplets=station_triplets,
            station_names=station_names,
            elements=elements,
            active_only=active_only,
            return_station_elements=include_sensor_metadata,  # Enable sensor metadata
        )

        if limit:
            stations = stations[:limit]

        result = []
        for station in stations:
            station_dict = {
                "station_triplet": station.station_triplet,
                "name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation": station.elevation,
                "network_code": station.network_code,
                "state": station.state,
                "county": station.county,
                "station_id": station.station_id,
                "dco_code": station.dco_code,
                "huc": station.huc,
            }

            # Add sensor metadata if requested
            if include_sensor_metadata:
                if station.station_elements:
                    try:
                        # Convert raw station elements to organized sensor metadata
                        sensors_by_property: Dict[str, List[Dict[str, Any]]] = {}
                        for elem in station.station_elements:  # type: ignore
                            element_code = elem.get("elementCode", "")
                            property_name = None

                            # Map element code to property name
                            for prop_name, elem_code in PROPERTY_ELEMENT_MAP.items():
                                if elem_code == element_code:
                                    property_name = prop_name
                                    break

                            if not property_name:
                                property_name = f"unknown_{element_code}"

                            if property_name not in sensors_by_property:
                                sensors_by_property[property_name] = []

                            sensor_info = {
                                "element_code": element_code,
                                "ordinal": elem.get("ordinal", 1),
                                "height_depth_inches": elem.get("heightDepth"),
                                "begin_date": elem.get("beginDate"),
                                "end_date": elem.get("endDate"),
                                "data_precision": elem.get("dataPrecision"),
                                "stored_unit_code": elem.get("storedUnitCode"),
                                "original_unit_code": elem.get("originalUnitCode"),
                                "derived_data": elem.get("derivedData", False),
                            }

                            sensors_by_property[property_name].append(sensor_info)

                        station_dict["sensor_metadata"] = sensors_by_property
                    except Exception as e:
                        station_dict["sensor_metadata"] = {"error": str(e)}

            result.append(station_dict)

        return result


def build_soil_element_string(
    element_code: str, height_depth_inches: int, ordinal: int = 1
) -> str:
    """
    Build proper element string for soil properties.

    Args:
        element_code: Base element code (e.g., 'SMS', 'STO')
        height_depth_inches: Depth in inches (negative for below surface)
        ordinal: Sensor ordinal (1, 2, 3, etc. for multiple sensors at same depth)

    Returns:
        Formatted element string (e.g., 'SMS:-20:1')
    """
    return f"{element_code}:{height_depth_inches}:{ordinal}"


@add_sync_version
async def station_sensor_depths(station_triplet: str, property_name: str) -> List[Dict]:
    """
    Get available sensor depths/heights for a specific station and property.

    Use this to discover what depths (for soil properties) or heights (for atmospheric
    properties) are available at a given station before querying data.

    Args:
        station_triplet: Station identifier (network:state:code format)
        property_name: Property name (e.g., 'soil_moisture', 'air_temp')
                      See PROPERTY_ELEMENT_MAP keys for complete list

    Returns:
        List of sensor configurations with metadata (height/depth, ordinal, dates, precision)

    Raises:
        AWDBError: If property doesn't support height/depth specification

    Examples:
        >>> # Get available depths for soil moisture at a station
        >>> depths = await station_sensor_depths('SCAN:IA:2080', 'soil_moisture')
        >>> print(depths)  # [{'height_depth_inches': -20, 'ordinal': 1, ...}, ...]
        >>>
        >>> # Get available heights for temperature sensors
        >>> heights = await station_sensor_depths('SNTL:OR:1017', 'air_temp')
    """
    if property_name not in HEIGHT_DEPTH_PROPERTIES:
        raise AWDBError(
            f"Property '{property_name}' does not support height/depth specification"
        )

    element_code = PROPERTY_ELEMENT_MAP[property_name]

    async with AWDBClient() as client:
        stations = await client.get_stations(
            station_triplets=[station_triplet], return_station_elements=True
        )

        if not stations or not stations[0].station_elements:
            return []

        # Find all elements matching the base element code
        sensor_elements = [
            elem
            for elem in stations[0].station_elements
            if elem.get("elementCode") == element_code
        ]

        sensors = []
        for elem in sensor_elements:
            sensor_info = {
                "height_depth_inches": elem.get("heightDepth", 0),
                "ordinal": elem.get("ordinal", 1),
                "element_string": build_soil_element_string(
                    element_code, elem.get("heightDepth", 0), elem.get("ordinal", 1)
                ),
                "begin_date": elem.get("beginDate"),
                "end_date": elem.get("endDate"),
                "data_precision": elem.get("dataPrecision"),
            }
            sensors.append(sensor_info)

        # Sort by height/depth (most negative first for depths, then by height)
        sensors.sort(key=lambda x: x["height_depth_inches"])

        return sensors


@add_sync_version
async def get_station_soil_depths(
    station_triplet: str, property_name: str = "soil_moisture"
) -> List[Dict]:
    """
    [DEPRECATED] Use station_sensor_depths() instead.

    Get available soil depths for a specific station and property.

    Args:
        station_triplet: Station identifier
        property_name: Soil property ('soil_moisture' or 'soil_temp')

    Returns:
        List of depth configurations with metadata
    """
    warnings.warn(
        "get_station_soil_depths() is deprecated, use station_sensor_depths() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await station_sensor_depths(station_triplet, property_name)


@add_sync_version
async def get_soil_moisture_by_depth(
    station_triplet: str,
    depths_inches: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Get soil moisture data for multiple depths at a station.

    This function automatically queries station metadata to find available soil moisture
    sensors at the requested depths, then retrieves time-series data for each.

    Args:
        station_triplet: Station identifier (network:state:code format)
        depths_inches: List of depths to query in inches (negative values for below surface).
                      If None, queries all available depths at the station.
        start_date: Start date in YYYY-MM-DD format. Defaults to 30 days before today.
        end_date: End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dictionary with soil moisture data for each depth, including timestamps with
        timezone-aware information

    Examples:
        >>> # Get all available depths
        >>> result = await get_soil_moisture_by_depth('SCAN:IA:2080')
        >>>
        >>> # Get specific depths
        >>> result = await get_soil_moisture_by_depth(
        ...     'SCAN:IA:2080',
        ...     depths_inches=[-4, -8, -20],
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31'
        ... )
    """
    if not start_date or not end_date:
        # Default to recent data
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Get available depths
    available_depths = await station_sensor_depths(station_triplet, "soil_moisture")

    if not available_depths:
        raise AWDBError(f"No soil moisture sensors found for station {station_triplet}")

    # Filter to requested depths or use all available
    if depths_inches:
        requested_depths = []
        for depth in depths_inches:
            # Find matching available depth
            matches = [d for d in available_depths if d["height_depth_inches"] == depth]
            if matches:
                requested_depths.extend(matches)
        target_depths = requested_depths
    else:
        target_depths = available_depths

    if not target_depths:
        raise AWDBError(
            f"None of the requested depths are available at station {station_triplet}"
        )

    async with AWDBClient() as client:
        from datetime import datetime

        result: Dict[str, Any] = {
            "station_triplet": station_triplet,
            "depths": {},
            "metadata": {
                "query_date": datetime.now().isoformat(),
                "date_range": {"start": start_date, "end": end_date},
            },
        }

        # Query each depth
        for depth_info in target_depths:
            element_string = depth_info["element_string"]
            depth_inches = depth_info["height_depth_inches"]

            try:
                data_points = await client.get_station_data(
                    station_triplet, element_string, start_date, end_date
                )

                result["depths"][depth_inches] = {
                    "element_string": element_string,
                    "data_points": [
                        {
                            "timestamp": pt.timestamp.isoformat(),
                            "value": pt.value,
                            "flags": pt.flags,
                            "qc_flag": pt.qc_flag,
                            "qa_flag": pt.qa_flag,
                            "orig_value": pt.orig_value,
                            "average": pt.average,
                            "median": pt.median,
                            "station_timezone_offset_hours": pt.station_timezone_offset,
                        }
                        for pt in data_points
                    ],
                    "n_data_points": len(data_points),
                }

            except Exception as e:
                result["depths"][depth_inches] = {
                    "element_string": element_string,
                    "error": str(e),
                    "data_points": [],
                    "n_data_points": 0,
                }

        return result


@add_sync_version
async def get_property_data_near(
    latitude: float,
    longitude: float,
    property_name: str,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
    height_depth_inches: Optional[int] = None,
    network_codes: Optional[List[str]] = None,
    auto_select_sensor: bool = True,
) -> Dict:
    """
    Get time-series data for a property from the nearest monitoring station.

    This is the primary function for retrieving measurement data near a geographic location.
    It discovers nearby stations and automatically selects an appropriate sensor for the
    requested property.

    For properties that support height/depth specification (e.g., soil_moisture, soil_temp,
    air_temp), you can optionally specify the exact depth/height:
    - Negative values for depth below surface (soil properties)
    - Positive values for height above surface (atmospheric properties)

    Args:
        latitude: Target latitude (WGS84)
        longitude: Target longitude (WGS84)
        property_name: Property/variable name (e.g., 'soil_moisture', 'air_temp', 'precipitation')
                      See PROPERTY_ELEMENT_MAP keys for complete list
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_distance_km: Maximum distance to search for stations (default: 50 km)
        height_depth_inches: Optional height/depth specification for compatible properties.
                            Leave None to auto-select the primary sensor.
        network_codes: Network codes to filter by (e.g., ['SCAN', 'SNTL'])
        auto_select_sensor: Automatically select best available sensor (recommended: True)

    Returns:
        Dictionary with time-series data, including:
        - data_points: List of measurements with timestamps (timezone-aware)
        - metadata: Station info, distance, element string, etc.
        - unit: Measurement unit for the property

    Raises:
        AWDBError: If property is invalid, no stations found, or data unavailable

    Examples:
        >>> # Get hourly air temperature from nearest station
        >>> data = await get_property_data_near(
        ...     42.0, -93.6, 'air_temp', '2024-01-01', '2024-12-31'
        ... )
        >>>
        >>> # Get soil moisture at specific depth
        >>> data = await get_property_data_near(
        ...     42.0, -93.6, 'soil_moisture', '2024-01-01', '2024-12-31',
        ...     height_depth_inches=-20
        ... )
        >>>
        >>> # Find data from SNOTEL stations only
        >>> data = await get_property_data_near(
        ...     42.0, -93.6, 'snow_water_equivalent', '2024-10-01', '2024-12-31',
        ...     network_codes=['SNTL']
        ... )
    """
    # Validate inputs
    if property_name not in PROPERTY_ELEMENT_MAP:
        available_props = list(PROPERTY_ELEMENT_MAP.keys())
        raise AWDBError(
            f"Unsupported property '{property_name}'. Available: {available_props}"
        )

    # Validate required height/depth for soil properties (unless auto-select is enabled)
    if (
        property_name in REQUIRED_DEPTH_PROPERTIES
        and height_depth_inches is None
        and not auto_select_sensor
    ):
        raise AWDBError(
            f"Soil property '{property_name}' requires height_depth_inches parameter "
            "(negative value for depth below surface, e.g., -20 for 20 inches deep) "
            "or set auto_select_sensor=True to automatically select an available sensor"
        )

    # Validate dates
    try:
        from datetime import datetime

        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError as e:
        raise AWDBError(f"Invalid date format: {e}") from e

    async with AWDBClient() as client:
        # Find nearby stations
        nearby_stations = await client.find_nearby_stations(
            latitude, longitude, max_distance_km, network_codes=network_codes
        )

        if not nearby_stations:
            raise AWDBError(f"No monitoring stations found within {max_distance_km} km")

        # Try to get data from the nearest station
        nearest_station, distance = nearby_stations[0]
        element_code = PROPERTY_ELEMENT_MAP[property_name]

        # Auto-select best sensor if enabled and height/depth specification is needed
        if auto_select_sensor and property_name in HEIGHT_DEPTH_PROPERTIES:
            if height_depth_inches is not None:
                # User specified specific height/depth - use it
                element_string = build_soil_element_string(
                    element_code, height_depth_inches, ordinal=1
                )
                ordinal = 1
            else:
                # Auto-select: query station metadata to find available sensors
                try:
                    available_sensors = await station_sensor_depths(
                        nearest_station.station_triplet, property_name
                    )
                    if available_sensors:
                        # Select the first available sensor (typically the primary one)
                        best_sensor = available_sensors[0]
                        element_string = best_sensor["element_string"]
                        ordinal = best_sensor["ordinal"]
                        height_depth_inches = best_sensor["height_depth_inches"]
                    else:
                        # Fallback to default if no sensors found
                        element_string = element_code
                        ordinal = 1
                except Exception:
                    # Fallback on metadata query failure
                    element_string = element_code
                    ordinal = 1
        else:
            # Manual sensor selection or no height/depth needed
            if (
                property_name in HEIGHT_DEPTH_PROPERTIES
                and height_depth_inches is not None
            ):
                element_string = build_soil_element_string(
                    element_code, height_depth_inches, ordinal=1
                )
                ordinal = 1
            else:
                element_string = element_code
                ordinal = 1

        # Fetch data
        raw_data = await client.get_station_data(
            nearest_station.station_triplet,
            element_string,
            start_date,
            end_date,
        )

        # Convert to TimeSeriesDataPoint objects
        data_points = []
        for point in raw_data:
            if point.value is not None:  # Skip null values
                data_points.append(point)

        # Sort by timestamp
        data_points.sort(key=lambda x: x.timestamp)

        # Create result
        result = {
            "site_id": nearest_station.station_triplet,
            "site_name": nearest_station.name,
            "latitude": nearest_station.latitude,
            "longitude": nearest_station.longitude,
            "property_name": property_name,
            "data_points": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "flags": point.flags,
                    "qc_flag": point.qc_flag,
                    "qa_flag": point.qa_flag,
                    "orig_value": point.orig_value,
                    "average": point.average,
                    "median": point.median,
                    "station_timezone_offset_hours": point.station_timezone_offset,
                }
                for point in data_points
            ],
            "unit": await get_property_unit_from_api(client, element_code)
            or PROPERTY_UNITS.get(property_name, ""),
            "metadata": {
                "distance_km": round(distance, 2),
                "network": nearest_station.network_code,
                "elevation": nearest_station.elevation,
                "height_depth_inches": height_depth_inches,
                "element_string": element_string,
                "ordinal": ordinal,
                "n_data_points": len(data_points),
                "station_timezone_offset_hours": nearest_station.data_time_zone,
                "query_date": datetime.now().isoformat(),
                "date_range": {"start": start_date, "end": end_date},
            },
        }

        return result


@add_sync_version
async def get_property_unit_from_api(client: AWDBClient, element_code: str) -> str:
    """
    Get the unit for an element from the AWDB API reference data.

    Args:
        client: AWDBClient instance
        element_code: Element code (e.g., 'TOBS', 'SMS')

    Returns:
        Unit string (e.g., 'degF', 'pct')
    """
    try:
        ref_data = await client.get_reference_data(["elements"])
        if ref_data.elements:
            for elem in ref_data.elements:
                if elem["code"] == element_code:
                    unit = elem.get("englishUnitCode", elem.get("storedUnitCode", ""))
                    return str(unit) if unit else ""
        return ""  # Not found
    except Exception:
        # Fallback to hardcoded units if API fails
        return ""


@add_sync_version
async def station_sensors(station_triplet: str) -> Dict[str, Any]:
    """
    Get comprehensive sensor metadata for a station.

    Returns complete information about all sensors at a station, including element codes,
    available depths/heights, date ranges, precision, and units.

    Args:
        station_triplet: Station identifier (network:state:code format)

    Returns:
        Dictionary with sensor metadata organized by property name:
        {
            'station_triplet': '...',
            'station_name': '...',
            'network': '...',
            'sensors': {
                'soil_moisture': [
                    {
                        'element_code': 'SMS',
                        'ordinal': 1,
                        'height_depth_inches': -20,
                        'begin_date': '...',
                        'end_date': '...',
                        ...
                    }
                ],
                ...
            }
        }

    Examples:
        >>> metadata = await station_sensors('SCAN:IA:2080')
        >>> # List available properties
        >>> for prop in metadata['sensors'].keys():
        ...     print(prop)
        >>>
        >>> # Check soil moisture depths
        >>> for sensor in metadata['sensors'].get('soil_moisture', []):
        ...     print(f"Depth: {sensor['height_depth_inches']} inches")
    """
    async with AWDBClient() as client:
        # Get station with element details
        stations = await client.get_stations(
            station_triplets=[station_triplet], return_station_elements=True
        )

        if not stations or not stations[0].station_elements:
            return {"station_triplet": station_triplet, "sensors": {}}

        station = stations[0]
        sensors_by_property: Dict[str, List[Dict[str, Any]]] = {}

        for elem in station.station_elements:  # type: ignore
            element_code = elem.get("elementCode", "")
            property_name = None

            # Map element code to property name
            for prop_name, elem_code in PROPERTY_ELEMENT_MAP.items():
                if elem_code == element_code:
                    property_name = prop_name
                    break

            if not property_name:
                property_name = f"unknown_{element_code}"

            if property_name not in sensors_by_property:
                sensors_by_property[property_name] = []

            sensor_info = {
                "element_code": element_code,
                "ordinal": elem.get("ordinal", 1),
                "height_depth_inches": elem.get("heightDepth"),
                "begin_date": elem.get("beginDate"),
                "end_date": elem.get("endDate"),
                "data_precision": elem.get("dataPrecision"),
                "stored_unit_code": elem.get("storedUnitCode"),
                "original_unit_code": elem.get("originalUnitCode"),
                "derived_data": elem.get("derivedData", False),
            }

            sensors_by_property[property_name].append(sensor_info)

        return {
            "station_triplet": station_triplet,
            "station_name": station.name,
            "network": station.network_code,
            "sensors": sensors_by_property,
        }


@add_sync_version
async def station_available_properties(station_triplet: str) -> List[Dict]:
    """
    List available measured properties/variables for a specific station.

    Use this to discover what measurements are available before querying data.
    Returns both human-readable property names and underlying AWDB element codes.

    Args:
        station_triplet: Station identifier (network:state:code format)

    Returns:
        List of available properties with metadata:
        [
            {
                'property_name': 'soil_moisture',
                'element_code': 'SMS',
                'unit': 'pct',
                'description': 'Soil Moisture',
                'sensors': [
                    {
                        'element_code': 'SMS',
                        'ordinal': 1,
                        'height_depth_inches': -20,
                        ...
                    }
                ]
            },
            ...
        ]

    Examples:
        >>> props = await station_available_properties('SCAN:IA:2080')
        >>> for prop in props:
        ...     print(f"{prop['property_name']}: {prop['element_code']}")
        >>>
        >>> # Filter to only soil properties
        >>> soil_props = [p for p in props if 'soil' in p['property_name']]
    """
    metadata = await station_sensors(station_triplet)

    variables = []
    sensors_dict = metadata.get("sensors", {})
    if sensors_dict:
        for property_name in sensors_dict.keys():
            sensors_list = sensors_dict[property_name]
            if isinstance(sensors_list, list):
                sensors = sensors_list
                if property_name.startswith("unknown_"):
                    # Unknown element codes
                    element_code = property_name.replace("unknown_", "")
                    variables.append(
                        {
                            "property_name": property_name,
                            "element_code": element_code,
                            "unit": "",
                            "description": f"Unknown element {element_code}",
                            "sensors": sensors,  # type: ignore
                        }
                    )
                else:
                    # Known properties
                    element_code = PROPERTY_ELEMENT_MAP.get(property_name, "")
                    api_unit = await get_property_unit_from_api(
                        AWDBClient(), element_code
                    )
                    unit = (
                        str(api_unit)
                        if api_unit
                        else PROPERTY_UNITS.get(property_name, "")
                    )
                    variables.append(
                        {
                            "property_name": property_name,
                            "element_code": element_code,
                            "unit": unit,
                            "description": f"{property_name.replace('_', ' ').title()}",
                            "sensors": sensors,  # type: ignore
                        }
                    )

    return variables


# ============================================================================
# DEPRECATED ALIASES (for backward compatibility)
# ============================================================================
# These functions are deprecated and will be removed in a future version.
# Use the new names instead. See migration guide in module docstring.


@add_sync_version
async def get_nearby_stations(
    latitude: float,
    longitude: float,
    max_distance_km: float = 50.0,
    network_codes: Optional[List[str]] = None,
    limit: int = 10,
    include_sensor_metadata: bool = False,
) -> List[Dict]:
    """
    [DEPRECATED] Use discover_stations_nearby() instead.

    Find AWDB stations near a location.
    """
    warnings.warn(
        "get_nearby_stations() is deprecated, use discover_stations_nearby() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await discover_stations_nearby(
        latitude=latitude,
        longitude=longitude,
        max_distance_km=max_distance_km,
        network_codes=network_codes,
        limit=limit,
        include_sensor_metadata=include_sensor_metadata,
    )


@add_sync_version
async def find_stations_by_criteria(
    network_codes: Optional[List[str]] = None,
    state_codes: Optional[List[str]] = None,
    station_triplets: Optional[List[str]] = None,
    station_names: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    active_only: bool = True,
    limit: Optional[int] = None,
    include_sensor_metadata: bool = False,
) -> List[Dict]:
    """
    [DEPRECATED] Use discover_stations() instead.

    Find stations using advanced filtering criteria with wildcard support.
    """
    warnings.warn(
        "find_stations_by_criteria() is deprecated, use discover_stations() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await discover_stations(
        network_codes=network_codes,
        state_codes=state_codes,
        station_triplets=station_triplets,
        station_names=station_names,
        elements=elements,
        active_only=active_only,
        limit=limit,
        include_sensor_metadata=include_sensor_metadata,
    )


@add_sync_version
async def get_station_sensor_heights(
    station_triplet: str, property_name: str
) -> List[Dict]:
    """
    [DEPRECATED] Use station_sensor_depths() instead.

    Get available sensor heights/depths for a specific station and property.
    """
    warnings.warn(
        "get_station_sensor_heights() is deprecated, use station_sensor_depths() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await station_sensor_depths(station_triplet, property_name)


@add_sync_version
async def get_soil_moisture_data(
    station_triplet: str,
    depths_inches: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    [DEPRECATED] Use get_soil_moisture_by_depth() instead.

    Get soil moisture data for multiple depths at a station.
    """
    warnings.warn(
        "get_soil_moisture_data() is deprecated, use get_soil_moisture_by_depth() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await get_soil_moisture_by_depth(
        station_triplet=station_triplet,
        depths_inches=depths_inches,
        start_date=start_date,
        end_date=end_date,
    )


@add_sync_version
async def get_monitoring_station_data(
    latitude: float,
    longitude: float,
    property_name: str,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
    height_depth_inches: Optional[int] = None,
    network_codes: Optional[List[str]] = None,
    auto_select_sensor: bool = True,
) -> Dict:
    """
    [DEPRECATED] Use get_property_data_near() instead.

    Get time-series data for a property from nearby stations.
    """
    warnings.warn(
        "get_monitoring_station_data() is deprecated, use get_property_data_near() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await get_property_data_near(
        latitude=latitude,
        longitude=longitude,
        property_name=property_name,
        start_date=start_date,
        end_date=end_date,
        max_distance_km=max_distance_km,
        height_depth_inches=height_depth_inches,
        network_codes=network_codes,
        auto_select_sensor=auto_select_sensor,
    )


@add_sync_version
async def get_station_sensor_metadata(station_triplet: str) -> Dict[str, Any]:
    """
    [DEPRECATED] Use station_sensors() instead.

    Get comprehensive sensor metadata for a station.
    """
    warnings.warn(
        "get_station_sensor_metadata() is deprecated, use station_sensors() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await station_sensors(station_triplet)


@add_sync_version
async def list_available_variables(station_triplet: str) -> List[Dict]:
    """
    [DEPRECATED] Use station_available_properties() instead.

    List available variables/measured elements for a specific station.
    """
    warnings.warn(
        "list_available_variables() is deprecated, use station_available_properties() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return await station_available_properties(station_triplet)
