"""
High-level convenience functions for AWDB data access.
"""

from typing import Dict, List, Optional

from .client import AWDBClient
from .exceptions import AWDBError
from .models import TimeSeriesDataPoint

# Mapping from property names to AWDB element codes
PROPERTY_ELEMENT_MAP = {
    "soil_moisture": "SMS",  # Soil Moisture Percent
    "soil_temp": "STO",  # Soil Temperature
    "precipitation": "PREC",  # Precipitation
    "air_temp": "TOBS",  # Air Temperature Observed
    "snow_water_equivalent": "WTEQ",  # Snow Water Equivalent
    "wind_speed": "WSPD",  # Wind Speed
    "wind_direction": "WDIR",  # Wind Direction
    "relative_humidity": "RHUM",  # Relative Humidity
    "solar_radiation": "SRAD",  # Solar Radiation
}

# Units for each property
PROPERTY_UNITS = {
    "soil_moisture": "volumetric %",
    "soil_temp": "deg C",
    "precipitation": "mm",
    "air_temp": "deg C",
    "snow_water_equivalent": "mm",
    "wind_speed": "m/s",
    "wind_direction": "degrees",
    "relative_humidity": "%",
    "solar_radiation": "W/m²",
}


def get_nearby_stations(
    latitude: float,
    longitude: float,
    max_distance_km: float = 50.0,
    network_codes: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Find SCAN/SNOTEL stations near a location.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        max_distance_km: Maximum search distance
        network_codes: Network codes to include ('SCAN', 'SNOTEL')
        limit: Maximum number of stations to return

    Returns:
        List of station dictionaries with distance information
    """
    with AWDBClient() as client:
        stations_with_distance = client.find_nearby_stations(
            latitude, longitude, max_distance_km, network_codes, limit
        )

        return [
            {
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
            for station, distance in stations_with_distance
        ]


def get_monitoring_station_data(
    latitude: float,
    longitude: float,
    property_name: str,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
    depth_cm: Optional[int] = None,
) -> Dict:
    """
    Get time-series data for a dynamic soil property from nearby stations.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        property_name: Property name ('soil_moisture', 'soil_temp', etc.)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_distance_km: Maximum distance to search for stations
        depth_cm: Soil depth in cm (for soil properties)

    Returns:
        Dictionary with time series data and metadata
    """
    # Validate inputs
    if property_name not in PROPERTY_ELEMENT_MAP:
        available_props = list(PROPERTY_ELEMENT_MAP.keys())
        raise AWDBError(
            f"Unsupported property '{property_name}'. Available: {available_props}"
        )

    # Validate dates
    try:
        from datetime import datetime

        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError as e:
        raise AWDBError(f"Invalid date format: {e}") from e

    with AWDBClient() as client:
        # Find nearby stations
        nearby_stations = client.find_nearby_stations(
            latitude, longitude, max_distance_km, network_codes=["SCAN", "SNOTEL"]
        )

        if not nearby_stations:
            raise AWDBError(f"No monitoring stations found within {max_distance_km} km")

        # Try to get data from the nearest station
        nearest_station, distance = nearby_stations[0]
        element_code = PROPERTY_ELEMENT_MAP[property_name]

        # Determine ordinal based on depth (for soil properties)
        ordinal = 1
        if depth_cm is not None and property_name in ["soil_moisture", "soil_temp"]:
            # SCAN typically has sensors at different depths
            # Ordinal mapping is approximate and may need adjustment
            if depth_cm <= 5:
                ordinal = 1  # 5cm depth
            elif depth_cm <= 10:
                ordinal = 2  # 10cm depth
            elif depth_cm <= 20:
                ordinal = 3  # 20cm depth
            elif depth_cm <= 50:
                ordinal = 4  # 50cm depth
            else:
                ordinal = 5  # 100cm depth

        # Fetch data
        raw_data = client.get_station_data(
            nearest_station.station_triplet,
            element_code,
            start_date,
            end_date,
            ordinal=ordinal,
        )

        # Convert to TimeSeriesDataPoint objects
        data_points = []
        for point in raw_data:
            if point["value"] is not None:  # Skip null values
                data_point = TimeSeriesDataPoint(
                    timestamp=point["timestamp"],
                    value=float(point["value"]),
                    flags=point.get("flags", []),
                )
                data_points.append(data_point)

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
                }
                for point in data_points
            ],
            "unit": PROPERTY_UNITS.get(property_name, ""),
            "metadata": {
                "distance_km": round(distance, 2),
                "network": nearest_station.network_code,
                "elevation": nearest_station.elevation,
                "depth_cm": depth_cm,
                "ordinal": ordinal,
                "n_data_points": len(data_points),
                "query_date": datetime.now().isoformat(),
                "date_range": {"start": start_date, "end": end_date},
            },
        }

        return result


def list_available_variables(station_triplet: str) -> List[Dict]:
    """
    List available variables/measured elements for a specific station.

    Note: This is a placeholder for future implementation.
    The AWDB API doesn't currently provide a direct endpoint for this.

    Args:
        station_triplet: Station identifier

    Returns:
        List of available variables (currently returns common SCAN/SNOTEL variables)
    """
    # For now, return the standard set of variables
    # In the future, this could query the API for station-specific variables
    variables = []
    for prop_name, element_code in PROPERTY_ELEMENT_MAP.items():
        variables.append(
            {
                "property_name": prop_name,
                "element_code": element_code,
                "unit": PROPERTY_UNITS.get(prop_name, ""),
                "description": f"{prop_name.replace('_', ' ').title()}",
            }
        )

    return variables
