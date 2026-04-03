"""
Data models for AWDB (Air and Water Database)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class StationInfo:
    """Information about an AWDB station."""

    station_triplet: str
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float]
    network_code: str
    state: Optional[str]
    county: Optional[str]

    # Additional metadata fields from API
    station_id: Optional[str] = None
    dco_code: Optional[str] = None
    huc: Optional[str] = None
    data_time_zone: Optional[int] = None
    pedon_code: Optional[str] = None
    shef_id: Optional[str] = None
    operator: Optional[str] = None
    begin_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_point: Optional[dict[str, Any]] = None
    reservoir_metadata: Optional[dict[str, Any]] = None
    station_elements: Optional[list[dict[str, Any]]] = None  # type: ignore


@dataclass
class TimeSeriesDataPoint:
    """A single data point in a time series."""

    timestamp: datetime
    value: Optional[float]
    flags: list[str] = field(default_factory=list)

    # Element/variable identification
    element_code: Optional[str] = None  # e.g., 'SMS:-20:1', 'TOBS:0:1'
    variable_name: Optional[str] = None  # e.g., 'soil_moisture', 'air_temp'

    # Additional fields from API
    qc_flag: Optional[str] = None
    qa_flag: Optional[str] = None
    orig_value: Optional[float] = None
    orig_qc_flag: Optional[str] = None
    average: Optional[float] = None
    median: Optional[float] = None
    month: Optional[int] = None
    month_part: Optional[str] = None
    year: Optional[int] = None
    collection_date: Optional[str] = None

    # Station timezone information (for hourly data)
    station_timezone_offset: Optional[int] = (
        None  # Hours offset from UTC (e.g., -8 for PST)
    )


@dataclass
class StationTimeSeries:
    """Time series data from a station."""

    station: StationInfo
    property_name: str
    data_points: list[TimeSeriesDataPoint]
    unit: str
    depth_cm: Optional[int] = None


@dataclass
class ForecastData:
    """Forecast data for a station."""

    station_triplet: str
    forecast_point_name: Optional[str] = None
    data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReferenceData:
    """Reference data from AWDB."""

    dcos: Optional[list[dict[str, Any]]] = None
    durations: Optional[list[dict[str, Any]]] = None
    elements: Optional[list[dict[str, Any]]] = None
    forecast_periods: Optional[list[dict[str, Any]]] = None
    functions: Optional[list[dict[str, Any]]] = None
    instruments: Optional[list[dict[str, Any]]] = None
    networks: Optional[list[dict[str, Any]]] = None
    physical_elements: Optional[list[dict[str, Any]]] = None
    states: Optional[list[dict[str, Any]]] = None
    units: Optional[list[dict[str, Any]]] = None
