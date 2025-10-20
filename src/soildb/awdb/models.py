"""
Data models for AWDB (Air and Water Database)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


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
    forecast_point: Optional[Dict[str, Any]] = None
    reservoir_metadata: Optional[Dict[str, Any]] = None
    station_elements: Optional[List[Dict[str, Any]]] = None


@dataclass
class TimeSeriesDataPoint:
    """A single data point in a time series."""

    timestamp: datetime
    value: Optional[float]
    flags: List[str] = field(default_factory=list)

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


@dataclass
class StationTimeSeries:
    """Time series data from a station."""

    station: StationInfo
    property_name: str
    data_points: List[TimeSeriesDataPoint]
    unit: str
    depth_cm: Optional[int] = None


@dataclass
class ForecastData:
    """Forecast data for a station."""

    station_triplet: str
    forecast_point_name: Optional[str] = None
    data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReferenceData:
    """Reference data from AWDB."""

    dcos: Optional[List[Dict[str, Any]]] = None
    durations: Optional[List[Dict[str, Any]]] = None
    elements: Optional[List[Dict[str, Any]]] = None
    forecast_periods: Optional[List[Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    instruments: Optional[List[Dict[str, Any]]] = None
    networks: Optional[List[Dict[str, Any]]] = None
    physical_elements: Optional[List[Dict[str, Any]]] = None
    states: Optional[List[Dict[str, Any]]] = None
    units: Optional[List[Dict[str, Any]]] = None
