"""
Data models for AWDB (Air and Water Database) SCAN/SNOTEL data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class StationInfo:
    """Information about a SCAN/SNOTEL station."""

    station_triplet: str
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float]
    network_code: str  # 'SCAN' or 'SNOTEL'
    state: Optional[str]
    county: Optional[str]


@dataclass
class TimeSeriesDataPoint:
    """A single data point in a time series."""

    timestamp: datetime
    value: Optional[float]
    flags: List[str] = field(default_factory=list)


@dataclass
class StationTimeSeries:
    """Time series data from a station."""

    station: StationInfo
    property_name: str
    data_points: List[TimeSeriesDataPoint]
    unit: str
    depth_cm: Optional[int] = None
