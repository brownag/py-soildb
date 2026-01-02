"""
Data models for Henry Mount Soil Climate Database.

These dataclasses represent the primary data structures returned by the Henry API
and database queries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class HenryStation:
    """Station metadata from the Henry Mount Soil Climate Database."""

    station_id: str
    """User site ID (usersiteid) - primary identifier."""

    station_name: str
    """Human-readable station name."""

    project_code: str
    """Project identifier code."""

    latitude: float
    """Geographic latitude (WGS84, EPSG:4326)."""

    longitude: float
    """Geographic longitude (WGS84, EPSG:4326)."""

    elevation_m: Optional[int] = None
    """Elevation in meters."""

    state: Optional[str] = None
    """State abbreviation (e.g., 'CA')."""

    county: Optional[str] = None
    """County name."""

    sso_code: Optional[str] = None
    """Soil Survey Office code (e.g., '2-SON')."""

    huc: Optional[str] = None
    """Hydrologic Unit Code."""

    installation_date: Optional[datetime] = None
    """Date station was installed."""

    removal_date: Optional[datetime] = None
    """Date station was removed (None if still active)."""

    installation_type: Optional[str] = None
    """Installation type (e.g., 'permanent', 'portable', 'temporary')."""

    project_description: Optional[str] = None
    """Description of the project."""


@dataclass
class HenrySensor:
    """Sensor/variable metadata for a Henry station."""

    station_id: str
    """Station identifier (foreign key to HenryStation)."""

    variable_name: str
    """Variable type (e.g., 'soiltemp', 'soilVWC', 'airtemp', 'waterlevel')."""

    element_code: Optional[str] = None
    """Standardized element code in AWDB format (e.g., 'STO:-4:1')."""

    sensor_description: Optional[str] = None
    """Original Henry sensor description."""

    depth_cm: Optional[float] = None
    """Sensor depth in centimeters (below surface for soil sensors)."""

    ordinal: int = 1
    """Ordinal number for multiple sensors of same type/depth."""


@dataclass
class HenryTimeSeriesDataPoint:
    """Single data point in a Henry time series."""

    station_id: str
    """Station identifier."""

    element_code: str
    """Element code identifier (e.g., 'STO:-4:1')."""

    timestamp: datetime
    """Measurement timestamp (ISO8601 format, UTC)."""

    value: Optional[float] = None
    """Measured value (None if missing/null)."""

    duration: str = "DAILY"
    """Data granularity ('DAILY' or 'HOURLY')."""

    qc_flag: Optional[str] = None
    """Quality control flag from Henry API (if present)."""

    qa_flag: Optional[str] = None
    """Quality assurance flag from Henry API (if present)."""

    orig_value: Optional[float] = None
    """Original value before any QC processing."""


@dataclass
class HenryDataCoverage:
    """Coverage metrics for a station/variable combination."""

    station_id: str
    """Station identifier."""

    variable_name: str
    """Variable type."""

    element_code: Optional[str] = None
    """Element code."""

    duration: str = "DAILY"
    """Data granularity ('DAILY' or 'HOURLY')."""

    date_start: Optional[datetime] = None
    """Start date of Period of Record."""

    date_end: Optional[datetime] = None
    """End date of Period of Record."""

    coverage_days: Optional[int] = None
    """Number of calendar days covered."""

    total_records: int = 0
    """Actual count of data points in POR."""

    expected_records: Optional[int] = None
    """Expected data points based on daily frequency."""

    completeness_percent: float = 0.0
    """Completeness percentage (total_records / expected_records * 100)."""

    gap_count: int = 0
    """Number of gaps larger than 7 days in POR."""

    largest_gap_days: Optional[int] = None
    """Size of largest gap in days."""

    mean_gap_days: Optional[float] = None
    """Average gap size in days."""

    por_fully_processed: bool = False
    """Flag indicating if full POR has been analyzed."""


@dataclass
class HenryStationStatus:
    """Current operational status and quality metrics for a station."""

    station_id: str
    """Station identifier."""

    overall_status: str
    """Status: 'active', 'inactive', 'partial', 'removed', or 'unknown'."""

    status_reason: Optional[str] = None
    """Reason for current status if not active."""

    total_sensors: int = 0
    """Total number of sensors at this station."""

    active_sensors: int = 0
    """Number of sensors with recent data (last 30 days)."""

    inactive_sensors: int = 0
    """Number of sensors without recent data."""

    days_since_last_data: Optional[int] = None
    """Days since most recent data point."""

    most_recent_data_date: Optional[datetime] = None
    """Date of most recent data point."""

    data_quality_score: float = 0.0
    """Overall quality score (0-100): 30% recency + 50% completeness + 20% consistency."""

    recency_score: Optional[float] = None
    """Recency component of quality score (0-100)."""

    completeness_score: Optional[float] = None
    """Completeness component of quality score (0-100)."""

    consistency_score: Optional[float] = None
    """Consistency component of quality score (0-100)."""

    last_health_check: Optional[datetime] = None
    """Timestamp of last status calculation."""


__all__ = [
    "HenryStation",
    "HenrySensor",
    "HenryTimeSeriesDataPoint",
    "HenryDataCoverage",
    "HenryStationStatus",
]
