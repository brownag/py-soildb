"""
Tests for Henry Mount Soil Climate Database module.

Tests the HenryClient, data models, utilities, and convenience functions
with mocked API responses.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from soildb.henry.client import HenryClient
from soildb.henry.exceptions import HenryAPIError, HenryNetworkError
from soildb.henry.models import HenryStation, HenryTimeSeriesDataPoint
from soildb.henry.utils import (
    cm_to_inches,
    construct_element_code,
    henry_variable_to_base_code,
    inches_to_cm,
    parse_element_code,
    parse_henry_timestamp,
)


class TestHenryUtilities:
    """Test Henry utility functions."""

    def test_cm_to_inches(self):
        """Test centimeter to inch conversion."""
        assert cm_to_inches(2.54) == 1.0
        assert cm_to_inches(5.08) == 2.0
        assert cm_to_inches(-10.16) == -4.0

    def test_inches_to_cm(self):
        """Test inch to centimeter conversion."""
        assert inches_to_cm(1.0) == 2.54
        assert inches_to_cm(2.0) == 5.08
        assert inches_to_cm(-4.0) == -10.16

    def test_construct_element_code(self):
        """Test element code construction."""
        # With depth
        assert construct_element_code("SMS", depth_cm=5.08, ordinal=1) == "SMS:-2:1"
        assert construct_element_code("STO", depth_cm=10.16, ordinal=1) == "STO:-4:1"

        # Without depth
        assert construct_element_code("PREC", ordinal=1) == "PREC::1"

        # Different ordinals
        assert construct_element_code("SMS", depth_cm=5.08, ordinal=2) == "SMS:-2:2"

    def test_parse_element_code(self):
        """Test element code parsing."""
        result = parse_element_code("SMS:-2:1")
        assert result["base_code"] == "SMS"
        assert result["depth_inches"] == -2
        assert result["ordinal"] == 1

        result = parse_element_code("PREC::1")
        assert result["base_code"] == "PREC"
        assert result["depth_inches"] is None
        assert result["ordinal"] == 1

    def test_henry_variable_to_base_code(self):
        """Test Henry variable to element code mapping."""
        assert henry_variable_to_base_code("soiltemp") == "STO"
        assert henry_variable_to_base_code("soilVWC") == "SMS"
        assert henry_variable_to_base_code("airtemp") == "TOBS"
        assert henry_variable_to_base_code("waterlevel") == "WL"

        # Case insensitive
        assert henry_variable_to_base_code("SOILTEMP") == "STO"
        assert henry_variable_to_base_code("SoilTemp") == "STO"

        # Unknown variable
        assert henry_variable_to_base_code("unknown") == "UNKNOWN"

    def test_parse_henry_timestamp(self):
        """Test Henry timestamp parsing."""
        # Space-separated format (Henry standard)
        result = parse_henry_timestamp("2024-01-15 14:30:00")
        assert result == "2024-01-15T14:30:00"

        # Already ISO format
        result = parse_henry_timestamp("2024-01-15T14:30:00")
        assert result == "2024-01-15T14:30:00"


class TestHenryClient:
    """Test HenryClient functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return HenryClient(timeout=5)

    def test_init(self, client):
        """Test client initialization."""
        assert client.timeout == 5
        assert client.BASE_URL == "http://soilmap2-1.lawr.ucdavis.edu/henry"

    def test_init_with_config(self):
        """Test client initialization with ClientConfig."""
        from soildb.base_client import ClientConfig

        config = ClientConfig.reliable()
        client = HenryClient(config=config)
        assert client.timeout == 120.0

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_stations_success(self, mock_client_class, client):
        """Test successful station retrieval."""
        # Mock gzipped response
        import gzip
        import json

        mock_response_data = {
            "sensors": [
                {
                    "sid": "CA_SITE_001",
                    "name": "Test Station",
                    "project": "CA_PROJECT",
                    "sso": "2-SON",
                    "wgs84_latitude": 38.5,
                    "wgs84_longitude": -120.5,
                    "elevation_m": 1500,
                    "state": "CA",
                    "county": "Kern",
                }
            ]
        }

        # Compress response
        compressed = gzip.compress(json.dumps(mock_response_data).encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test
        stations = await client.get_stations(sso_code="2-SON")

        assert len(stations) == 1
        station = stations[0]
        assert station.station_id == "CA_SITE_001"
        assert station.station_name == "Test Station"
        assert station.project_code == "CA_PROJECT"
        assert station.latitude == 38.5
        assert station.longitude == -120.5
        assert station.elevation_m == 1500
        assert station.state == "CA"
        assert station.county == "Kern"

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_stations_no_filters(self, mock_client_class, client):
        """Test that get_stations requires at least one filter."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        client._client = mock_client

        with pytest.raises(HenryAPIError):
            await client.get_stations()

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_station_data_success(self, mock_client_class, client):
        """Test successful time series data retrieval."""
        import gzip
        import json

        mock_response_data = {
            "soiltemp": [
                {
                    "sid": "CA_SITE_001",
                    "date_time": "2024-01-15 12:00:00",
                    "sensor_value": 15.3,
                },
                {
                    "sid": "CA_SITE_001",
                    "date_time": "2024-01-16 12:00:00",
                    "sensor_value": 16.1,
                },
            ]
        }

        compressed = gzip.compress(json.dumps(mock_response_data).encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test
        data = await client.get_station_data(
            station_id="CA_SITE_001",
            variable_name="soiltemp",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert len(data) == 2
        assert data[0].station_id == "CA_SITE_001"
        assert data[0].value == 15.3
        assert data[0].timestamp == datetime(2024, 1, 15, 12, 0, 0)
        assert data[1].value == 16.1

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_station_data_multiple_variables(self, mock_client_class, client):
        """Test time series with multiple variable types."""
        import gzip
        import json

        mock_response_data = {
            "soiltemp": [
                {
                    "sid": "CA_SITE_001",
                    "date_time": "2024-01-15 12:00:00",
                    "sensor_value": 15.3,
                }
            ],
            "soilVWC": [
                {
                    "sid": "CA_SITE_001",
                    "date_time": "2024-01-15 12:00:00",
                    "sensor_value": 0.45,
                }
            ],
        }

        compressed = gzip.compress(json.dumps(mock_response_data).encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test with 'all' variable
        data = await client.get_station_data(
            station_id="CA_SITE_001",
            variable_name="all",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        # Should return both temperature and moisture records
        assert len(data) == 2

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_available_variables(self, mock_client_class, client):
        """Test fetching available variables for a station."""
        import gzip
        import json

        mock_response_data = {
            "sensors": [
                {
                    "sid": "CA_SITE_001",
                    "name": "Test Station",
                    "project": "CA_PROJECT",
                    "sso": "2-SON",
                    "wgs84_latitude": 38.5,
                    "wgs84_longitude": -120.5,
                }
            ]
        }

        compressed = gzip.compress(json.dumps(mock_response_data).encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test
        variables = await client.get_available_variables("CA_SITE_001")

        assert len(variables) == 4
        assert any(v.variable_name == "soiltemp" for v in variables)
        assert any(v.variable_name == "soilVWC" for v in variables)
        assert any(v.variable_name == "airtemp" for v in variables)
        assert any(v.variable_name == "waterlevel" for v in variables)

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_stations_station_not_found(self, mock_client_class, client):
        """Test handling of not found response."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Should raise HenryAPIError when data is invalid
        with pytest.raises(HenryAPIError):
            await client.get_stations(sso_code="2-SON")

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_connect(self, mock_client_class, client):
        """Test connection test."""
        import gzip
        import json

        mock_response_data = {"sensors": []}
        compressed = gzip.compress(json.dumps(mock_response_data).encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test
        result = await client.connect()
        assert result is True

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_client_class, client):
        """Test connection failure."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Connection failed")

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client._client = mock_client

        # Test
        with pytest.raises(HenryNetworkError):
            await client.connect()


class TestHenryDataModels:
    """Test Henry data models."""

    def test_henry_station(self):
        """Test HenryStation dataclass."""
        station = HenryStation(
            station_id="CA_SITE_001",
            station_name="Test Station",
            project_code="CA_PROJECT",
            latitude=38.5,
            longitude=-120.5,
            elevation_m=1500,
            state="CA",
        )

        assert station.station_id == "CA_SITE_001"
        assert station.station_name == "Test Station"
        assert station.latitude == 38.5
        assert station.elevation_m == 1500

    def test_henry_time_series_data_point(self):
        """Test HenryTimeSeriesDataPoint dataclass."""
        dt = datetime(2024, 1, 15, 12, 0, 0)
        dp = HenryTimeSeriesDataPoint(
            station_id="CA_SITE_001",
            element_code="STO:-4:1",
            timestamp=dt,
            value=15.3,
            duration="DAILY",
        )

        assert dp.station_id == "CA_SITE_001"
        assert dp.element_code == "STO:-4:1"
        assert dp.timestamp == dt
        assert dp.value == 15.3
        assert dp.duration == "DAILY"

    def test_henry_time_series_with_flags(self):
        """Test HenryTimeSeriesDataPoint with QC/QA flags."""
        dt = datetime(2024, 1, 15, 12, 0, 0)
        dp = HenryTimeSeriesDataPoint(
            station_id="CA_SITE_001",
            element_code="SMS:-2:1",
            timestamp=dt,
            value=0.45,
            qc_flag="OK",
            qa_flag="FLAGGED",
        )

        assert dp.qc_flag == "OK"
        assert dp.qa_flag == "FLAGGED"


class TestHenryConvenience:
    """Test Henry convenience functions."""

    @patch("soildb.henry.convenience.HenryClient")
    @pytest.mark.asyncio
    async def test_find_henry_stations(self, mock_client_class):
        """Test find_henry_stations convenience function."""
        from soildb.henry import find_henry_stations

        # Mock client
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mock_station = HenryStation(
            station_id="CA_SITE_001",
            station_name="Test Station",
            project_code="CA_PROJECT",
            latitude=38.5,
            longitude=-120.5,
        )
        mock_client.get_stations.return_value = [mock_station]
        mock_client_class.return_value = mock_client

        # Test
        stations = await find_henry_stations(sso_code="2-SON")

        assert len(stations) == 1
        assert stations[0]["station_id"] == "CA_SITE_001"

    @patch("soildb.henry.convenience.HenryClient")
    @pytest.mark.asyncio
    async def test_fetch_henry_data(self, mock_client_class):
        """Test fetch_henry_data convenience function."""
        from soildb.henry import fetch_henry_data

        # Mock client
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        dt = datetime(2024, 1, 15, 12, 0, 0)
        mock_dp = HenryTimeSeriesDataPoint(
            station_id="CA_SITE_001",
            element_code="STO:-4:1",
            timestamp=dt,
            value=15.3,
        )
        mock_client.get_station_data.return_value = [mock_dp]
        mock_client_class.return_value = mock_client

        # Test
        data = await fetch_henry_data(
            station_id="CA_SITE_001",
            variable_name="soiltemp",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert len(data) == 1
        assert data[0]["station_id"] == "CA_SITE_001"
        assert data[0]["value"] == 15.3


__all__ = [
    "TestHenryUtilities",
    "TestHenryClient",
    "TestHenryDataModels",
    "TestHenryConvenience",
]
