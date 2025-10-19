"""
Tests for AWDB (SCAN/SNOTEL) module.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from soildb.awdb.client import AWDBClient
from soildb.awdb.exceptions import AWDBConnectionError, AWDBQueryError
from soildb.awdb.models import StationInfo


class TestAWDBClient:
    """Test AWDBClient functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return AWDBClient(timeout=5)

    def test_init(self, client):
        """Test client initialization."""
        assert client.timeout == 5
        assert client.BASE_URL == "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

    @patch("httpx.Client")
    def test_get_stations_success(self, mock_client_class, client):
        """Test successful station retrieval."""
        # Mock response data
        mock_response_data = [
            {
                "stationTriplet": "1234:UT:SNTL",
                "name": "Test Station",
                "latitude": 40.0,
                "longitude": -110.0,
                "elevation": 2500,
                "networkCode": "SNOTEL",
                "state": "UT",
                "county": "Salt Lake",
            }
        ]

        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test
        stations = client.get_stations()

        assert len(stations) == 1
        station = stations[0]
        assert station.station_triplet == "1234:UT:SNTL"
        assert station.name == "Test Station"
        assert station.latitude == 40.0
        assert station.longitude == -110.0
        assert station.elevation == 2500
        assert station.network_code == "SNOTEL"
        assert station.state == "UT"
        assert station.county == "Salt Lake"

    @patch("httpx.Client")
    def test_get_stations_with_filters(self, mock_client_class, client):
        """Test station retrieval with network and state filters."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test with filters
        client.get_stations(network_codes=["SCAN"], state_codes=["CA"])

        # Verify correct parameters were sent
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["networkCodes"] == "SCAN"
        assert params["stateCodes"] == "CA"
        assert params["logicalAnd"] == "true"

    def test_find_nearby_stations(self, client):
        """Test finding nearby stations."""
        # Mock stations
        mock_stations = [
            StationInfo(
                "1001:CA:SCAN",
                "Station A",
                37.0,
                -120.0,
                1000,
                "SCAN",
                "CA",
                "County A",
            ),
            StationInfo(
                "1002:CA:SCAN",
                "Station B",
                38.0,
                -121.0,
                1500,
                "SCAN",
                "CA",
                "County B",
            ),
        ]

        with patch.object(client, "get_stations", return_value=mock_stations):
            nearby = client.find_nearby_stations(
                37.5, -120.5, max_distance_km=100, limit=5
            )

            assert len(nearby) == 2
            # Should be sorted by distance
            assert nearby[0][1] <= nearby[1][1]  # First should be closer

    @patch("httpx.Client")
    def test_get_station_data_success(self, mock_client_class, client):
        """Test successful station data retrieval."""
        # Mock response data
        mock_response_data = [
            {
                "data": [
                    {
                        "values": [
                            {"date": "2023-01-01", "value": 25.5},
                            {"date": "2023-01-02", "value": 26.0},
                        ]
                    }
                ]
            }
        ]

        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test
        data = client.get_station_data(
            "1234:UT:SNTL", "SMS", "2023-01-01", "2023-01-02"
        )

        assert len(data) == 2
        assert data[0]["timestamp"] == datetime(2023, 1, 1)
        assert data[0]["value"] == 25.5
        assert data[1]["timestamp"] == datetime(2023, 1, 2)
        assert data[1]["value"] == 26.0

    @patch("httpx.Client")
    def test_get_station_data_with_timezone(self, mock_client_class, client):
        """Test station data retrieval with ISO timestamp including timezone."""
        mock_response_data = [
            {
                "data": [
                    {
                        "values": [
                            {"date": "2023-01-01T12:00:00Z", "value": 25.5},
                        ]
                    }
                ]
            }
        ]

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        data = client.get_station_data(
            "1234:UT:SNTL", "SMS", "2023-01-01", "2023-01-01"
        )

        assert len(data) == 1
        assert data[0]["timestamp"] == datetime(
            2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
        )

    @patch("httpx.Client")
    def test_timeout_error(self, mock_client_class, client):
        """Test timeout error handling."""
        from httpx import TimeoutException

        mock_client = Mock()
        mock_client.get.side_effect = TimeoutException("Timeout")
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        with pytest.raises(AWDBConnectionError, match="Request timeout"):
            client.get_stations()

    @patch("httpx.Client")
    def test_http_error_404(self, mock_client_class, client):
        """Test 404 error handling."""
        from httpx import HTTPStatusError

        mock_response = Mock()
        mock_response.status_code = 404

        mock_client = Mock()
        mock_client.get.side_effect = HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        with pytest.raises(AWDBQueryError, match="Station or data not found"):
            client.get_stations()

    def test_haversine_distance(self, client):
        """Test distance calculation."""
        # Test known distance (approximately)
        distance = client._haversine_distance(40.0, -110.0, 41.0, -110.0)
        assert distance > 110  # Roughly 111 km
        assert distance < 112

        # Test same point
        distance = client._haversine_distance(40.0, -110.0, 40.0, -110.0)
        assert distance == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("soildb.awdb.convenience.AWDBClient")
    def test_get_nearby_stations(self, mock_client_class):
        """Test get_nearby_stations convenience function."""
        from soildb.awdb.convenience import get_nearby_stations

        # Mock client and response
        mock_client = Mock()
        mock_station = StationInfo(
            "1234:UT:SNTL",
            "Test Station",
            40.0,
            -110.0,
            2500,
            "SNOTEL",
            "UT",
            "Salt Lake",
        )
        mock_client.find_nearby_stations.return_value = [(mock_station, 10.5)]
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client_class.return_value.__exit__.return_value = None

        result = get_nearby_stations(40.0, -110.0, max_distance_km=50)

        assert len(result) == 1
        assert result[0]["station_triplet"] == "1234:UT:SNTL"
        assert result[0]["name"] == "Test Station"
        assert result[0]["distance_km"] == 10.5

    @patch("soildb.awdb.convenience.AWDBClient")
    def test_get_monitoring_station_data(self, mock_client_class):
        """Test get_monitoring_station_data convenience function."""
        from soildb.awdb.convenience import get_monitoring_station_data

        # Mock client and response
        mock_client = Mock()
        mock_station = StationInfo(
            "1234:UT:SNTL",
            "Test Station",
            40.0,
            -110.0,
            2500,
            "SNOTEL",
            "UT",
            "Salt Lake",
        )

        # Mock find_nearby_stations
        mock_client.find_nearby_stations.return_value = [(mock_station, 10.5)]

        # Mock get_station_data
        mock_client.get_station_data.return_value = [
            {"timestamp": datetime(2023, 1, 1), "value": 25.5, "flags": []},
            {"timestamp": datetime(2023, 1, 2), "value": 26.0, "flags": []},
        ]

        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client_class.return_value.__exit__.return_value = None

        result = get_monitoring_station_data(
            latitude=40.0,
            longitude=-110.0,
            property_name="soil_moisture",
            start_date="2023-01-01",
            end_date="2023-01-02",
        )

        assert result["site_id"] == "1234:UT:SNTL"
        assert result["property_name"] == "soil_moisture"
        assert result["unit"] == "volumetric %"
        assert len(result["data_points"]) == 2
        assert result["metadata"]["distance_km"] == 10.5
        assert result["metadata"]["n_data_points"] == 2

    def test_invalid_property_name(self):
        """Test error handling for invalid property names."""
        from soildb.awdb.convenience import get_monitoring_station_data
        from soildb.awdb.exceptions import AWDBError

        with pytest.raises(AWDBError, match="Unsupported property"):
            get_monitoring_station_data(
                latitude=40.0,
                longitude=-110.0,
                property_name="invalid_property",
                start_date="2023-01-01",
                end_date="2023-01-02",
            )

    def test_invalid_date_format(self):
        """Test error handling for invalid date formats."""
        from soildb.awdb.convenience import get_monitoring_station_data
        from soildb.awdb.exceptions import AWDBError

        with pytest.raises(AWDBError, match="Invalid date format"):
            get_monitoring_station_data(
                latitude=40.0,
                longitude=-110.0,
                property_name="soil_moisture",
                start_date="invalid-date",
                end_date="2023-01-02",
            )

    def test_list_available_variables(self):
        """Test list_available_variables function."""
        from soildb.awdb.convenience import list_available_variables

        variables = list_available_variables("1234:UT:SNTL")

        assert len(variables) > 0
        # Check that soil_moisture is included
        soil_moisture_vars = [
            v for v in variables if v["property_name"] == "soil_moisture"
        ]
        assert len(soil_moisture_vars) == 1
        assert soil_moisture_vars[0]["element_code"] == "SMS"
        assert soil_moisture_vars[0]["unit"] == "volumetric %"
