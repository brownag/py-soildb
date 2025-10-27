"""
Tests for AWDB (SCAN/SNOTEL) module.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

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

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_stations_success(self, mock_client_class, client):
        """Test successful station retrieval."""
        # Mock response data
        mock_response_data = [
            {
                "stationTriplet": "1234:UT:SNTL",
                "name": "Test Station",
                "latitude": 40.0,
                "longitude": -110.0,
                "elevation": 2500,
                "networkCode": "SNTL",
                "state": "UT",
                "county": "Salt Lake",
            }
        ]

        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test
        stations = await client.get_stations()

        assert len(stations) == 1
        station = stations[0]
        assert station.station_triplet == "1234:UT:SNTL"
        assert station.name == "Test Station"
        assert station.latitude == 40.0
        assert station.longitude == -110.0
        assert station.elevation == 2500
        assert station.network_code == "SNTL"
        assert station.state == "UT"
        assert station.county == "Salt Lake"

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_stations_with_filters(self, mock_client_class, client):
        """Test station retrieval with network and state filters."""
        # Mock response with stations from different networks and states
        mock_response_data = [
            {
                "stationTriplet": "1001:CA:SCAN",
                "name": "Station A",
                "latitude": 37.0,
                "longitude": -120.0,
                "elevation": 1000,
                "networkCode": "SCAN",
                "state": "CA",
                "county": "County A",
            },
            {
                "stationTriplet": "1002:UT:SNTL",
                "name": "Station B",
                "latitude": 40.0,
                "longitude": -110.0,
                "elevation": 2000,
                "networkCode": "SNTL",
                "state": "UT",
                "county": "County B",
            },
            {
                "stationTriplet": "1003:WY:SNOW",
                "name": "Station C",
                "latitude": 41.0,
                "longitude": -105.0,
                "elevation": 3000,
                "networkCode": "SNOW",
                "state": "WY",
                "county": "County C",
            },
        ]

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test with network filter only - now uses server-side filtering
        # Note: Mock returns all stations, but in real API this would be filtered server-side
        stations = await client.get_stations(network_codes=["SCAN"])
        # Since we're using a mock that returns all stations, we can't test the filtering result
        # But we can verify the API call includes the filter parameter
        assert len(stations) == 3  # All mock stations returned
        # The filtering test is in the parameter verification below

        # Test with state filter only - mock returns all, but verify parameter is sent
        stations = await client.get_stations(state_codes=["UT"])
        assert len(stations) == 3  # Mock returns all stations

        # Test with both filters - mock returns all, but verify parameters are sent
        stations = await client.get_stations(network_codes=["SNOW"], state_codes=["WY"])
        assert len(stations) == 3  # Mock returns all stations

        # Test with filters that match nothing - mock returns all, but verify parameter is sent
        stations = await client.get_stations(network_codes=["NONEXISTENT"])
        assert len(stations) == 3  # Mock returns all stations

        # Verify filtering parameters are sent to API (server-side filtering)
        # Note: Since multiple calls are made, we check the last call (NONEXISTENT network)
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert "stationTriplets" in params
        assert "*:*:NONEXISTENT" in params["stationTriplets"]
        # The test verifies that parameters are sent to API, not the filtering behavior

    @pytest.mark.asyncio
    async def test_find_nearby_stations(self, client):
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
            nearby = await client.find_nearby_stations(
                37.5, -120.5, max_distance_km=100, limit=5
            )

            assert len(nearby) == 2
            # Should be sorted by distance
            assert nearby[0][1] <= nearby[1][1]  # First should be closer

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_station_data_success(self, mock_client_class, client):
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

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        # Test
        data = await client.get_station_data(
            "1234:UT:SNTL", "SMS", "2023-01-01", "2023-01-02"
        )

        assert len(data) == 2
        assert data[0].timestamp == datetime(2023, 1, 1)
        assert data[0].value == 25.5
        assert data[1].timestamp == datetime(2023, 1, 2)
        assert data[1].value == 26.0

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_get_station_data_with_timezone(self, mock_client_class, client):
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

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        data = await client.get_station_data(
            "1234:UT:SNTL", "SMS", "2023-01-01", "2023-01-01"
        )

        assert len(data) == 1
        assert data[0].timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_timeout_error(self, mock_client_class, client):
        """Test timeout error handling."""
        from httpx import TimeoutException

        mock_client = AsyncMock()
        mock_client.get.side_effect = TimeoutException("Timeout")
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        with pytest.raises(AWDBConnectionError, match="Request timeout"):
            await client.get_stations()

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_http_error_404(self, mock_client_class, client):
        """Test 404 error handling."""
        from httpx import HTTPStatusError

        mock_response = Mock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.side_effect = HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Replace the client's _client with the mock
        client._client = mock_client

        with pytest.raises(AWDBQueryError, match="Station or data not found"):
            await client.get_stations()

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
    @pytest.mark.asyncio
    async def test_get_nearby_stations(self, mock_client_class):
        """Test get_nearby_stations convenience function."""
        from soildb.awdb.convenience import get_nearby_stations

        # Mock client and response
        mock_client = AsyncMock()
        mock_station = StationInfo(
            "1234:UT:SNTL",
            "Test Station",
            40.0,
            -110.0,
            2500,
            "SNTL",
            "UT",
            "Salt Lake",
        )
        mock_client.find_nearby_stations.return_value = [(mock_station, 10.5)]
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        result = await get_nearby_stations(40.0, -110.0, max_distance_km=50)

        assert len(result) == 1
        assert result[0]["station_triplet"] == "1234:UT:SNTL"
        assert result[0]["name"] == "Test Station"
        assert result[0]["distance_km"] == 10.5

    @patch("soildb.awdb.convenience.AWDBClient")
    @pytest.mark.asyncio
    async def test_get_monitoring_station_data(self, mock_client_class):
        """Test get_monitoring_station_data convenience function."""
        from soildb.awdb.convenience import get_monitoring_station_data

        # Mock client and response
        mock_client = AsyncMock()
        mock_station = StationInfo(
            "1234:UT:SNTL",
            "Test Station",
            40.0,
            -110.0,
            2500,
            "SNTL",
            "UT",
            "Salt Lake",
        )

        # Mock find_nearby_stations
        mock_client.find_nearby_stations.return_value = [(mock_station, 10.5)]

        # Mock get_station_data - now returns TimeSeriesDataPoint objects
        from soildb.awdb.models import TimeSeriesDataPoint

        mock_client.get_station_data.return_value = [
            TimeSeriesDataPoint(timestamp=datetime(2023, 1, 1), value=25.5, flags=[]),
            TimeSeriesDataPoint(timestamp=datetime(2023, 1, 2), value=26.0, flags=[]),
        ]

        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        result = await get_monitoring_station_data(
            latitude=40.0,
            longitude=-110.0,
            property_name="soil_moisture",
            start_date="2023-01-01",
            end_date="2023-01-02",
            height_depth_inches=-20,  # Required for soil properties
        )

        assert result["site_id"] == "1234:UT:SNTL"
        assert result["property_name"] == "soil_moisture"
        assert result["unit"] == "pct"
        assert len(result["data_points"]) == 2
        assert result["metadata"]["distance_km"] == 10.5
        assert result["metadata"]["n_data_points"] == 2

    @pytest.mark.asyncio
    async def test_invalid_property_name(self):
        """Test error handling for invalid property names."""
        from soildb.awdb.convenience import get_monitoring_station_data
        from soildb.awdb.exceptions import AWDBError

        with pytest.raises(AWDBError, match="Unsupported property"):
            await get_monitoring_station_data(
                latitude=40.0,
                longitude=-110.0,
                property_name="invalid_property",
                start_date="2023-01-01",
                end_date="2023-01-02",
            )

    @pytest.mark.asyncio
    async def test_invalid_date_format(self):
        """Test error handling for invalid date formats."""
        from soildb.awdb.convenience import get_monitoring_station_data
        from soildb.awdb.exceptions import AWDBError

        with pytest.raises(AWDBError, match="Invalid date format"):
            await get_monitoring_station_data(
                latitude=40.0,
                longitude=-110.0,
                property_name="soil_moisture",
                start_date="invalid-date",
                end_date="2023-01-02",
                height_depth_inches=-20,  # Required for soil properties
            )

    @pytest.mark.asyncio
    async def test_list_available_variables(self):
        """Test list_available_variables function."""
        from soildb.awdb.convenience import list_available_variables

        # Mock the underlying API call to avoid real network requests
        with patch("soildb.awdb.convenience.get_station_sensor_metadata") as mock_metadata:
            mock_metadata.return_value = {
                "sensors": {
                    "soil_moisture": [{"code": "SMS", "depth": "-20"}],
                    "air_temp": [{"code": "AT", "depth": "0"}],
                }
            }
            
            with patch("soildb.awdb.convenience.get_property_unit_from_api") as mock_unit:
                mock_unit.return_value = "pct"
                
                variables = await list_available_variables("301:CA:SNTL")
                
                # Verify the function returned the expected structure
                assert len(variables) == 2
                soil_moisture_vars = [
                    v for v in variables if v["property_name"] == "soil_moisture"
                ]
                assert len(soil_moisture_vars) == 1
                assert soil_moisture_vars[0]["element_code"] == "SMS"
                assert soil_moisture_vars[0]["unit"] == "pct"

    def test_property_element_map_validation(self):
        """Test that PROPERTY_ELEMENT_MAP contains expected properties."""
        from soildb.awdb.convenience import PROPERTY_ELEMENT_MAP, PROPERTY_UNITS

        # Test that core properties exist
        core_properties = [
            "soil_moisture",
            "air_temp",
            "precipitation",
            "snow_depth",
            "snow_water_equivalent",
        ]

        for prop in core_properties:
            assert prop in PROPERTY_ELEMENT_MAP, f"Missing core property: {prop}"
            assert prop in PROPERTY_UNITS, f"Missing units for core property: {prop}"

    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation for client methods."""
        client = AWDBClient()

        # Test invalid coordinates in find_nearby_stations
        with pytest.raises(ValueError, match="Latitude must be between"):
            await client.find_nearby_stations(latitude=91, longitude=0)

        with pytest.raises(ValueError, match="Longitude must be between"):
            await client.find_nearby_stations(latitude=0, longitude=181)

    @pytest.mark.asyncio
    async def test_convenience_function_property_validation(self):
        """Test that convenience functions validate properties correctly."""
        from soildb.awdb.convenience import get_monitoring_station_data
        from soildb.awdb.exceptions import AWDBError

        # Test invalid property
        with pytest.raises(AWDBError, match="Unsupported property"):
            await get_monitoring_station_data(
                latitude=40.0,
                longitude=-120.0,
                property_name="invalid_property",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )

    @pytest.mark.asyncio
    async def test_data_format_compatibility(self):
        """Test that AWDB data format works with basic analysis operations."""
        from datetime import datetime

        from soildb.awdb.models import TimeSeriesDataPoint

        # Create sample data points
        data_points = [
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, i + 1),  # Days 1-5
                value=15.0 + i * 0.5,
                flags=["QC:V"],
                qc_flag="V",
            )
            for i in range(5)  # Small sample for testing
        ]

        # Test basic data processing operations
        values = [dp.value for dp in data_points if dp.value is not None]
        assert len(values) == len(data_points)

        # Test statistical calculations
        if values:
            avg_value = sum(values) / len(values)
            assert isinstance(avg_value, (int, float))

        # Test quality flag extraction
        quality_flags = [dp.qc_flag for dp in data_points if dp.qc_flag]
        assert len(quality_flags) > 0
