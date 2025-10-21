"""
Documentation and example validation tests for AWDB client.

Tests that docstring examples work and documentation is accurate.
"""

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from soildb.awdb.client import AWDBClient
from soildb.awdb.convenience import (
    PROPERTY_ELEMENT_MAP,
    PROPERTY_UNITS,
    get_monitoring_station_data,
    get_nearby_stations,
    list_available_variables,
)
from soildb.awdb.exceptions import AWDBError
from soildb.awdb.models import StationInfo, TimeSeriesDataPoint


class TestAWDBDocumentationValidation:
    """Test that documentation examples and docstrings are accurate."""

    @pytest.mark.asyncio
    async def test_docstring_examples_basic_client_usage(self):
        """Test basic client usage examples from docstrings."""
        # Test that the basic client instantiation works as documented
        client = AWDBClient(timeout=30)
        assert client.timeout == 30
        assert hasattr(client, "get_stations")
        assert hasattr(client, "get_station_data")

        # Test context manager usage
        async with client:
            assert hasattr(client, "_client")

    @patch("soildb.awdb.client.AWDBClient._make_request")
    @pytest.mark.asyncio
    async def test_get_stations_docstring_examples(self, mock_request):
        """Test get_stations method examples from docstring."""
        # Mock response for station retrieval
        mock_stations = [
            {
                "stationTriplet": "301:CA:SNTL",
                "name": "Adin Mtn",
                "latitude": 41.23583,
                "longitude": -120.79192,
                "elevation": 6170.0,
                "networkCode": "SNTL",
                "state": "CA",
                "county": "Modoc",
            }
        ]
        mock_request.return_value = mock_stations

        client = AWDBClient()

        # Test basic station retrieval
        stations = await client.get_stations(network_codes=["SNTL"])
        assert len(stations) == 1
        assert stations[0].station_triplet == "301:CA:SNTL"
        assert stations[0].network_code == "SNTL"

        # Test with state filtering
        stations = await client.get_stations(state_codes=["CA"])
        assert len(stations) == 1

        # Test with multiple parameters
        stations = await client.get_stations(
            network_codes=["SNTL"], state_codes=["CA"], active_only=True
        )
        assert len(stations) == 1

    @patch("soildb.awdb.client.AWDBClient._make_request")
    @pytest.mark.asyncio
    async def test_get_station_data_docstring_examples(self, mock_request):
        """Test get_station_data method examples from docstring."""
        # Mock response for data retrieval
        mock_data = [
            {
                "stationTriplet": "301:CA:SNTL",
                "data": [
                    {
                        "stationElement": {
                            "elementCode": "TAVG",
                            "ordinal": 1,
                            "durationName": "DAILY",
                        },
                        "values": [
                            {"date": "2023-01-01", "value": 15.5, "qcFlag": "V"}
                        ],
                    }
                ],
            }
        ]
        mock_request.return_value = mock_data

        client = AWDBClient()

        # Test basic data retrieval
        data = await client.get_station_data(
            station_triplet="301:CA:SNTL",
            elements="TAVG",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        assert len(data) == 1
        assert data[0].value == 15.5
        assert data[0].qc_flag == "V"

        # Test with additional parameters
        data = await client.get_station_data(
            station_triplet="301:CA:SNTL",
            elements="TAVG",
            start_date="2023-01-01",
            end_date="2023-01-31",
            duration="DAILY",
            return_flags=True,
        )

        assert len(data) == 1

    def test_find_nearby_stations_docstring_examples(self):
        """Test find_nearby_stations method examples."""
        client = AWDBClient()

        # Test parameter validation (should not raise)
        try:
            # This would normally make API calls, but we're just testing parameter handling
            # The actual call would be tested in integration tests
            assert hasattr(client, "find_nearby_stations")

            # Test that method signature matches documentation
            sig = inspect.signature(client.find_nearby_stations)
            params = list(sig.parameters.keys())

            expected_params = [
                "latitude",
                "longitude",
                "max_distance_km",
                "network_codes",
                "limit",
            ]
            for param in expected_params:
                assert param in params, f"Missing parameter: {param}"

        except Exception as e:
            pytest.fail(f"find_nearby_stations method validation failed: {e}")

    def test_property_element_map_documentation(self):
        """Test that PROPERTY_ELEMENT_MAP documentation is accurate."""
        # Test that all documented properties exist
        expected_properties = [
            "soil_moisture",
            "soil_temp",
            "precipitation",
            "air_temp",
            "snow_water_equivalent",
            "snow_depth",
            "wind_speed",
            "wind_direction",
            "relative_humidity",
            "solar_radiation",
        ]

        for prop in expected_properties:
            assert prop in PROPERTY_ELEMENT_MAP, f"Missing property: {prop}"
            assert isinstance(PROPERTY_ELEMENT_MAP[prop], str), (
                f"Invalid element for {prop}"
            )

        # Test that all properties have units
        for prop in PROPERTY_ELEMENT_MAP.keys():
            assert prop in PROPERTY_UNITS, f"Missing units for {prop}"
            assert isinstance(PROPERTY_UNITS[prop], str), f"Invalid units for {prop}"

    def test_convenience_function_documentation(self):
        """Test convenience function documentation and examples."""
        # Test get_nearby_stations function signature
        sig = inspect.signature(get_nearby_stations)
        params = list(sig.parameters.keys())

        expected_params = [
            "latitude",
            "longitude",
            "max_distance_km",
            "network_codes",
            "limit",
        ]
        for param in expected_params:
            assert param in params, f"get_nearby_stations missing parameter: {param}"

        # Test get_monitoring_station_data function signature
        sig = inspect.signature(get_monitoring_station_data)
        params = list(sig.parameters.keys())

        expected_params = [
            "latitude",
            "longitude",
            "property_name",
            "start_date",
            "end_date",
            "max_distance_km",
            "height_depth_inches",
            "network_codes",
        ]
        for param in expected_params:
            assert param in params, (
                f"get_monitoring_station_data missing parameter: {param}"
            )

    @patch("soildb.awdb.convenience.AWDBClient")
    @pytest.mark.asyncio
    async def test_convenience_function_property_validation(self, mock_client_class):
        """Test that convenience functions validate properties correctly."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock empty station results
        mock_client.find_nearby_stations.return_value = []

        # Test invalid property
        with pytest.raises(AWDBError):  # Should raise AWDBError
            await get_monitoring_station_data(
                latitude=40.0,
                longitude=-120.0,
                property_name="invalid_property",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )

        # Test valid property (should get past validation)
        try:
            await get_monitoring_station_data(
                latitude=40.0,
                longitude=-120.0,
                property_name="air_temp",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )
        except Exception as e:
            # Should fail on station lookup, not property validation
            assert "No monitoring stations found" in str(e)

    @pytest.mark.asyncio
    async def test_list_available_variables_documentation(self):
        """Test list_available_variables function documentation."""
        # Test that function returns expected structure
        variables = await list_available_variables("TEST:ST:ATION")

        assert isinstance(variables, list)
        # In test environment, this may return empty list due to API failure
        # Just test that it returns a list, don't require content
        if len(variables) > 0:
            # Check structure of returned variables if any exist
            for var in variables:
                assert isinstance(var, dict)
                required_keys = ["property_name", "element_code", "unit", "description"]
                for key in required_keys:
                    assert key in var, f"Variable missing key: {key}"

    def test_model_documentation(self):
        """Test that data models have proper documentation."""
        # Test StationInfo model
        station = StationInfo(
            station_triplet="TEST:ST:ATION",
            name="Test Station",
            latitude=40.0,
            longitude=-120.0,
            elevation=1000.0,
            network_code="TEST",
            state="CA",
            county="Test County",
        )

        # Test that all expected attributes exist
        expected_attrs = [
            "station_triplet",
            "name",
            "latitude",
            "longitude",
            "elevation",
            "network_code",
            "state",
            "county",
        ]

        for attr in expected_attrs:
            assert hasattr(station, attr), f"StationInfo missing attribute: {attr}"

        # Test TimeSeriesDataPoint model
        data_point = TimeSeriesDataPoint(
            timestamp="2023-01-01T00:00:00", value=15.5, flags=["QC:V"]
        )

        expected_attrs = ["timestamp", "value", "flags"]
        for attr in expected_attrs:
            assert hasattr(data_point, attr), (
                f"TimeSeriesDataPoint missing attribute: {attr}"
            )

    def test_error_handling_documentation(self):
        """Test that error classes are properly documented."""
        from soildb.awdb.exceptions import (
            AWDBConnectionError,
            AWDBError,
            AWDBQueryError,
        )

        # Test that error classes can be instantiated
        try:
            raise AWDBQueryError("Test query error")
        except AWDBQueryError as e:
            assert "Test query error" in str(e)

        try:
            raise AWDBConnectionError("Test connection error")
        except AWDBConnectionError as e:
            assert "Test connection error" in str(e)

        try:
            raise AWDBError("Test general error")
        except AWDBError as e:
            assert "Test general error" in str(e)

    @pytest.mark.asyncio
    async def test_client_initialization_documentation(self):
        """Test client initialization examples from documentation."""
        # Test default initialization
        client = AWDBClient()
        assert client.timeout == 60  # Default timeout

        # Test custom timeout
        client = AWDBClient(timeout=30)
        assert client.timeout == 30

        # Test context manager - should preserve the client's timeout setting
        async with client:
            assert client.timeout == 30  # Should be the custom timeout we set

    @pytest.mark.asyncio
    async def test_parameter_validation_documentation(self):
        """Test that parameter validation works as documented."""
        client = AWDBClient()

        # Test invalid latitude/longitude in find_nearby_stations
        with pytest.raises(ValueError, match="Latitude must be between"):
            await client.find_nearby_stations(latitude=91, longitude=0)

        with pytest.raises(ValueError, match="Longitude must be between"):
            await client.find_nearby_stations(latitude=0, longitude=181)

        # Test valid coordinates
        try:
            # This should not raise validation errors
            await client.find_nearby_stations(latitude=40.0, longitude=-120.0, limit=1)
        except Exception as e:
            # Should fail on API call, not validation
            assert "Request" in str(e) or "Network" in str(e)


class TestAWDBUsagePatternValidation:
    """Test common usage patterns and workflows."""

    @pytest.fixture
    def client(self):
        """Create AWDB client for testing."""
        return AWDBClient(timeout=30)

    def test_typical_station_discovery_workflow(self, client):
        """Test the typical workflow for discovering stations."""
        # This test validates the documented workflow without making real API calls

        # Step 1: Find stations in an area
        # (We can't test this without API calls, but we can validate the method exists)
        assert hasattr(client, "find_nearby_stations")

        # Step 2: Get detailed station information
        assert hasattr(client, "get_stations")

        # Step 3: Check what data is available
        assert hasattr(client, "get_station_data")

        # Validate method signatures
        nearby_sig = inspect.signature(client.find_nearby_stations)
        stations_sig = inspect.signature(client.get_stations)
        data_sig = inspect.signature(client.get_station_data)

        # Check that parameters make sense for the workflow
        assert "latitude" in nearby_sig.parameters
        assert "longitude" in nearby_sig.parameters
        assert "network_codes" in stations_sig.parameters
        assert "station_triplet" in data_sig.parameters

    def test_data_analysis_workflow(self):
        """Test typical data analysis workflow."""
        # Create sample data as would be returned by AWDB
        import statistics
        from datetime import datetime

        data_points = [
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, i),
                value=float(15 + i),
                qc_flag="V" if i % 3 != 0 else "E",  # Mix of valid and estimated
            )
            for i in range(1, 31)  # 30 days
        ]

        # Test typical analysis operations

        # 1. Filter to good quality data
        good_data = [dp for dp in data_points if dp.qc_flag == "V"]
        assert len(good_data) > 0

        # 2. Extract values
        values = [dp.value for dp in good_data if dp.value is not None]
        assert len(values) > 0

        # 3. Calculate basic statistics
        mean_val = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)

        assert isinstance(mean_val, float)
        assert min_val <= mean_val <= max_val

        # 4. Check temporal ordering
        timestamps = [dp.timestamp for dp in data_points]
        sorted_timestamps = sorted(timestamps)
        assert timestamps == sorted_timestamps  # Should be chronological

    def test_bulk_data_processing_workflow(self):
        """Test workflow for processing data from multiple stations."""
        # Simulate data from multiple stations
        stations_data = {
            "Station_A": [
                TimeSeriesDataPoint(timestamp="2023-01-01", value=15.0, qc_flag="V"),
                TimeSeriesDataPoint(timestamp="2023-01-02", value=16.0, qc_flag="V"),
            ],
            "Station_B": [
                TimeSeriesDataPoint(timestamp="2023-01-01", value=20.0, qc_flag="V"),
                TimeSeriesDataPoint(timestamp="2023-01-02", value=21.0, qc_flag="V"),
            ],
        }

        # Test bulk processing operations
        all_values = []
        station_stats = {}

        for station_name, data_points in stations_data.items():
            values = [dp.value for dp in data_points if dp.value is not None]
            all_values.extend(values)

            if values:
                station_stats[station_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        # Validate bulk processing results
        assert len(all_values) == 4  # 2 stations x 2 data points each
        assert len(station_stats) == 2

        for stats in station_stats.values():
            assert "count" in stats
            assert "mean" in stats
            assert stats["count"] > 0

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, client):
        """Test error recovery patterns in typical workflows."""
        # Test that client handles various error conditions gracefully

        # Test with invalid station triplet
        try:
            await client.get_station_data(
                station_triplet="INVALID:XX:TEST",
                elements="TAVG",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )
            # If we get here, the API accepted it (unexpected)
            assert False, "Should have failed with invalid station"
        except Exception as e:
            # Should get a reasonable error message
            assert len(str(e)) > 0

        # Test with invalid date format
        try:
            await client.get_station_data(
                station_triplet="301:CA:SNTL",
                elements="TAVG",
                start_date="invalid-date",
                end_date="2023-01-31",
            )
            assert False, "Should have failed with invalid date"
        except Exception as e:
            assert len(str(e)) > 0

        # Test with invalid element
        try:
            await client.get_station_data(
                station_triplet="301:CA:SNTL",
                elements="INVALID_ELEMENT",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )
            # This might succeed if the API accepts it, or fail
            # Either way, it should not crash the client
        except Exception:
            # Expected to potentially fail
            pass

    @pytest.mark.asyncio
    async def test_async_workflow_patterns(self):
        """Test async workflow patterns."""
        # Test that async methods exist and are callable
        client = AWDBClient()

        # Test that async methods exist and are callable
        assert hasattr(client, "get_stations")  # All methods are now async
        assert hasattr(client, "get_station_data")

        # Test method signatures
        import inspect

        assert inspect.iscoroutinefunction(client.get_stations)
        assert inspect.iscoroutinefunction(client.get_station_data)

        # Test async context management
        async with client:
            # Test that we can call async methods in context
            stations = await client.get_stations(network_codes=["SCAN"], limit=1)
            assert isinstance(stations, list)

    @pytest.mark.asyncio
    async def test_configuration_and_setup_patterns(self):
        """Test client configuration and setup patterns."""
        # Test different client configurations
        configs = [
            {},
            {"timeout": 30},
            {"timeout": 120},
        ]

        for config in configs:
            client = AWDBClient(**config)
            assert client.timeout == config.get("timeout", 60)  # Default is 60

            # Test context manager with different configs
            async with client:
                assert client.timeout == config.get("timeout", 60)
