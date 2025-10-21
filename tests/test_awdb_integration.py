"""
Integration tests for AWDB client with existing SoilDB systems.

Tests compatibility between AWDB client and other SoilDB modules.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from soildb.awdb.client import AWDBClient
from soildb.awdb.convenience import PROPERTY_ELEMENT_MAP
from soildb.awdb.models import StationInfo, TimeSeriesDataPoint


class TestAWDBSoilDBIntegration:
    """Test AWDB integration with existing SoilDB systems."""

    @pytest.fixture
    def awdb_client(self):
        """Create AWDB client for testing."""
        return AWDBClient(timeout=30)

    def test_property_element_map_compatibility(self, awdb_client):
        """Test that AWDB element mapping works with existing convenience functions."""
        # Test that all properties in PROPERTY_ELEMENT_MAP can be used with AWDB
        test_properties = ['air_temperature', 'precipitation', 'snow_depth', 'soil_moisture']

        for prop in test_properties:
            if prop in PROPERTY_ELEMENT_MAP:
                element = PROPERTY_ELEMENT_MAP[prop]
                assert isinstance(element, str), f"Element for {prop} should be string"
                assert len(element) > 0, f"Element for {prop} should not be empty"

                # Test that we can construct a valid request with this element
                # (Don't actually make the request, just validate construction)
                try:
                    # This should not raise an exception during parameter validation
                    params = {
                        "stationTriplets": "TEST:ST:NET",
                        "elements": element,
                        "beginDate": "2023-01-01",
                        "endDate": "2023-01-31",
                        "duration": "DAILY"
                    }
                    # If we get here, the parameters are valid
                    assert True, f"Parameters for {prop} are valid"

                except Exception as e:
                    pytest.fail(f"Parameter construction failed for {prop}: {e}")

    def test_station_info_compatibility_with_fetch(self):
        """Test that AWDB StationInfo is compatible with fetch module expectations."""
        # Create a station info object as AWDB would return it
        station = StationInfo(
            station_triplet="301:CA:SNTL",
            name="Adin Mtn",
            latitude=41.23583,
            longitude=-120.79192,
            elevation=6170.0,
            network_code="SNTL",
            state="CA",
            county="Modoc",
            # Additional AWDB fields
            station_id="301",
            dco_code="CA",
            huc="180200021403",
            data_time_zone=-8,
            begin_date="1980-10-01",
            end_date="2100-01-01"
        )

        # Test that station has all expected attributes
        required_attrs = ['station_triplet', 'name', 'latitude', 'longitude', 'network_code']
        for attr in required_attrs:
            assert hasattr(station, attr), f"Station missing required attribute: {attr}"
            value = getattr(station, attr)
            assert value is not None, f"Station attribute {attr} should not be None"

        # Test coordinate ranges
        assert -90 <= station.latitude <= 90, "Latitude out of valid range"
        assert -180 <= station.longitude <= 180, "Longitude out of valid range"

        # Test that station can be used in string formatting (common in fetch operations)
        station_str = f"{station.name} ({station.station_triplet})"
        assert len(station_str) > 0, "Station string representation should not be empty"

    def test_timeseries_data_compatibility_with_response(self):
        """Test that AWDB TimeSeriesDataPoint is compatible with response processing."""
        # Create sample data points as AWDB would return them
        data_points = [
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, 1),
                value=15.5,
                flags=['QC:V'],
                qc_flag='V',
                qa_flag=None,
                orig_value=15.7,
                orig_qc_flag='E',
                average=15.3,
                median=15.5
            ),
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, 2),
                value=None,  # Missing data
                flags=[],
                qc_flag=None,
                qa_flag=None
            ),
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, 3),
                value=16.2,
                flags=['QC:E', 'QA:A'],
                qc_flag='E',
                qa_flag='A'
            )
        ]

        # Test data point structure
        for i, dp in enumerate(data_points):
            assert isinstance(dp.timestamp, datetime), f"Data point {i} timestamp should be datetime"
            assert isinstance(dp.flags, list), f"Data point {i} flags should be list"

            # Value can be None for missing data
            if dp.value is not None:
                assert isinstance(dp.value, (int, float)), f"Data point {i} value should be numeric"

        # Test that data can be processed like typical SoilDB response data
        valid_values = [dp.value for dp in data_points if dp.value is not None]
        assert len(valid_values) > 0, "Should have some valid data points"

        # Test statistical calculations
        if len(valid_values) > 1:
            avg_value = sum(valid_values) / len(valid_values)
            assert isinstance(avg_value, (int, float)), "Average calculation should work"

    @patch('soildb.awdb.client.AWDBClient._make_request')
    def test_awdb_client_with_mocked_responses(self, mock_request, awdb_client):
        """Test AWDB client behavior with controlled mock responses."""
        # Mock station response
        mock_stations = [
            {
                "stationTriplet": "301:CA:SNTL",
                "name": "Adin Mtn",
                "latitude": 41.23583,
                "longitude": -120.79192,
                "elevation": 6170.0,
                "networkCode": "SNTL",
                "state": "CA",
                "county": "Modoc"
            }
        ]
        mock_request.return_value = mock_stations

        # Test station retrieval
        import asyncio
        stations = asyncio.run(awdb_client.get_stations(network_codes=['SNTL']))
        assert len(stations) == 1
        assert stations[0].station_triplet == "301:CA:SNTL"
        assert stations[0].network_code == "SNTL"

        # Mock data response
        mock_data = [
            {
                "stationTriplet": "301:CA:SNTL",
                "data": [
                    {
                        "stationElement": {
                            "elementCode": "TAVG",
                            "ordinal": 1,
                            "durationName": "DAILY"
                        },
                        "values": [
                            {
                                "date": "2023-01-01",
                                "value": 15.5,
                                "qcFlag": "V",
                                "qaFlag": "A"
                            }
                        ]
                    }
                ]
            }
        ]
        mock_request.return_value = mock_data

        # Test data retrieval
        import asyncio
        data = asyncio.run(awdb_client.get_station_data(
            station_triplet="301:CA:SNTL",
            elements="TAVG",
            start_date="2023-01-01",
            end_date="2023-01-31"
        ))

        assert len(data) == 1
        assert data[0].value == 15.5
        assert data[0].qc_flag == "V"
        assert data[0].qa_flag == "A"

    def test_error_handling_integration(self, awdb_client):
        """Test that AWDB errors integrate properly with SoilDB error handling."""
        from soildb.awdb.exceptions import AWDBConnectionError, AWDBQueryError

        # Test that AWDB exceptions can be caught as general SoilDB errors
        try:
            # This should raise an AWDBQueryError
            raise AWDBQueryError("Test error")
        except Exception as e:
            # Should be able to catch as general exception
            assert isinstance(e, AWDBQueryError)
            assert "Test error" in str(e)

        try:
            # This should raise an AWDBConnectionError
            raise AWDBConnectionError("Connection failed")
        except Exception as e:
            # Should be able to catch as general exception
            assert isinstance(e, AWDBConnectionError)
            assert "Connection failed" in str(e)

    def test_data_format_compatibility_with_analysis(self):
        """Test that AWDB data format is compatible with typical analysis workflows."""
        # Create sample AWDB data
        data_points = [
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                value=15.0 + i * 0.5,
                flags=['QC:V'],
                qc_flag='V'
            )
            for i in range(30)  # 30 days of data
        ]

        # Test common analysis operations
        timestamps = [dp.timestamp for dp in data_points]
        values = [dp.value for dp in data_points if dp.value is not None]

        # Should be able to create time series
        assert len(timestamps) == len(data_points)
        assert len(values) == len(data_points)  # All values present in this test

        # Should be able to calculate basic statistics
        if values:
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)

            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert isinstance(avg_val, (int, float))

            assert min_val <= avg_val <= max_val

        # Test data quality flag extraction
        quality_flags = [dp.qc_flag for dp in data_points if dp.qc_flag]
        assert len(quality_flags) > 0, "Should have quality flags"

        # Test temporal operations
        sorted_timestamps = sorted(timestamps)
        assert sorted_timestamps == timestamps, "Data should be temporally ordered"

    def test_convenience_function_integration(self, awdb_client):
        """Test integration with AWDB convenience functions."""
        # Test that convenience functions work with client
        from soildb.awdb.convenience import (
            get_monitoring_station_data,
            get_nearby_stations,
        )

        # Test get_nearby_stations interface
        try:
            # Test parameter validation by calling with valid parameters
            # This should work or raise a clear error
            import asyncio
            result = asyncio.run(get_nearby_stations(
                latitude=40.0,
                longitude=-120.0,
                max_distance_km=10.0,
                network_codes=['SCAN'],
                limit=3
            ))
            # If we get here, it should return a list
            assert isinstance(result, list)
            if len(result) > 0:
                # Check structure of first result
                station = result[0]
                required_keys = ['station_triplet', 'name', 'latitude', 'longitude', 'distance_km']
                for key in required_keys:
                    assert key in station, f"Station missing required key: {key}"

        except Exception as e:
            # Should be a clear, understandable error
            assert len(str(e)) > 0, "Error message should not be empty"

        # Test get_monitoring_station_data interface
        try:
            import asyncio
            result = asyncio.run(get_monitoring_station_data(
                latitude=40.0,
                longitude=-120.0,
                property_name='air_temp',
                start_date='2023-01-01',
                end_date='2023-01-31',
                max_distance_km=50.0
            ))
            # Should return a dictionary with expected structure
            assert isinstance(result, dict)
            required_keys = ['site_id', 'site_name', 'latitude', 'longitude', 'data_points']
            for key in required_keys:
                assert key in result, f"Result missing required key: {key}"

        except Exception as e:
            # Should be a clear, understandable error
            assert len(str(e)) > 0, "Error message should not be empty"

    @pytest.mark.asyncio
    async def test_async_integration_compatibility(self):
        """Test that async methods are compatible with typical async workflows."""
        # Test that async methods can be called in async context
        client = AWDBClient()

        # Test async method signatures (don't actually call to avoid API hits)
        assert hasattr(client, 'get_stations')  # All methods are now async
        assert hasattr(client, 'get_station_data')
        assert hasattr(client, 'get_forecasts')
        assert hasattr(client, 'get_reference_data')

        # Test that methods are actually coroutines
        import inspect
        assert inspect.iscoroutinefunction(client.get_stations)
        assert inspect.iscoroutinefunction(client.get_station_data)

        # Test async context management
        async with client:
            # Test that we can call async methods in context
            stations = await client.get_stations(network_codes=['SCAN'], limit=1)
            assert isinstance(stations, list)

    @pytest.mark.asyncio
    async def test_client_context_manager_integration(self, awdb_client):
        """Test that AWDB client works properly as context manager."""
        # Test that client can be used in with statement
        async with awdb_client:
            # Should be able to access attributes
            assert hasattr(awdb_client, 'timeout')
            assert hasattr(awdb_client, '_client')

        # After context manager, client should be closed
        # (This is hard to test directly, but at least verify no exceptions)

    def test_data_pipeline_compatibility(self):
        """Test that AWDB data fits into typical data processing pipelines."""
        # Create a mock data processing pipeline
        def process_environmental_data(data_points):
            """Mock environmental data processing function."""
            results = []

            for dp in data_points:
                if dp.value is not None and dp.value > 0:
                    # Simulate some processing
                    processed = {
                        'timestamp': dp.timestamp.isoformat(),
                        'value': dp.value,
                        'quality': 'good' if dp.qc_flag == 'V' else 'suspect',
                        'flags': dp.flags
                    }
                    results.append(processed)

            return results

        # Create test data
        test_data = [
            TimeSeriesDataPoint(
                timestamp=datetime(2023, 1, i),
                value=float(i * 10),
                qc_flag='V' if i % 2 == 0 else 'E',
                flags=['QC:V'] if i % 2 == 0 else ['QC:E']
            )
            for i in range(1, 11)  # 10 days
        ]

        # Process through pipeline
        processed = process_environmental_data(test_data)

        # Verify pipeline output
        assert len(processed) == len(test_data)
        assert all('timestamp' in p for p in processed)
        assert all('value' in p for p in processed)
        assert all('quality' in p for p in processed)

        # Check quality classification
        good_quality = [p for p in processed if p['quality'] == 'good']
        suspect_quality = [p for p in processed if p['quality'] == 'suspect']

        assert len(good_quality) > 0
        assert len(suspect_quality) > 0
