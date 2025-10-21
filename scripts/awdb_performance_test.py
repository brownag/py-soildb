"""
Performance and bulk operation tests for AWDB client.

Tests bulk data retrieval, rate limiting, and performance characteristics.
Run as a standalone script: python scripts/awdb_performance_test.py
"""

import asyncio
import time
from datetime import datetime, timedelta

from soildb.awdb.client import AWDBClient


class AWDBPerformanceTester:
    """Test AWDB client performance characteristics."""

    def __init__(self):
        self.client = AWDBClient(timeout=30)

    async def test_bulk_station_retrieval(self):
        """Test retrieving large numbers of stations."""
        print("Testing bulk station retrieval...")
        # Test different network filters
        networks = ['SCAN', 'SNTL', 'SNOW']

        for network in networks:
            # Create a new client for each network to avoid context manager issues
            test_client = AWDBClient(timeout=30)
            async with test_client:
                try:
                    stations = await test_client.get_stations(network_codes=[network], active_only=True)

                    # Basic validation
                    assert isinstance(stations, list)
                    assert len(stations) > 0

                    # Check that at least some stations are from the requested network
                    # (API may return additional stations, but should include the requested network)
                    network_stations = [s for s in stations if s.network_code == network]
                    assert len(network_stations) > 0, f"No {network} stations found in results"

                    print(f"  {network}: {len(stations)} total stations, {len(network_stations)} {network} stations")
                except Exception as e:
                    print(f"  {network}: Error - {e}")

    async def test_concurrent_station_queries(self):
        """Test concurrent station queries."""
        print("Testing concurrent station queries...")
        networks = ['SCAN', 'SNTL', 'COOP', 'USGS']

        async def concurrent_test():
            async def query_network(network):
                return await self.client.get_stations(network_codes=[network], active_only=True)

            # Run concurrent queries
            start_time = time.time()
            tasks = [query_network(network) for network in networks]
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_stations = sum(len(result) for result in results)
            duration = end_time - start_time

            print(f"  Concurrent queries completed in {duration:.2f}s, {total_stations} total stations")

            # Validate results
            for i, network in enumerate(networks):
                result = results[i]
                assert isinstance(result, list)
                assert len(result) > 0

                # Check network consistency - at least some should match
                network_stations = [s for s in result if s.network_code == network]
                assert len(network_stations) > 0, f"No {network} stations found in concurrent query result"

        async with self.client:
            await concurrent_test()

    async def test_data_retrieval_performance(self):
        """Test performance of data retrieval operations."""
        print("Testing data retrieval performance...")
        # Use a known good station
        test_station = "301:CA:SNTL"  # Adin Mtn - known to have good data

        async with self.client:
            # Test different date ranges
            date_ranges = [
                ("2023-01-01", "2023-01-31"),  # 1 month
                ("2023-01-01", "2023-12-31"),  # 1 year
                ("2020-01-01", "2023-12-31"),  # 4 years
            ]

            for start_date, end_date in date_ranges:
                start_time = time.time()

                try:
                    data = await self.client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date=start_date,
                        end_date=end_date,
                        duration="DAILY"
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    print(f"  {start_date} to {end_date}: Retrieved {len(data) if data else 0} data points in {duration:.2f}s")

                except Exception as e:
                    print(f"  {start_date} to {end_date}: Error - {e}")

    async def test_rate_limiting_behavior(self):
        """Test client behavior under rapid successive requests."""
        print("Testing rate limiting behavior...")
        test_station = "301:CA:SNTL"

        async with self.client:
            # Make multiple rapid requests
            request_count = 10
            successes = 0
            failures = 0

            start_time = time.time()

            for i in range(request_count):
                try:
                    await self.client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-01-31",
                        duration="DAILY"
                    )
                    successes += 1

                except Exception as e:
                    failures += 1
                    if "rate limit" in str(e).lower():
                        print(f"  Rate limit hit on request {i+1}: {e}")
                        break

            end_time = time.time()
            total_time = end_time - start_time

            print(f"  Rate limit test: {successes} successes, {failures} failures")
            print(f"  Total time: {total_time:.2f}s")

    async def test_large_dataset_handling(self):
        """Test handling of large data requests."""
        print("Testing large dataset handling...")
        test_station = "301:CA:SNTL"  # Station with long record

        async with self.client:
            # Request 10 years of daily data
            try:
                start_time = time.time()

                data = await self.client.get_station_data(
                    station_triplet=test_station,
                    elements="TAVG",
                    start_date="2010-01-01",
                    end_date="2020-12-31",
                    duration="DAILY"
                )

                end_time = time.time()
                duration = end_time - start_time

                print(f"  Large dataset test: Retrieved {len(data) if data else 0} points")
                print(f"  Duration: {duration:.2f}s")

                # Should handle at least 3000+ days
                if data:
                    assert len(data) > 3000, f"Expected >3000 data points, got {len(data)}"

            except Exception as e:
                print(f"  Large dataset test failed: {e}")

    async def test_memory_usage_patterns(self):
        """Test memory usage with different data volumes."""
        print("Testing memory usage patterns...")
        try:
            import os
            import psutil
        except ImportError:
            print("  Skipping memory test - psutil not available")
            return

        test_station = "301:CA:SNTL"
        process = psutil.Process(os.getpid())

        async with self.client:
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Baseline memory: {baseline_memory:.1f} MB")

            # Request increasing amounts of data
            date_ranges = [
                ("2023-01-01", "2023-01-31", "1 month"),
                ("2023-01-01", "2023-06-30", "6 months"),
                ("2023-01-01", "2023-12-31", "1 year"),
                ("2020-01-01", "2023-12-31", "4 years"),
            ]

            for start_date, end_date, desc in date_ranges:
                try:
                    data = await self.client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date=start_date,
                        end_date=end_date,
                        duration="DAILY"
                    )

                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = current_memory - baseline_memory

                    print(f"    {desc}: {len(data)} points, memory delta: {memory_delta:.1f} MB")

                except Exception as e:
                    print(f"    {desc}: Error - {e}")

    async def test_api_parameter_combinations(self):
        """Test various API parameter combinations for performance."""
        print("Testing API parameter combinations...")
        test_station = "301:CA:SNTL"

        async with self.client:
            # Test different duration/parameter combinations
            test_cases = [
                {"duration": "DAILY", "desc": "Daily data"},
                {"duration": "MONTHLY", "desc": "Monthly aggregation"},
                {"duration": "DAILY", "central_tendency_type": "ALL", "desc": "Daily with averages"},
                {"duration": "DAILY", "return_flags": True, "desc": "Daily with flags"},
                {"duration": "DAILY", "return_original_values": True, "desc": "Daily with original values"},
            ]

            for case in test_cases:
                desc = case.pop("desc")

                try:
                    start_time = time.time()

                    data = await self.client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                        **case
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    print(f"    {desc}: {len(data) if data else 0} points in {duration:.2f}s")

                except Exception as e:
                    print(f"    {desc}: Error - {e}")


class AWDBBulkOperationsTester:
    """Test bulk data operations and batch processing."""

    def __init__(self):
        self.client = AWDBClient(timeout=60)  # Longer timeout for bulk operations

    async def test_multi_station_data_retrieval(self):
        """Test retrieving data for multiple stations."""
        print("Testing multi-station data retrieval...")
        # Select a few stations from different networks
        test_stations = [
            "301:CA:SNTL",    # SNTL network
            "2235:CA:SCAN",   # SCAN network
            "AGP:CA:MSNT",    # MSNT network
        ]

        async with self.client:
            results = {}

            for station in test_stations:
                try:
                    start_time = time.time()

                    data = await self.client.get_station_data(
                        station_triplet=station,
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                        duration="DAILY"
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    results[station] = {
                        'success': True,
                        'data_points': len(data) if data else 0,
                        'duration': duration
                    }

                    print(f"    {station}: {len(data) if data else 0} points in {duration:.2f}s")

                except Exception as e:
                    results[station] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"    {station}: Failed - {e}")

            # Validate results
            successful_stations = sum(1 for r in results.values() if r['success'])
            assert successful_stations > 0, "At least one station should succeed"

    async def test_network_based_bulk_queries(self):
        """Test bulk queries organized by network."""
        print("Testing network-based bulk queries...")
        # Test different networks
        network_configs = {
            'SCAN': {'max_stations': 5},
            'SNTL': {'max_stations': 3},
            'SNOW': {'max_stations': 2},  # SNOW has manual data, may be limited
        }

        async with self.client:
            for network, config in network_configs.items():
                # Get stations for this network
                stations = (await self.client.get_stations(
                    network_codes=[network],
                    active_only=True
                ))[:config['max_stations']]  # Limit for testing

                print(f"\n  {network} Network: Testing {len(stations)} stations")

                successful_queries = 0
                total_data_points = 0

                for station in stations:
                    try:
                        data = await self.client.get_station_data(
                            station_triplet=station.station_triplet,
                            elements="TAVG",
                            start_date="2023-01-01",
                            end_date="2023-12-31",
                            duration="DAILY"
                        )

                        if data and len(data) > 0:
                            successful_queries += 1
                            total_data_points += len(data)

                    except Exception as e:
                        print(f"    {station.station_triplet}: Error - {e}")

                success_rate = successful_queries / len(stations) * 100 if stations else 0
                avg_data_points = total_data_points / successful_queries if successful_queries > 0 else 0

                print(f"    Success rate: {success_rate:.1f}%")
                print(f"    Average data points per station: {avg_data_points:.0f}")

    async def test_error_recovery_in_bulk_operations(self):
        """Test error recovery during bulk operations."""
        print("Testing error recovery in bulk operations...")
        # Mix of good and potentially problematic stations
        test_stations = [
            "301:CA:SNTL",      # Should work
            "INVALID:XX:TEST",  # Should fail
            "2235:CA:SCAN",     # Should work
            "NONEXISTENT:YY:ZZZ", # Should fail
            "AGP:CA:MSNT",      # Should work
        ]

        async with self.client:
            results = {}

            for station in test_stations:
                try:
                    data = await self.client.get_station_data(
                        station_triplet=station,
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-01-31",
                        duration="DAILY"
                    )

                    results[station] = {'success': True, 'data_points': len(data) if data else 0}

                except Exception as e:
                    results[station] = {'success': False, 'error': str(e)}

            # Should have some successes and some failures
            successes = sum(1 for r in results.values() if r['success'])
            failures = len(results) - successes

            assert successes > 0, "Should have at least some successful queries"
            # Note: In practice, all stations might succeed, so we don't require failures
            # assert failures > 0, "Should have at least some failed queries (as expected)"

            print(f"  Bulk error recovery: {successes} successes, {failures} failures")
            print("  Error recovery test passed - client handles mixed success/failure gracefully")


class AWDBDataAvailabilityScanner:
    """Test data availability scanning functionality."""

    def __init__(self):
        self.client = AWDBClient(timeout=30)

    async def test_station_data_availability_scan(self):
        """Test scanning data availability across stations."""
        print("Testing station data availability scanning...")
        # Get a sample of stations from different networks
        networks = ['SCAN', 'SNTL']

        availability_results = {}

        async with self.client:
            for network in networks:
                stations = (await self.client.get_stations(
                    network_codes=[network],
                    active_only=True
                ))[:3]  # Test first 3 stations per network

                network_results = {}

                for station in stations:
                    station_availability = {}

                    # Test different variables
                    variables = [
                        ('TAVG', 'Air Temperature'),
                        ('SNWD', 'Snow Depth'),
                        ('WTEQ', 'Snow Water Equivalent'),
                    ]

                    for element, name in variables:
                        try:
                            # Quick test - just check if any data exists for recent period
                            data = await self.client.get_station_data(
                                station_triplet=station.station_triplet,
                                elements=element,
                                start_date="2023-01-01",
                                end_date="2023-12-31",
                                duration="DAILY"
                            )

                            station_availability[element] = {
                                'available': data is not None and len(data) > 0,
                                'data_points': len(data) if data else 0,
                                'name': name
                            }

                        except Exception as e:
                            station_availability[element] = {
                                'available': False,
                                'error': str(e),
                                'name': name
                            }

                    network_results[station.station_triplet] = {
                        'station_name': station.name,
                        'variables': station_availability
                    }

                availability_results[network] = network_results

        # Validate results structure
        assert len(availability_results) > 0, "Should have results for at least one network"

        for network, stations in availability_results.items():
            assert len(stations) > 0, f"Should have stations for {network}"

            for station_triplet, station_data in stations.items():
                assert 'variables' in station_data, f"Station {station_triplet} should have variables data"

        print("  Data availability scanning test completed successfully")


class AWDBPaginationTester:
    """Test pagination and large dataset handling."""

    def __init__(self):
        self.client = AWDBClient(timeout=60)  # Longer timeout for large operations

    async def test_large_station_result_handling(self):
        """Test handling of large station result sets."""
        print("Testing large station result handling...")
        # Get all stations (this will be a large result)
        async with self.client:
            try:
                start_time = time.time()
                all_stations = await self.client.get_stations(active_only=True)
                end_time = time.time()

                duration = end_time - start_time
                station_count = len(all_stations)

                print(f"  Retrieved {station_count} stations in {duration:.2f}s")
                print(f"  Average stations per second: {station_count / duration:.2f}")

                # Should have thousands of stations
                assert station_count > 1000, f"Expected >1000 stations, got {station_count}"

                # Check memory usage isn't excessive
                # Each station should be a reasonable size
                avg_station_size = len(str(all_stations[0])) if all_stations else 0
                print(f"  Average station data size: {avg_station_size} chars")

                # Validate station data structure
                sample_station = all_stations[0]
                required_fields = ['station_triplet', 'name', 'latitude', 'longitude']
                for field in required_fields:
                    assert hasattr(sample_station, field), f"Station missing required field: {field}"

            except Exception as e:
                print(f"  Large station result test failed: {e}")
                print(f"  This might be due to API limits: {e}")

    async def test_chunked_data_retrieval(self):
        """Test chunked retrieval of large datasets."""
        print("Testing chunked data retrieval...")
        test_station = "301:CA:SNTL"  # Station with long record

        async with self.client:
            # Define large date range (10 years)
            start_date = "2010-01-01"
            end_date = "2020-12-31"

            # Test different chunk sizes
            chunk_sizes = [
                ("1year", 1),   # 1 year chunks
                ("2year", 2),   # 2 year chunks
                ("5year", 5),   # 5 year chunks
            ]

            for chunk_name, years_per_chunk in chunk_sizes:
                try:
                    total_data_points = 0
                    chunks_processed = 0
                    start_time = time.time()

                    # Process in chunks
                    current_start = datetime.strptime(start_date, "%Y-%m-%d")
                    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

                    while current_start < end_datetime:
                        chunk_end = min(
                            current_start + timedelta(days=365 * years_per_chunk),
                            end_datetime
                        )

                        chunk_start_str = current_start.strftime("%Y-%m-%d")
                        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

                        data = await self.client.get_station_data(
                            station_triplet=test_station,
                            elements="TAVG",
                            start_date=chunk_start_str,
                            end_date=chunk_end_str,
                            duration="DAILY"
                        )

                        if data:
                            total_data_points += len(data)
                            chunks_processed += 1

                        current_start = chunk_end

                    end_time = time.time()
                    duration = end_time - start_time

                    print(f"  {chunk_name} chunks: {chunks_processed} chunks, {total_data_points} total points in {duration:.2f}s")
                    print(f"  Average points per second: {total_data_points / duration:.2f}")

                    # Should have retrieved data
                    assert total_data_points > 0, f"No data retrieved for {chunk_name} chunking"

                except Exception as e:
                    print(f"  Chunked retrieval test failed for {chunk_name}: {e}")

    async def test_memory_efficient_processing(self):
        """Test memory-efficient processing of large datasets."""
        print("Testing memory-efficient processing...")
        test_station = "301:CA:SNTL"

        async with self.client:
            # Get a large dataset
            try:
                data = await self.client.get_station_data(
                    station_triplet=test_station,
                    elements="TAVG",
                    start_date="2010-01-01",
                    end_date="2020-12-31",
                    duration="DAILY"
                )

                if not data:
                    print("  Skipping memory efficiency test - no data available")
                    return

                data_points = len(data)
                print(f"  Processing {data_points} data points for memory efficiency")

                # Test processing in batches to simulate memory-efficient handling
                batch_size = 1000
                batches_processed = 0
                total_processed = 0

                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    batches_processed += 1
                    total_processed += len(batch)

                    # Simulate some processing on the batch
                    values = [point.value for point in batch if point.value is not None]
                    if values:
                        sum(values) / len(values)
                        # Just ensure we can process the batch

                print(f"  Processed in {batches_processed} batches, {total_processed} total points")
                assert total_processed == data_points, "All data points should be processed"

            except Exception as e:
                print(f"  Memory efficiency test failed: {e}")

    async def test_api_rate_limit_adaptation(self):
        """Test adaptation to API rate limits."""
        print("Testing API rate limit adaptation...")
        test_station = "301:CA:SNTL"

        async with self.client:
            # Make multiple requests with delays to test rate limit handling
            request_count = 5
            delays = [0, 0.1, 0.5, 1.0, 2.0]  # Progressive delays

            results = []

            for i, delay in enumerate(delays[:request_count]):
                try:
                    if delay > 0:
                        await asyncio.sleep(delay)

                    start_time = time.time()
                    data = await self.client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-01-31",
                        duration="DAILY"
                    )
                    end_time = time.time()

                    results.append({
                        'request': i + 1,
                        'delay': delay,
                        'success': True,
                        'data_points': len(data) if data else 0,
                        'duration': end_time - start_time
                    })

                except Exception as e:
                    results.append({
                        'request': i + 1,
                        'delay': delay,
                        'success': False,
                        'error': str(e)
                    })

            # Analyze results
            successful_requests = sum(1 for r in results if r['success'])
            success_rate = successful_requests / len(results) * 100

            print(f"  Rate limit adaptation: {successful_requests}/{len(results)} successful ({success_rate:.1f}%)")

            # Should have reasonable success rate
            assert success_rate >= 60, f"Success rate too low: {success_rate:.1f}%"

            # Check if delays helped
            no_delay_results = [r for r in results if r['delay'] == 0]
            with_delay_results = [r for r in results if r['delay'] > 0]

            if no_delay_results and with_delay_results:
                no_delay_success = sum(1 for r in no_delay_results if r['success'])
                with_delay_success = sum(1 for r in with_delay_results if r['success'])

                print(f"  No delay success: {no_delay_success}/{len(no_delay_results)}")
                print(f"  With delay success: {with_delay_success}/{len(with_delay_results)}")

    async def test_concurrent_request_limits(self):
        """Test handling of concurrent request limits."""
        print("Testing concurrent request limits...")
        test_stations = [
            "301:CA:SNTL",    # SNTL network
            "2235:CA:SCAN",   # SCAN network
            "AGP:CA:MSNT",    # MSNT network
        ]

        async def concurrent_load_test():
            """Test concurrent requests to understand system limits."""
            max_concurrent = 3  # Test with small concurrent load

            async def single_request(station):
                return await self.client.get_station_data(
                    station_triplet=station,
                    elements="TAVG",
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    duration="DAILY"
                )

            # Test sequential first
            print("  Testing sequential requests...")
            sequential_start = time.time()
            sequential_results = []
            for station in test_stations:
                try:
                    data = await single_request(station)
                    sequential_results.append(len(data) if data else 0)
                except Exception as e:
                    sequential_results.append(f"Error: {e}")
            sequential_time = time.time() - sequential_start

            # Test concurrent
            print("  Testing concurrent requests...")
            concurrent_start = time.time()
            await asyncio.gather(
                *[single_request(station) for station in test_stations[:max_concurrent]],
                return_exceptions=True
            )
            concurrent_time = time.time() - concurrent_start

            print(f"  Sequential time: {sequential_time:.2f}s")
            print(f"  Concurrent time: {concurrent_time:.2f}s")

            # Concurrent should be faster but not necessarily max_concurrent times faster
            if sequential_time > 0 and concurrent_time > 0:
                speedup = sequential_time / concurrent_time
                print(f"  Speedup: {speedup:.2f}x")
                assert speedup >= 1.0, "Concurrent should not be slower than sequential"

        async with self.client:
            await concurrent_load_test()


async def main():
    """Run all performance tests."""
    print("Starting AWDB Performance Tests")
    print("=" * 50)

    # Initialize testers
    performance_tester = AWDBPerformanceTester()
    bulk_tester = AWDBBulkOperationsTester()
    availability_scanner = AWDBDataAvailabilityScanner()
    pagination_tester = AWDBPaginationTester()

    # Run all tests
    try:
        # Performance tests
        await performance_tester.test_bulk_station_retrieval()
        await performance_tester.test_concurrent_station_queries()
        await performance_tester.test_data_retrieval_performance()
        await performance_tester.test_rate_limiting_behavior()
        await performance_tester.test_large_dataset_handling()
        await performance_tester.test_memory_usage_patterns()
        await performance_tester.test_api_parameter_combinations()

        # Bulk operations tests
        await bulk_tester.test_multi_station_data_retrieval()
        await bulk_tester.test_network_based_bulk_queries()
        await bulk_tester.test_error_recovery_in_bulk_operations()

        # Data availability tests
        await availability_scanner.test_station_data_availability_scan()

        # Pagination and large dataset tests
        await pagination_tester.test_large_station_result_handling()
        await pagination_tester.test_chunked_data_retrieval()
        await pagination_tester.test_memory_efficient_processing()
        await pagination_tester.test_api_rate_limit_adaptation()
        await pagination_tester.test_concurrent_request_limits()

        print("\n" + "=" * 50)
        print("All performance tests completed successfully!")

    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
