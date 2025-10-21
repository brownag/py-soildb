"""
Performance benchmarking tests for AWDB client.

Provides detailed performance metrics and benchmarking capabilities.
"""

import asyncio
import statistics
import time

import pytest

from soildb.awdb.client import AWDBClient


class TestAWDBPerformanceBenchmarking:
    """Comprehensive performance benchmarking for AWDB client."""

    @pytest.fixture
    def client(self):
        """Create AWDB client for benchmarking."""
        return AWDBClient(timeout=60)

    @pytest.mark.asyncio
    async def test_station_retrieval_performance_benchmark(self, client):
        """Benchmark station retrieval performance across different scenarios."""
        benchmark_results = {}

        async with client:
            # Test different network sizes
            networks_to_test = ['SCAN', 'SNTL', 'SNOW']

            for network in networks_to_test:
                # Run multiple trials for statistical significance
                trial_times = []

                for trial in range(3):  # 3 trials per network
                    start_time = time.time()

                    try:
                        stations = await client.get_stations(
                            network_codes=[network],
                            active_only=True
                        )
                        end_time = time.time()

                        trial_times.append(end_time - start_time)
                        station_count = len(stations)

                    except Exception as e:
                        print(f"Trial {trial + 1} for {network} failed: {e}")
                        trial_times.append(float('inf'))  # Mark as failed
                        station_count = 0

                # Calculate statistics
                successful_trials = [t for t in trial_times if t != float('inf')]

                if successful_trials:
                    benchmark_results[network] = {
                        'mean_time': statistics.mean(successful_trials),
                        'median_time': statistics.median(successful_trials),
                        'min_time': min(successful_trials),
                        'max_time': max(successful_trials),
                        'success_rate': len(successful_trials) / len(trial_times),
                        'station_count': station_count,
                        'stations_per_second': station_count / statistics.mean(successful_trials) if successful_trials else 0
                    }

                    print(f"{network} Benchmark: {station_count} stations in {benchmark_results[network]['mean_time']:.2f}s "
                          f"({benchmark_results[network]['stations_per_second']:.1f} stations/sec)")
                else:
                    benchmark_results[network] = {'success_rate': 0}

        # Validate benchmark results
        assert len(benchmark_results) > 0, "No benchmark results collected"

        # At least one network should have successful benchmarks
        successful_networks = [n for n, r in benchmark_results.items() if r.get('success_rate', 0) > 0]
        assert len(successful_networks) > 0, "No networks had successful benchmarks"

        return benchmark_results

    @pytest.mark.asyncio
    async def test_data_retrieval_performance_benchmark(self, client):
        """Benchmark data retrieval performance for different scenarios."""
        # Use a known good station for consistent benchmarking
        test_station = "301:CA:SNTL"

        benchmark_scenarios = [
            {
                'name': '1_month_daily',
                'start_date': '2023-01-01',
                'end_date': '2023-01-31',
                'duration': 'DAILY',
                'expected_points': 31
            },
            {
                'name': '3_month_daily',
                'start_date': '2023-01-01',
                'end_date': '2023-03-31',
                'duration': 'DAILY',
                'expected_points': 90
            },
            {
                'name': '1_year_monthly',
                'start_date': '2022-01-01',
                'end_date': '2022-12-31',
                'duration': 'MONTHLY',
                'expected_points': 12
            }
        ]

        results = {}

        async with client:
            for scenario in benchmark_scenarios:
                trial_times = []

                for trial in range(3):  # 3 trials per scenario
                    start_time = time.time()

                    try:
                        data = await client.get_station_data(
                            station_triplet=test_station,
                            elements="TAVG",
                            start_date=scenario['start_date'],
                            end_date=scenario['end_date'],
                            duration=scenario['duration']
                        )
                        end_time = time.time()

                        trial_times.append(end_time - start_time)
                        data_points = len(data) if data else 0

                    except Exception as e:
                        print(f"Trial {trial + 1} for {scenario['name']} failed: {e}")
                        trial_times.append(float('inf'))
                        data_points = 0

                successful_trials = [t for t in trial_times if t != float('inf')]

                if successful_trials:
                    results[scenario['name']] = {
                        'mean_time': statistics.mean(successful_trials),
                        'median_time': statistics.median(successful_trials),
                        'data_points': data_points,
                        'points_per_second': data_points / statistics.mean(successful_trials) if successful_trials else 0,
                        'success_rate': len(successful_trials) / len(trial_times)
                    }

                    print(f"{scenario['name']}: {data_points} points in {results[scenario['name']]['mean_time']:.2f}s "
                          f"({results[scenario['name']]['points_per_second']:.1f} points/sec)")

        return results

    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential_performance(self, client):
        """Benchmark concurrent vs sequential request performance."""
        import asyncio

        test_stations = [
            "301:CA:SNTL",    # SNTL network
            "2235:CA:SCAN",   # SCAN network
            "AGP:CA:MSNT",    # MSNT network
        ]

        async def benchmark_concurrent_requests():
            """Benchmark concurrent request performance."""
            concurrent_times = []

            for trial in range(3):
                start_time = time.time()

                # Create concurrent tasks
                tasks = []
                for station in test_stations:
                    task = asyncio.create_task(self._async_get_station_data(client, station))
                    tasks.append(task)

                # Wait for all to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                concurrent_times.append(end_time - start_time)

                successful_requests = sum(1 for r in results if not isinstance(r, Exception))
                print(f"Concurrent trial {trial + 1}: {successful_requests}/{len(tasks)} successful in {concurrent_times[-1]:.2f}s")

            return concurrent_times

        async def benchmark_sequential_requests():
            """Benchmark sequential request performance."""
            sequential_times = []

            for trial in range(3):
                start_time = time.time()

                results = []
                for station in test_stations:
                    try:
                        data = await client.get_station_data(
                            station_triplet=station,
                            elements="TAVG",
                            start_date="2023-01-01",
                            end_date="2023-01-31",
                            duration="DAILY"
                        )
                        results.append(len(data) if data else 0)
                    except Exception as e:
                        results.append(f"Error: {e}")

                end_time = time.time()
                sequential_times.append(end_time - start_time)

                successful_requests = sum(1 for r in results if not isinstance(r, str) or not r.startswith("Error"))
                print(f"Sequential trial {trial + 1}: {successful_requests}/{len(results)} successful in {sequential_times[-1]:.2f}s")

            return sequential_times

        async with client:
            # Run sequential benchmark
            print("Running sequential benchmarks...")
            sequential_times = await benchmark_sequential_requests()

            # Run concurrent benchmark
            print("Running concurrent benchmarks...")
            concurrent_times = await benchmark_concurrent_requests()

            # Calculate statistics
            seq_mean = statistics.mean(sequential_times)
            conc_mean = statistics.mean(concurrent_times)
            speedup = seq_mean / conc_mean if conc_mean > 0 else 0

            results = {
                'sequential_mean': seq_mean,
                'concurrent_mean': conc_mean,
                'speedup': speedup,
                'efficiency': speedup / len(test_stations) if len(test_stations) > 0 else 0
            }

            print(f"Performance comparison: Sequential {seq_mean:.2f}s, Concurrent {conc_mean:.2f}s")
            print(f"Speedup: {speedup:.2f}x, Efficiency: {results['efficiency']:.2f}")

            return results

    async def _async_get_station_data(self, client, station_triplet):
        """Helper method for async data retrieval."""
        return await client.get_station_data(
            station_triplet=station_triplet,
            elements="TAVG",
            start_date="2023-01-01",
            end_date="2023-01-31",
            duration="DAILY"
        )

    @pytest.mark.asyncio
    async def test_memory_usage_during_large_operations(self, client):
        """Benchmark memory usage during large data operations."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory benchmarking")

        process = psutil.Process(os.getpid())

        async with client:
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform increasingly large operations
            operations = [
                {
                    'name': 'small_query',
                    'start_date': '2023-01-01',
                    'end_date': '2023-01-31'
                },
                {
                    'name': 'medium_query',
                    'start_date': '2023-01-01',
                    'end_date': '2023-06-30'
                },
                {
                    'name': 'large_query',
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31'
                }
            ]

            memory_usage = {}

            for op in operations:
                try:
                    data = await client.get_station_data(
                        station_triplet="301:CA:SNTL",
                        elements="TAVG",
                        start_date=op['start_date'],
                        end_date=op['end_date'],
                        duration="DAILY"
                    )

                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = current_memory - baseline_memory

                    memory_usage[op['name']] = {
                        'memory_mb': current_memory,
                        'delta_mb': memory_delta,
                        'data_points': len(data) if data else 0,
                        'memory_per_point_kb': (memory_delta * 1024) / len(data) if data and len(data) > 0 else 0
                    }

                    print(f"{op['name']}: {memory_usage[op['name']]['data_points']} points, "
                          f"{memory_usage[op['name']]['delta_mb']:.1f} MB memory delta")

                except Exception as e:
                    print(f"{op['name']} failed: {e}")
                    memory_usage[op['name']] = {'error': str(e)}

            return memory_usage

    @pytest.mark.asyncio
    async def test_api_rate_limiting_behavior_detailed(self, client):
        """Detailed analysis of API rate limiting behavior."""
        test_station = "301:CA:SNTL"

        async with client:
            # Test different request patterns
            patterns = [
                {'name': 'steady_1_per_sec', 'delays': [1.0] * 10},
                {'name': 'burst_then_pause', 'delays': [0.1] * 5 + [2.0] + [0.1] * 4},
                {'name': 'increasing_delays', 'delays': [0.5, 1.0, 1.5, 2.0, 3.0] * 2},
            ]

            results = {}

            for pattern in patterns:
                request_times = []
                successes = 0
                failures = 0

                print(f"Testing pattern: {pattern['name']}")

                for i, delay in enumerate(pattern['delays']):
                    if delay > 0:
                        await asyncio.sleep(delay)

                    start_time = time.time()

                    try:
                        data = await client.get_station_data(
                            station_triplet=test_station,
                            elements="TAVG",
                            start_date="2023-01-01",
                            end_date="2023-01-31",
                            duration="DAILY"
                        )
                        successes += 1
                        data_points = len(data) if data else 0

                    except Exception as e:
                        failures += 1
                        data_points = 0
                        if "rate limit" in str(e).lower():
                            print(f"  Rate limit hit on request {i+1}")

                    end_time = time.time()
                    request_times.append(end_time - start_time)

                success_rate = successes / (successes + failures) if (successes + failures) > 0 else 0
                avg_response_time = statistics.mean(request_times) if request_times else 0

                results[pattern['name']] = {
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'total_requests': successes + failures,
                    'successes': successes,
                    'failures': failures
                }

                print(f"  Result: {successes}/{successes + failures} successful ({success_rate:.1f}), "
                      f"avg {avg_response_time:.2f}s per request")

            return results

    @pytest.mark.asyncio
    async def test_network_efficiency_analysis(self, client):
        """Analyze network efficiency and data transfer patterns."""
        test_station = "301:CA:SNTL"

        async with client:
            # Test different response sizes and measure transfer efficiency
            test_cases = [
                {'duration': 'DAILY', 'periods': 30, 'name': '1_month_daily'},
                {'duration': 'DAILY', 'periods': 365, 'name': '1_year_daily'},
                {'duration': 'MONTHLY', 'periods': 12, 'name': '1_year_monthly'},
            ]

            efficiency_results = {}

            for case in test_cases:
                try:
                    # Calculate date range
                    if case['duration'] == 'DAILY':
                        days = case['periods']
                    else:  # MONTHLY
                        days = case['periods'] * 30  # Approximate

                    start_date = "2023-01-01"
                    end_date = f"2023-12-{min(days, 31):02d}"  # Keep within one year

                    start_time = time.time()

                    data = await client.get_station_data(
                        station_triplet=test_station,
                        elements="TAVG",
                        start_date=start_date,
                        end_date=end_date,
                        duration=case['duration']
                    )

                    end_time = time.time()

                    transfer_time = end_time - start_time
                    data_points = len(data) if data else 0

                    efficiency_results[case['name']] = {
                        'transfer_time': transfer_time,
                        'data_points': data_points,
                        'points_per_second': data_points / transfer_time if transfer_time > 0 else 0,
                        'bytes_per_point': len(str(data)) / data_points if data and data_points > 0 else 0
                    }

                    print(f"{case['name']}: {data_points} points in {transfer_time:.2f}s "
                          f"({efficiency_results[case['name']]['points_per_second']:.1f} points/sec)")

                except Exception as e:
                    print(f"{case['name']} failed: {e}")
                    efficiency_results[case['name']] = {'error': str(e)}

            return efficiency_results

    @pytest.mark.asyncio
    async def test_scalability_with_multiple_clients(self):
        """Test scalability when using multiple client instances."""
        # Test with different numbers of concurrent clients
        client_counts = [1, 2, 3]

        scalability_results = {}

        for num_clients in client_counts:
            clients = [AWDBClient(timeout=30) for _ in range(num_clients)]
            test_stations = ["301:CA:SNTL", "2235:CA:SCAN", "AGP:CA:MSNT"][:num_clients]

            start_time = time.time()

            results = []
            for i, client in enumerate(clients):
                async with client:
                    try:
                        data = await client.get_station_data(
                            station_triplet=test_stations[i],
                            elements="TAVG",
                            start_date="2023-01-01",
                            end_date="2023-01-31",
                            duration="DAILY"
                        )
                        results.append(len(data) if data else 0)
                    except Exception as e:
                        results.append(f"Error: {e}")

            end_time = time.time()
            total_time = end_time - start_time

            successful_requests = sum(1 for r in results if not isinstance(r, str) or not r.startswith("Error"))

            scalability_results[num_clients] = {
                'total_time': total_time,
                'successful_requests': successful_requests,
                'time_per_client': total_time / num_clients if num_clients > 0 else 0,
                'efficiency': successful_requests / num_clients if num_clients > 0 else 0
            }

            print(f"{num_clients} clients: {total_time:.2f}s total, "
                  f"{scalability_results[num_clients]['time_per_client']:.2f}s per client")

        return scalability_results

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, client):
        """Test performance impact of error recovery mechanisms."""
        # Mix of valid and invalid requests
        test_requests = [
            {"station": "301:CA:SNTL", "valid": True},   # Valid
            {"station": "INVALID:XX:TEST", "valid": False},  # Invalid
            {"station": "2235:CA:SCAN", "valid": True},  # Valid
            {"station": "NONEXISTENT:YY:ZZZ", "valid": False},  # Invalid
            {"station": "AGP:CA:MSNT", "valid": True},   # Valid
        ]

        async with client:
            start_time = time.time()

            results = []
            for req in test_requests:
                try:
                    data = await client.get_station_data(
                        station_triplet=req["station"],
                        elements="TAVG",
                        start_date="2023-01-01",
                        end_date="2023-01-31",
                        duration="DAILY"
                    )
                    results.append({
                        'station': req['station'],
                        'valid': req['valid'],
                        'success': True,
                        'data_points': len(data) if data else 0
                    })
                except Exception as e:
                    results.append({
                        'station': req['station'],
                        'valid': req['valid'],
                        'success': False,
                        'error': str(e)
                    })

            end_time = time.time()
            total_time = end_time - start_time

            # Analyze results
            valid_successes = sum(1 for r in results if r['valid'] and r['success'])
            invalid_failures = sum(1 for r in results if not r['valid'] and not r['success'])
            valid_failures = sum(1 for r in results if r['valid'] and not r['success'])
            invalid_successes = sum(1 for r in results if not r['valid'] and r['success'])

            recovery_results = {
                'total_time': total_time,
                'total_requests': len(results),
                'valid_successes': valid_successes,
                'invalid_failures': invalid_failures,
                'valid_failures': valid_failures,
                'invalid_successes': invalid_successes,
                'error_recovery_rate': invalid_failures / sum(1 for r in results if not r['valid']) if any(not r['valid'] for r in results) else 0
            }

            print(f"Error recovery: {valid_successes} valid successes, {invalid_failures} invalid properly failed")
            print(f"Total time: {total_time:.2f}s for {len(results)} requests")

            return recovery_results
