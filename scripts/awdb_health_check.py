#!/usr/bin/env python3
"""
Simple AWDB API health check script.

This script performs basic connectivity and functionality tests for the AWDB API.
Useful for verifying API availability before running data pipelines.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, 'src')

from soildb.awdb.client import AWDBClient
from soildb.awdb.exceptions import AWDBError


async def check_basic_connectivity() -> Dict[str, Any]:
    """Test basic API connectivity."""
    print(" Testing basic connectivity...")

    try:
        async with AWDBClient(timeout=10) as client:
            start_time = time.time()

            # Test stations endpoint
            stations = await client.get_stations(active_only=True, network_codes=['SCAN'])
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': round(response_time, 2),
                'stations_found': len(stations),
                'error': None
            }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'response_time': None,
            'stations_found': 0,
            'error': str(e)
        }


async def check_data_retrieval() -> Dict[str, Any]:
    """Test data retrieval functionality."""
    print(" Testing data retrieval...")

    try:
        async with AWDBClient(timeout=15) as client:
            # Use a known reliable station
            test_station = "301:CA:SNTL"  # Adin Mtn SNTL station
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            start_time = time.time()
            data = await client.get_station_data(
                station_triplet=test_station,
                elements="TAVG",
                start_date=start_date,
                end_date=end_date,
                duration="DAILY"
            )
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': round(response_time, 2),
                'data_points': len(data) if data else 0,
                'station_tested': test_station,
                'error': None
            }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'response_time': None,
            'data_points': 0,
            'station_tested': "301:CA:SNTL",
            'error': str(e)
        }


async def check_reference_data() -> Dict[str, Any]:
    """Test reference data retrieval."""
    print(" Testing reference data...")

    try:
        async with AWDBClient(timeout=10) as client:
            start_time = time.time()

            ref_data = await client.get_reference_data(['elements', 'networks'])
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': round(response_time, 2),
                'elements_count': len(getattr(ref_data, 'elements', []) or []),
                'networks_count': len(getattr(ref_data, 'networks', []) or []),
                'error': None
            }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'response_time': None,
            'elements_count': 0,
            'networks_count': 0,
            'error': str(e)
        }


async def run_health_check() -> Dict[str, Any]:
    """Run complete health check suite."""
    print(" AWDB API Health Check")
    print("=" * 50)

    results = {
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Run all checks
    checks = [
        ('connectivity', check_basic_connectivity),
        ('data_retrieval', check_data_retrieval),
        ('reference_data', check_reference_data),
    ]

    all_healthy = True

    for check_name, check_func in checks:
        print(f"\n{'' * 20}")
        result = await check_func()
        results['checks'][check_name] = result

        status_emoji = "" if result['status'] == 'healthy' else ""
        print(f"{status_emoji} {check_name}: {result['status']}")

        if result['response_time']:
            print(f"   Response time: {result['response_time']}s")

        if result['error']:
            print(f"   Error: {result['error']}")
            all_healthy = False

        # Print additional metrics
        if check_name == 'connectivity' and result['status'] == 'healthy':
            print(f"   Stations found: {result['stations_found']}")
        elif check_name == 'data_retrieval' and result['status'] == 'healthy':
            print(f"   Data points: {result['data_points']}")
        elif check_name == 'reference_data' and result['status'] == 'healthy':
            print(f"   Elements: {result['elements_count']}, Networks: {result['networks_count']}")

    # Overall status
    results['overall_status'] = 'healthy' if all_healthy else 'unhealthy'

    print(f"\n{'' * 20}")
    overall_emoji = "" if all_healthy else ""
    print(f"{overall_emoji} Overall Status: {results['overall_status']}")

    if not all_healthy:
        print("\n  Some checks failed. Check the AWDB API status or network connectivity.")
        return results  # Don't exit in async context
    else:
        print("\n All checks passed! AWDB API is healthy.")

    return results


async def main():
    """Main entry point for the health check script."""
    try:
        results = await run_health_check()

        # Exit with appropriate code
        if results['overall_status'] == 'healthy':
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n  Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Unexpected error during health check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())