"""
AWDB Data Availability Assessment Script

This script assesses data availability across California stations for key variables
(air temperature, snow depth) at configurable temporal resolutions.

Current configuration: Daily data assessment (30-year period)
- For each station in California (limited to 5 for testing)
- For each variable (air temp, snow depth)
- Process 1 station at a time in chunks
- Return full year of daily dates for each year in 30-year interval
- Count days with available data out of each year

Configurable for different temporal resolutions and sampling strategies.

NOTE: This example demonstrates the data availability assessment framework.
In a real scenario, you would have network connectivity to the AWDB API.
For demonstration purposes, this script shows the structure and approach.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json

from soildb.awdb.client import AWDBClient
from soildb.awdb.models import StationInfo


# Key variables to assess (focused on air temperature and snow depth)
VARIABLES = {
    'air_temperature': {
        'elements': ['TAVG'],
        'name': 'Air Temperature Average'
    },
    'snow_depth': {
        'elements': ['SNWD', 'WTEQ'],
        'name': 'Snow Depth/Water Equivalent'
    }
}


async def get_days_with_data(client: AWDBClient, station_triplet: str,
                           elements: List[str], start_year: int, end_year: int) -> Dict[str, Dict[int, int]]:
    """
    Check data availability for a station across multiple years at daily resolution.

    Returns dict mapping element codes to dict mapping years to count of days with data.
    """
    days_with_data = defaultdict(lambda: defaultdict(int))

    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"

        for element in elements:
            try:
                # Get DAILY data for the full year
                data = await client.get_station_data(
                    station_triplet=station_triplet,
                    elements=element,
                    start_date=year_start,
                    end_date=year_end,
                    duration="DAILY"
                )

                # Count days with data for this year
                if data and len(data) > 0:
                    days_with_data[element][year] = len(data)

            except Exception as e:
                # Skip this element/year combination if there's an error
                continue

    return dict(days_with_data)


async def assess_station_data_availability(station: StationInfo,
                                         variables: Dict[str, Dict],
                                         start_year: int, end_year: int) -> Dict[str, Any]:
    """
    Assess data availability for a single station.
    """
    client = AWDBClient(timeout=30)
    station_results = {
        'station_triplet': station.station_triplet,
        'station_name': station.name,
        'network': station.network_code,
        'latitude': station.latitude,
        'longitude': station.longitude,
        'elevation': station.elevation,
        'variables': {}
    }

    for var_key, var_info in variables.items():
        days_data = await get_days_with_data(
            client=client,
            station_triplet=station.station_triplet,
            elements=var_info['elements'],
            start_year=start_year,
            end_year=end_year
        )

        # Calculate overall statistics
        total_days_with_data = 0
        total_days_possible = 0
        yearly_breakdown = {}

        for element, years_data in days_data.items():
            for year, days_count in years_data.items():
                total_days_with_data += days_count
                # Approximate days in year (ignoring leap years for simplicity)
                total_days_possible += 365
                yearly_breakdown[f"{year}_{element}"] = days_count

        # Find the element with the most data
        if days_data:
            best_element = max(days_data.keys(), key=lambda x: sum(days_data[x].values()))
            overall_percentage = round(total_days_with_data / total_days_possible * 100, 1) if total_days_possible > 0 else 0.0

            station_results['variables'][var_key] = {
                'element_used': best_element,
                'total_days_with_data': total_days_with_data,
                'total_days_possible': total_days_possible,
                'overall_data_percentage': overall_percentage,
                'yearly_data_by_element': days_data,
                'all_elements_tried': list(days_data.keys())
            }
        else:
            station_results['variables'][var_key] = {
                'element_used': None,
                'total_days_with_data': 0,
                'total_days_possible': total_days_possible,
                'overall_data_percentage': 0.0,
                'yearly_data_by_element': {},
                'all_elements_tried': []
            }

    return station_results


async def main():
    """
    Main assessment workflow for California stations.
    """
    print("AWDB Data Availability Assessment - California Stations")
    print("=" * 60)

    # Define assessment period (last 30 years, full daily data)
    end_year = datetime.now().year - 1  # Last complete year
    start_year = end_year - 29  # 30 years back

    print(f"Assessment Period: {start_year} - {end_year} ({end_year - start_year + 1} years)")
    print(f"Resolution: Daily data (full year for each year)")
    print(f"Processing: 1 station at a time")
    print(f"Variables: {', '.join([v['name'] for v in VARIABLES.values()])}")
    print()

    # Get all California stations
    print("Fetching California stations...")
    async with AWDBClient(timeout=60) as client:
        # First get all stations, then filter by state from station triplet (format: stationId:stateCode:networkCode)
        all_stations = await client.get_stations(state_codes=['CA'], active_only=True)
        ca_stations = all_stations  # Already filtered by state

    print(f"Found {len(ca_stations)} active stations in California")
    print()

    # Group stations by network for better organization
    stations_by_network = defaultdict(list)
    for station in ca_stations:
        stations_by_network[station.network_code].append(station)

    print("Stations by network:")
    for network, stations in sorted(stations_by_network.items()):
        print(f"  {network}: {len(stations)} stations")
    print()

    # Limit to 5 stations for testing
    max_stations = 5
    if len(ca_stations) > max_stations:
        print(f"Limiting assessment to first {max_stations} stations for testing...")
        ca_stations = ca_stations[:max_stations]
        print()

    # Assess data availability for all stations (1 station at a time)
    print("Assessing data availability...")
    print("(Processing 1 station at a time - this may take a while)")
    print()

    all_results = []
    total_stations = len(ca_stations)

    # Process stations one at a time
    for i, station in enumerate(ca_stations):
        print(f"Processing station {i+1}/{total_stations}: {station.station_triplet} ({station.name})")

        try:
            # Assess this single station
            result = await assess_station_data_availability(station, VARIABLES, start_year, end_year)
            all_results.append(result)

            # Quick progress indicator
            var_summary = []
            for var_key, var_data in result['variables'].items():
                if var_data.get('total_days_with_data', 0) > 0:
                    var_summary.append(f"{var_key[:4]}:{var_data['overall_data_percentage']:.1f}%")
            if var_summary:
                print(f"   {result['station_triplet']}: {', '.join(var_summary)}")
            else:
                print(f"    {result['station_triplet']}: No data found")

        except Exception as e:
            print(f"   Error assessing {station.station_triplet}: {e}")
            # Add error result
            all_results.append({
                'station_triplet': station.station_triplet,
                'station_name': station.name,
                'network': station.network_code,
                'error': str(e),
                'variables': {}
            })

    # Generate summary report
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 60)

    # Overall statistics
    total_stations_assessed = len(all_results)
    successful_assessments = len([r for r in all_results if 'error' not in r])

    print(f"Total stations assessed: {total_stations_assessed}")
    print(f"Successful assessments: {successful_assessments}")
    print(f"Failed assessments: {total_stations_assessed - successful_assessments}")
    print()

    # Statistics by network
    network_stats = defaultdict(lambda: defaultdict(int))

    for result in all_results:
        if 'error' in result:
            network_stats[result['network']]['errors'] += 1
            continue

        network = result['network']
        network_stats[network]['total'] += 1

        for var_key in VARIABLES.keys():
            var_data = result['variables'].get(var_key, {})
            days_data = var_data.get('total_days_with_data', 0)
            if days_data > 0:
                network_stats[network][f'{var_key}_stations'] += 1
                network_stats[network][f'{var_key}_total_days'] += days_data
                network_stats[network][f'{var_key}_max_days'] = max(
                    network_stats[network].get(f'{var_key}_max_days', 0), days_data
                )

    print("By Network:")
    print("-" * 40)
    for network in sorted(network_stats.keys()):
        stats = network_stats[network]
        print(f"\n{network} Network:")
        print(f"  Total stations: {stats['total'] + stats.get('errors', 0)}")
        if stats.get('errors', 0) > 0:
            print(f"  Assessment errors: {stats['errors']}")

        for var_key in VARIABLES.keys():
            var_stations = stats.get(f'{var_key}_stations', 0)
            if var_stations > 0:
                avg_days = stats[f'{var_key}_total_days'] / var_stations
                max_days = stats[f'{var_key}_max_days']
                print(f"  {var_key.replace('_', ' ').title()}:")
                print(f"    Stations with data: {var_stations}")
                print(f"    Avg days/station: {avg_days:.1f}")
                print(f"    Max days/station: {max_days}")

    # Save detailed results
    output_file = f"awdb_ca_data_availability_{start_year}_{end_year}.json"
    print(f"\nSaving detailed results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump({
            'assessment_period': {
                'start_year': start_year,
                'end_year': end_year,
                'total_years': end_year - start_year + 1
            },
            'variables_assessed': VARIABLES,
            'total_stations': len(ca_stations),
            'results': all_results
        }, f, indent=2, default=str)

    print("Assessment complete!")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Run the assessment
    asyncio.run(main())