"""
Example of using the soildb.awdb module to fetch data for various station types.

This example demonstrates the enhanced AWDB client with:
- Semantic property names instead of cryptic element codes
- Comprehensive sensor metadata reporting
- Improved station filtering and data retrieval
- Support for all AWDB API parameters and response fields
- Automatic sensor selection and dynamic property discovery
"""

import asyncio

from soildb.awdb.convenience import (
    find_stations_by_criteria,
    get_monitoring_station_data,
    get_nearby_stations,
    list_available_variables,
)


async def fetch_and_print_data(client, network_code, property_name, start_date="2023-01-01", end_date="2023-01-10", height_depth_inches=None, state_codes=None, max_stations=3):
    """
    Fetches and prints data for a given network and property.
    Updated to use proper API element format for soil properties.
    """
    print(f"\n--- Example: Fetching {property_name} for {network_code} network ---")

    # Get stations for this network (now using API filtering)
    stations = await client.get_stations(network_codes=[network_code], state_codes=state_codes, active_only=True)

    if not stations:
        print(f"No active stations found for network: {network_code}")
        return

    print(f"Found {len(stations)} active {network_code} stations total.")
    stations = stations[:max_stations]  # Limit to first few stations
    print(f"Testing data retrieval for first {len(stations)} {network_code} stations.")

    # Try to get data from each station directly
    for station in stations:
        try:
            print(f"\nTrying station: {station.name} ({station.station_triplet})")

            # Get the element code for this property
            from soildb.awdb.convenience import (
                PROPERTY_ELEMENT_MAP,
                SOIL_PROPERTIES,
                build_soil_element_string,
            )
            if property_name not in PROPERTY_ELEMENT_MAP:
                print(f"Unknown property: {property_name}")
                continue

            element_code = PROPERTY_ELEMENT_MAP[property_name]

            # Build proper element string for soil properties
            if property_name in SOIL_PROPERTIES:
                if height_depth_inches is None:
                    print(f"   Soil property '{property_name}' requires height_depth_inches parameter")
                    continue
                element_string = build_soil_element_string(element_code, height_depth_inches, ordinal=1)
                print(f"   Using element: {element_string}")
            else:
                element_string = element_code

            # Get data directly from this specific station
            raw_data = await client.get_station_data(
                station.station_triplet,
                element_string,
                start_date,
                end_date,
                ordinal=1,
            )

            if raw_data:
                print(f" Successfully fetched {len(raw_data)} data points for {station.name}")
                print(f"   Sample: {raw_data[0]}")
                break  # Found data, stop trying other stations
            else:
                print(f"   No data available for {station.name} in date range")

        except Exception as e:
            print(f"    Failed to fetch data for {station.name}: {e}")

    else:
        print(f"\n Could not find any {network_code} station with data for the specified criteria.")
        if network_code == "SNOW":
            print("   NOTE: SNOW network data is manually collected and may not be available for all stations or date ranges.")


async def run_awdb_example():
    """
    Fetches data for various station types using the soildb.awdb module.
    Demonstrates automatic sensor selection and sensor metadata features.
    """
    print("Fetching station data from AWDB...")
    print("Note: This example demonstrates automatic sensor selection and")
    print("sensor metadata discovery features.\n")

    # Example 1: Automatic sensor selection for soil moisture
    print("=== Example 1: Automatic Sensor Selection ===")
    print("Finding nearby stations with sensor metadata...")

    try:
        stations = await get_nearby_stations(
            latitude=39.7392,  # Denver, CO
            longitude=-104.9903,
            max_distance_km=50,
            network_codes=['SCAN'],
            limit=3,
            include_sensor_metadata=True
        )

        print(f"Found {len(stations)} SCAN stations with sensor metadata:")
        for station in stations:
            print(f"\nStation: {station['name']} ({station['station_triplet']})")
            print(f"Distance: {station['distance_km']} km")

            if 'sensor_metadata' in station and station['sensor_metadata']:
                sensors = station['sensor_metadata']
                print("Available sensors:")
                for prop_name, sensor_list in sensors.items():
                    print(f"  {prop_name}: {len(sensor_list)} sensors")
                    for sensor in sensor_list[:2]:  # Show first 2
                        depth = sensor.get('height_depth_inches', 'N/A')
                        ordinal = sensor.get('ordinal', 'N/A')
                        print(f"    Depth: {depth}\", Ordinal: {ordinal}")
            else:
                print("  No sensor metadata available")

    except Exception as e:
        print(f"Error getting nearby stations: {e}")

    # Example 2: Auto sensor selection for soil moisture
    print("\n\n=== Example 2: Auto Sensor Selection for Soil Moisture ===")
    print("Automatically selecting best available soil moisture sensor...")

    try:
        result = await get_monitoring_station_data(
            latitude=39.7392,
            longitude=-104.9903,
            property_name='soil_moisture',
            start_date='2023-10-01',
            end_date='2023-10-05',
            auto_select_sensor=True  # This is the key new feature!
        )

        print("✅ Success with auto sensor selection!")
        print(f"Station: {result['site_name']} ({result['site_id']})")
        print(f"Distance: {result['metadata']['distance_km']} km")
        print(f"Auto-selected element: {result['metadata']['element_string']}")
        print(f"Data points: {len(result['data_points'])}")
        print(f"Unit: {result.get('unit', 'N/A')}")

        if result['data_points']:
            sample = result['data_points'][0]
            print(f"Sample: {sample['timestamp'][:10]} = {sample['value']} {result.get('unit', '')}")

    except Exception as e:
        print(f"Auto sensor selection failed: {e}")

    # Example 3: Traditional manual sensor specification still works
    print("\n\n=== Example 3: Manual Sensor Specification (Traditional) ===")
    print("Manually specifying sensor depth...")

    try:
        result = await get_monitoring_station_data(
            latitude=39.7392,
            longitude=-104.9903,
            property_name='air_temp',
            start_date='2023-01-01',
            end_date='2023-01-05',
            height_depth_inches=10,  # 10 feet above ground
            auto_select_sensor=False  # Explicitly disable auto-selection
        )

        print("✅ Success with manual sensor specification!")
        print(f"Station: {result['site_name']} ({result['site_id']})")
        print(f"Element used: {result['metadata']['element_string']}")
        print(f"Data points: {len(result['data_points'])}")

    except Exception as e:
        print(f"Manual sensor specification failed: {e}")

    # Example 4: Advanced station filtering with sensor metadata
    print("\n\n=== Example 4: Advanced Station Filtering ===")
    print("Finding SCAN stations in Colorado with sensor metadata...")

    try:
        stations = await find_stations_by_criteria(
            network_codes=['SCAN'],
            state_codes=['CO'],
            active_only=True,
            limit=5,
            include_sensor_metadata=True
        )

        print(f"Found {len(stations)} SCAN stations with sensor metadata:")
        for station in stations:
            sensor_types = []
            if 'sensor_metadata' in station:
                sensor_types = list(station['sensor_metadata'].keys())

            print(f"  {station['name']} ({station['station_triplet']}): {len(sensor_types)} sensor types")
            if sensor_types:
                print(f"    Available: {', '.join(sensor_types[:3])}")  # Show first 3

    except Exception as e:
        print(f"Advanced filtering failed: {e}")

    # Example 5: Comprehensive sensor inventory for a station
    print("\n\n=== Example 5: Complete Sensor Inventory ===")
    print("Getting comprehensive sensor inventory for a SCAN station...")

    try:
        # Use a station we know has comprehensive sensors
        station_triplet = "2197:CO:SCAN"  # CPER station

        variables = await list_available_variables(station_triplet)

        print(f"Station {station_triplet} has {len(variables)} different variable types:")
        print()

        # Group by known vs unknown
        known_vars = [v for v in variables if not v['property_name'].startswith('unknown_')]
        unknown_vars = [v for v in variables if v['property_name'].startswith('unknown_')]

        print(f"✅ Known properties ({len(known_vars)}):")
        for var in known_vars[:8]:  # Show first 8
            prop_name = var['property_name']
            element_code = var['element_code']
            unit = var['unit']
            sensor_count = len(var['sensors'])
            print(f"  {prop_name} ({element_code}): {unit}, {sensor_count} sensors")

        if len(known_vars) > 8:
            print(f"  ... and {len(known_vars) - 8} more known properties")

        if unknown_vars:
            print(f"\n❓ Unknown properties ({len(unknown_vars)}):")
            for var in unknown_vars[:3]:  # Show first 3
                prop_name = var['property_name']
                element_code = var['element_code']
                sensor_count = len(var['sensors'])
                print(f"  {prop_name} ({element_code}): {sensor_count} sensors")

    except Exception as e:
        print(f"Comprehensive inventory failed: {e}")

    print("\n\n=== Summary ===")
    print("✅ Automatic sensor selection eliminates guesswork")
    print("✅ Sensor metadata provides complete station inventories")
    print("✅ Backward compatibility maintained for manual specification")
    print("✅ Advanced filtering works with sensor metadata")
    print("✅ Comprehensive property mapping covers all AWDB sensor types")
    print("✅ Dynamic property discovery handles unknown elements gracefully")
    print("✅ Enhanced AWDB client supports all API parameters and features")


async def fetch_soil_moisture_example(client):
    """
    Demonstrate soil moisture data retrieval using proper API format.
    """
    from soildb.awdb.convenience import (
        get_monitoring_station_data,
        get_soil_moisture_data,
        get_station_soil_depths,
    )

    station_triplet = "2057:AL:SCAN"

    try:
        print(f"Station: {station_triplet} (Alabama Hills SCAN)")

        # Get available soil depths
        depths = await get_station_soil_depths(station_triplet, "soil_moisture")
        print(f"Available soil moisture depths: {len(depths)}")

        for depth in depths[:5]:  # Show first 5 depths
            d_inches = depth['height_depth_inches']
            ordinal = depth['ordinal']
            element = depth['element_string']
            print(f"  {d_inches}\": ordinal {ordinal}, element: {element}")

        if depths:
            # Demonstrate single depth query using get_monitoring_station_data
            print("\n--- Single Depth Query (using get_monitoring_station_data) ---")
            try:
                # Get data for 20-inch depth
                result = await get_monitoring_station_data(
                    latitude=36.6,  # Approximate location of Alabama Hills
                    longitude=-118.1,
                    property_name="soil_moisture",
                    start_date="2024-09-01",
                    end_date="2024-10-01",
                    height_depth_inches=-20,  # 20 inches deep
                    max_distance_km=100  # Allow wider search
                )

                print(f" Found station: {result['site_name']} ({result['site_id']})")
                print(f"   Distance: {result['metadata']['distance_km']} km")
                print(f"   Data points: {result['metadata']['n_data_points']}")
                print(f"   Element used: {result['metadata']['element_string']}")

                if result['data_points']:
                    sample = result['data_points'][0]
                    print(f"   Sample: {sample['timestamp'][:10]} = {sample['value']}% volumetric moisture")

            except Exception as e:
                print(f" Single depth query failed: {e}")

            # Get soil moisture data for multiple depths
            print("\n--- Multi-Depth Query (using get_soil_moisture_data) ---")

            soil_data = await get_soil_moisture_data(
                station_triplet,
                depths_inches=[-40, -20, -8, -4, -2],  # Specific depths
                start_date="2024-09-01",
                end_date="2024-10-01"
            )

            print(f"Retrieved data for {len(soil_data['depths'])} depths:")

            for depth_inches, depth_data in soil_data['depths'].items():
                n_points = depth_data['n_data_points']
                element = depth_data['element_string']
                if n_points > 0:
                    print(f" Depth {depth_inches}\": {n_points} points")
                    # Show sample value
                    if depth_data['data_points']:
                        sample = depth_data['data_points'][0]
                        print(f"   Sample: {sample['timestamp'][:10]} = {sample['value']}%")
                else:
                    error = depth_data.get('error', 'No data')
                    print(f" Depth {depth_inches}\": {error}")

        else:
            print("   No soil moisture sensors found")

    except Exception as e:
        print(f" Error in soil moisture example: {e}")


async def fetch_data_for_known_station(client, station_triplet, element_code, description):
    """
    Fetch data for a specific known station and element.
    """
    try:
        print(f"Trying station: {station_triplet} ({description})")

        # Get data directly from this specific station
        raw_data = await client.get_station_data(
            station_triplet,
            element_code,
            "2023-01-01",
            "2023-01-05"
        )

        if raw_data:
            print(f" Successfully fetched {len(raw_data)} data points")
            print(f"   Sample: {raw_data[0]}")
        else:
            print("   No data available for date range")

    except Exception as e:
        print(f" Error fetching data: {e}")


if __name__ == "__main__":
    asyncio.run(run_awdb_example())
