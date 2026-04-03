"""
Example of using the soildb.awdb module to fetch SCAN/SNOTEL monitoring data.

This example demonstrates core AWDB functionality with minimal server load:
- Finding monitoring stations by location or criteria
- Retrieving soil moisture data from stations
- Working with multi-depth sensor data
"""

import asyncio

from soildb.awdb import (
    discover_stations,
    discover_stations_nearby,
    get_property_data_near,
    get_soil_moisture_by_depth,
    station_available_properties,
)


async def main():
    """Run AWDB examples with throttling to avoid server overload."""
    print("=" * 70)
    print("AWDB Monitoring Station Examples")
    print("=" * 70)

    # 1. Find nearby stations
    print("\n1. Finding SCAN stations near Denver, CO...")
    stations = await discover_stations_nearby(
        latitude=39.74,
        longitude=-104.99,
        max_distance_km=50,
        network_codes=["SCAN"],
        limit=2,  # Keep it small
    )

    if stations:
        station = stations[0]  # Use first station
        triplet = station["station_triplet"]
        print(f"   Found: {station['name']} ({triplet})")
        print(f"   Distance: {station.get('distance_km', 'N/A')} km")
    else:
        print("   No stations found, using example triplet")
        triplet = "2197:CO:SCAN"  # Fallback for demo

    # Brief pause between requests
    await asyncio.sleep(0.5)

    # 2. Get available properties for station
    print(f"\n2. Available properties at {triplet}...")
    try:
        properties = await station_available_properties(triplet)
        print(f"   {len(properties)} property types available")

        # Show first few
        soil_props = [
            p for p in properties if "soil" in p.get("property_name", "").lower()
        ]
        if soil_props:
            print("   Soil properties:")
            for prop in soil_props[:3]:
                print(f"     - {prop['property_name']}")
    except Exception as e:
        print(f"   Could not retrieve properties: {e}")

    await asyncio.sleep(0.5)

    # 3. Get soil moisture at specific depths
    print(f"\n3. Soil moisture data from {triplet}...")
    try:
        data = await get_soil_moisture_by_depth(
            triplet,
            depths_inches=[-2, -8, -20],
            start_date="2024-11-01",
            end_date="2024-11-05",
        )

        if data and data.get("depths"):
            for depth_inches, depth_data in sorted(data["depths"].items()):
                n_points = depth_data.get("n_data_points", 0)
                if n_points > 0:
                    print(f'   {depth_inches}" depth: {n_points} measurements')
                    if depth_data.get("data_points"):
                        sample = depth_data["data_points"][0]
                        val = sample.get("value", "?")
                        ts = sample.get("timestamp", "?")
                        print(f"      Sample: {ts[:10]} = {val}%")
        else:
            print("   No data available for date range")

    except Exception as e:
        print(f"   Error retrieving moisture data: {e}")
        print("   (AWDB API may have rate limit or data availability issues)")

    await asyncio.sleep(0.5)

    # 4. Get data by location (alternative approach)
    print("\n4. Get soil moisture near a point...")
    try:
        data = await get_property_data_near(
            latitude=39.74,
            longitude=-104.99,
            property_name="soil_moisture",
            start_date="2024-11-01",
            end_date="2024-11-03",
            max_distance_km=50,
            height_depth_inches=-4,
        )

        if data and data.get("data_points"):
            print(f"   Station: {data['site_name']}")
            print(f"   Distance: {data['metadata'].get('distance_km', '?')} km")
            print(f"   Data points: {len(data['data_points'])}")
            sample = data["data_points"][0]
            print(
                f"   Sample: {sample.get('timestamp', '?')[:10]} = {sample.get('value', '?')}%"
            )
        else:
            print("   No data found")

    except Exception as e:
        print(f"   Error: {e}")

    await asyncio.sleep(0.5)

    # 5. Filter stations by criteria
    print("\n5. Finding active SCAN stations in Colorado...")
    try:
        stations = await discover_stations(
            network_codes=["SCAN"],
            state_codes=["CO"],
            active_only=True,
            limit=3,
        )
        print(f"   Found {len(stations)} active stations")
        for st in stations[:2]:
            print(f"     - {st['name']} ({st['station_triplet']})")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("Examples completed")
    print("=" * 70)
    print("\nNote: AWDB is a free public resource. Be respectful of server load:")
    print("      - Space requests to avoid concurrent hammering")
    print("      - Reduce query frequency for production use")
    print("      - Implement backoff/retry logic on errors")


if __name__ == "__main__":
    asyncio.run(main())
