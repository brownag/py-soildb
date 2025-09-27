#!/usr/bin/env python3
"""
Basic usage example for soildb package.

This example demonstrates the core functionality of soildb including:
- Connecting to SDA service
- Querying soil data by location
- Getting map units for survey areas
- Converting results to pandas DataFrame
"""

import asyncio

import soildb


async def main():
    """Run basic soildb examples."""

    print("soildb Basic Usage Example")
    print("=" * 40)

    # Create a client
    print("\n1. Creating SDA client...")
    client = soildb.SDAClient()

    try:
        # Test connection
        print("2. Testing connection to SDA service...")
        await client.connect()
        print("Connection successful!")

        # Example 1: Get soil data for a specific point
        print("\n3. Getting soil data for Ames, Iowa...")
        longitude, latitude = -93.6319, 42.0308  # Ames, Iowa

        soil_data = await soildb.get_mapunit_by_point(longitude, latitude, client)
        print(f"Found {len(soil_data)} soil records at ({longitude}, {latitude})")

        if not soil_data.is_empty():
            # Show first few records
            records = soil_data.to_dict()
            print("\nSample soil data:")
            for _i, record in enumerate(records[:3]):
                comp_name = record.get("compname", "Unknown")
                hz_name = record.get("hzname", "Unknown")
                sand = record.get("sandtotal_r", "N/A")
                print(f"  • {comp_name} - {hz_name} (Sand: {sand}%)")

            if len(records) > 3:
                print(f"  ... and {len(records) - 3} more records")

        # Example 2: Get map units for a survey area
        print("\n4. Getting map units for Story County, Iowa (IA169)...")
        areasymbol = "IA169"

        mapunits = await soildb.get_mapunit_by_areasymbol(areasymbol, client)
        print(f"Found {len(mapunits)} map units in {areasymbol}")

        if not mapunits.is_empty():
            # Show sample map units
            mu_records = mapunits.to_dict()
            print("\nSample map units:")
            for record in mu_records[:5]:
                musym = record.get("musym", "Unknown")
                muname = record.get("muname", "Unknown")
                print(f"  • {musym}: {muname}")

            if len(mu_records) > 5:
                print(f"  ... and {len(mu_records) - 5} more map units")

        # Example 3: Custom query
        print("\n5. Running custom query...")
        query = (
            soildb.Query()
            .select("COUNT(*) as total_components")
            .from_("component c")
            .inner_join("mapunit m", "c.mukey = m.mukey")
            .inner_join("legend l", "m.lkey = l.lkey")
            .where(f"l.areasymbol = '{areasymbol}'")
        )

        result = await client.execute(query)
        if not result.is_empty():
            total = result.data[0][0]
            print(f"Total components in {areasymbol}: {total}")

        # Example 4: Get survey areas
        print("\n6. Listing Iowa survey areas...")
        all_areas = await soildb.list_survey_areas(client)
        iowa_areas = [area for area in all_areas if area.startswith("IA")]
        print(f"Found {len(iowa_areas)} Iowa survey areas")
        print(f"   Examples: {', '.join(iowa_areas[:10])}")

        # Example 5: Bounding box query
        print("\n7. Getting map units in bounding box around Ames...")
        bbox_data = await soildb.get_mapunit_by_bbox(-93.7, 42.0, -93.6, 42.1, client)
        print(f"Found {len(bbox_data)} map units in bounding box")

    except soildb.SDAMaintenanceError:
        print("SDA service is currently under maintenance. Please try again later.")
    except soildb.SDAConnectionError as e:
        print(f"Connection error: {e}")
    except soildb.SDAQueryError as e:
        print(f"Query error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await client.close()
        print("\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main())
