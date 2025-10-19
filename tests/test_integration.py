#!/usr/bin/env python3
"""
Quick integration test to verify SDA connectivity.
"""

import asyncio

import pytest

import soildb


@pytest.mark.asyncio
async def test_sda_connection():
    """Test basic SDA connection and simple query."""
    print("Testing soildb SDA connection...")

    try:
        # basic query building
        query = soildb.QueryBuilder.available_survey_areas()
        print(f" Query built: {query.to_sql()[:60]}...")

        # client creation
        client = soildb.SDAClient(timeout=10.0)  # Short timeout for quick test
        print(" Client created")

        # real HTTP request
        print(" Testing SDA connection...")
        connected = await client.connect()
        print(f" Connection test: {'SUCCESS' if connected else 'FAILED'}")

        if connected:
            # Try a very simple query
            print(" Testing simple query...")
            simple_query = soildb.Query().select("COUNT(*)").from_("sacatalog").limit(1)
            response = await client.execute(simple_query)
            print(f" Query executed, got {len(response)} result rows")

            if not response.is_empty():
                data = response.to_dict()
                print(f" Survey areas count: {data[0] if data else 'N/A'}")

            # Test flexible query parameters
            success = await test_flexible_query_parameters()
            if not success:
                return False

        await client.close()
        print(" Client closed cleanly")

    except soildb.SDAMaintenanceError:
        print(" SDA service is under maintenance - this is expected occasionally")
        return True
    except soildb.SDAConnectionError as e:
        print(f" Connection error: {e}")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False

    return True


@pytest.mark.asyncio
async def test_flexible_query_parameters():
    """Test that the new flexible query parameters work with SDA."""
    print("Testing flexible query parameters...")

    try:
        client = soildb.SDAClient(timeout=10.0)

        # Test pedons_intersecting_bbox with custom column names
        print(" Testing pedons_intersecting_bbox with custom columns...")
        query = soildb.QueryBuilder.pedons_intersecting_bbox(
            -94.0,
            42.0,
            -93.0,
            43.0,
            columns=["pedon_key", "upedonid"],
            lon_column="longitude_decimal_degrees",
            lat_column="latitude_decimal_degrees",
        )
        response = await client.execute(query)
        print(f"  Custom columns query: {len(response)} results")

        # Test pedon_by_pedon_key with related tables (if we have data)
        if not response.is_empty():
            sample_pedon_key = response.to_dict()[0]["pedon_key"]
            print(
                f" Testing pedon_by_pedon_key with related tables for key: {sample_pedon_key}"
            )
            query = soildb.QueryBuilder.pedon_by_pedon_key(
                sample_pedon_key, related_tables=["lab_physical_properties"]
            )
            response = await client.execute(query)
            print(f"  Related tables query: {len(response)} results")

        await client.close()
        print(" Flexible query parameters test completed")

    except soildb.SDAMaintenanceError:
        print(" SDA service is under maintenance - skipping flexible query test")
        return True
    except Exception as e:
        print(f" Flexible query test error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_sda_connection())
    if success:
        print("\n Integration test completed successfully!")
    else:
        print("\n Integration test failed")
        exit(1)
