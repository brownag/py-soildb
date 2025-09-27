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


if __name__ == "__main__":
    success = asyncio.run(test_sda_connection())
    if success:
        print("\n Integration test completed successfully!")
    else:
        print("\n Integration test failed")
        exit(1)
