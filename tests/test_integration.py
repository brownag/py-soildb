#!/usr/bin/env python3
"""
Integration tests to verify SDA connectivity and functionality.
"""

import pytest

import soildb


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_sda_connection():
    """Test basic SDA connection and simple query."""
    print("Testing soildb SDA connection...")

    # basic query building
    query = soildb.QueryBuilder.available_survey_areas()
    assert query is not None
    print(f" Query built: {query.to_sql()[:60]}...")

    # client creation
    client = soildb.SDAClient(timeout=10.0)  # Short timeout for quick test
    print(" Client created")

    # real HTTP request
    print(" Testing SDA connection...")
    connected = await client.connect()
    assert connected, "SDA connection failed"
    print(" Connection test: SUCCESS")

    # Try a very simple query
    print(" Testing simple query...")
    simple_query = soildb.Query().select("COUNT(*)").from_("sacatalog").limit(1)
    response = await client.execute(simple_query)
    assert response is not None
    assert len(response) >= 0
    print(f" Query executed, got {len(response)} result rows")

    if not response.is_empty():
        data = response.to_dict()
        assert data is not None
        print(f" Survey areas count: {data[0] if data else 'N/A'}")

    # Test flexible query parameters
    await test_flexible_query_parameters()

    await client.close()
    print(" Client closed cleanly")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_flexible_query_parameters():
    """Test that the new flexible query parameters work with SDA."""
    print("Testing flexible query parameters...")

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
    assert query is not None
    response = await client.execute(query)
    assert response is not None
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
        assert response is not None
        print(f"  Related tables query: {len(response)} results")

    await client.close()
    print(" Flexible query parameters test completed")
