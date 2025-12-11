"""
soildb QueryBuilder usage examples.
"""

import asyncio

import soildb


async def basic_point_query():
    """Get soil components at a specific point."""
    print("=== Point Query ===")

    client = soildb.SDAClient()
    query = soildb.QueryBuilder.components_at_point(-93.6, 42.0)
    response = await client.execute(query)
    df = response.to_pandas()

    print(f"Found {len(df)} components")
    if not df.empty:
        print("Component names:", df["compname"].tolist()[:5])
    print()


async def basic_area_query():
    """Get map units for a survey area."""
    print("=== Area Query ===")

    client = soildb.SDAClient()
    query = soildb.QueryBuilder.mapunits_by_legend("IA015")
    response = await client.execute(query)
    df = response.to_pandas()

    print(f"Found {len(df)} map units in IA015")
    if not df.empty:
        print("Sample map units:")
        print(df[["mukey", "muname"]].head())
    print()


async def basic_spatial_query():
    """Simple spatial query with bounding box."""
    print("=== Spatial Query ===")

    client = soildb.SDAClient()
    query = soildb.QueryBuilder.mapunits_intersecting_bbox(-93.7, 42.0, -93.6, 42.1)
    response = await client.execute(query)
    df = response.to_pandas()

    print(f"Found {len(df)} map units in bounding box")
    if not df.empty:
        print("Sample spatial data:")
        print(df[["mukey", "musym", "muname"]].head())
    print()


async def basic_bulk_fetch():
    """Fetch data for multiple keys."""
    print("=== Bulk Fetch ===")

    client = soildb.SDAClient()
    query = soildb.QueryBuilder.mapunits_by_legend("IA015")
    area_response = await client.execute(query)
    area_df = area_response.to_pandas()
    sample_mukeys = area_df["mukey"].tolist()[:5]  # Just first 5

    print(f"Fetching detailed data for {len(sample_mukeys)} map units...")

    # Fetch component data for these mukeys
    response = await soildb.fetch_by_keys(
        keys=sample_mukeys,
        table="component",
        key_column="mukey",
        columns=["mukey", "cokey", "compname", "comppct_r"],
    )
    df = response.to_pandas()

    print(f"Found {len(df)} components")
    if not df.empty:
        print("Sample components:")
        print(df.head())
    print()


async def main():
    """Run all basic examples."""
    print("soildb Basic Examples")
    print("=" * 50)

    await basic_point_query()
    await basic_area_query()
    await basic_spatial_query()
    await basic_bulk_fetch()

    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
