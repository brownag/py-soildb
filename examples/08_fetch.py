"""
Example usage of the fetch module for bulk data retrieval.

This demonstrates the generic fetch_by_keys function and specialized
helper functions for common SSURGO data access patterns.
"""

import asyncio

from soildb import (
    fetch_by_keys,
    fetch_chorizon_by_cokey,
    fetch_component_by_mukey,
    fetch_mapunit_polygon,
    fetch_survey_area_polygon,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
)


async def main():
    print("=== Bulk Data Fetching Examples ===\n")

    # 1. Basic key-based fetching
    print("1. Basic fetch by keys")
    print("-" * 40)

    # Get some mukeys for California survey areas
    mukeys = await get_mukey_by_areasymbol(["CA630", "CA632"])
    print(f"Found {len(mukeys)} map units in CA630 and CA632")

    # Take a sample for demonstration
    sample_mukeys = mukeys[:10] if len(mukeys) > 10 else mukeys
    print(f"Using sample of {len(sample_mukeys)} mukeys: {sample_mukeys[:5]}...")

    # Fetch map unit data
    response = await fetch_by_keys(sample_mukeys, "mapunit")
    df = response.to_pandas()
    print(f"\nFetched {len(df)} map units:")
    print(df[["mukey", "muname", "mukind"]].head() if not df.empty else "No data")

    print("\n" + "=" * 60 + "\n")

    # 2. Specialized polygon fetching
    print("2. Fetch map unit polygons with geometry")
    print("-" * 40)

    # Fetch polygons for a few mukeys
    poly_mukeys = sample_mukeys[:5]
    response = await fetch_mapunit_polygon(poly_mukeys)
    df = response.to_pandas()

    print(f"Fetched {len(df)} map unit polygons:")
    if not df.empty:
        print(df[["mukey", "musym", "muareaacres"]].head())
        print("\nGeometry sample (first 100 chars):")
        if "geometry" in df.columns:
            print(
                df["geometry"].iloc[0][:100] + "..." if len(df) > 0 else "No geometry"
            )

    print("\n" + "=" * 60 + "\n")

    # 3. Hierarchical data fetching (mukey -> cokey -> chkey)
    print("3. Hierarchical data fetching")
    print("-" * 40)

    # Start with a few mukeys
    hier_mukeys = sample_mukeys[:3]
    print(f"Starting with mukeys: {hier_mukeys}")

    # Get components for these mukeys
    response = await fetch_component_by_mukey(hier_mukeys)
    comp_df = response.to_pandas()
    print(f"\nFound {len(comp_df)} components:")
    if not df.empty:
        print(comp_df[["mukey", "cokey", "compname", "comppct_r"]].head())

    # Get cokeys for horizon fetching
    if not comp_df.empty:
        cokeys = comp_df["cokey"].tolist()[:5]  # Limit for demo
        print(f"\nFetching horizons for {len(cokeys)} components...")

        response = await fetch_chorizon_by_cokey(cokeys)
        hz_df = response.to_pandas()
        print(f"Found {len(hz_df)} horizons:")
        if not hz_df.empty:
            print(hz_df[["cokey", "chkey", "hzname", "hzdept_r", "hzdepb_r"]].head())

    print("\n" + "=" * 60 + "\n")

    # 4. Survey area polygons
    print("4. Survey area polygon fetching")
    print("-" * 40)

    # Fetch survey area boundaries
    areas = ["CA630", "CA632", "CA644"]
    response = await fetch_survey_area_polygon(areas)
    sa_df = response.to_pandas()

    print(f"Fetched {len(sa_df)} survey area polygons:")
    if not sa_df.empty:
        print(sa_df[["areasymbol", "spatialversion", "lkey"]].head())

    print("\n" + "=" * 60 + "\n")

    # 5. Custom fetch with specific columns
    print("5. Custom fetch with column selection")
    print("-" * 40)

    # Fetch only specific component columns
    custom_columns = ["mukey", "cokey", "compname", "comppct_r", "majcompflag"]
    response = await fetch_by_keys(
        hier_mukeys, "component", key_column="mukey", columns=custom_columns
    )
    custom_df = response.to_pandas()

    print(f"Fetched {len(custom_df)} components with custom columns:")
    print(f"Columns: {list(custom_df.columns) if not custom_df.empty else 'No data'}")
    if not custom_df.empty:
        print(custom_df.head())

    print("\n" + "=" * 60 + "\n")

    # 6. Pagination demonstration
    print("6. Pagination with large key lists")
    print("-" * 40)

    # Use all available mukeys to demonstrate chunking
    if len(mukeys) > 20:
        print(f"Fetching data for {len(mukeys)} mukeys using pagination...")

        # Use small chunk size to demonstrate pagination
        response = await fetch_by_keys(
            mukeys,
            "mapunit",
            columns=["mukey", "muname"],
            chunk_size=25,  # Small chunks for demo
        )
        paginated_df = response.to_pandas()

        print(
            f"Successfully fetched {len(paginated_df)} map units using chunked queries"
        )
        print("Sample results:")
        print(paginated_df.head())
    else:
        print("Not enough mukeys for pagination demo")

    print("\n" + "=" * 60 + "\n")

    # 7. Key extraction helpers
    print("7. Key extraction helper functions")
    print("-" * 40)

    # Get all mukeys from multiple survey areas
    test_areas = ["CA630", "CA632"]
    all_mukeys = await get_mukey_by_areasymbol(test_areas)
    print(f"Found {len(all_mukeys)} total mukeys in {test_areas}")

    # Get cokeys for a sample of mukeys
    sample_for_cokeys = all_mukeys[:5] if len(all_mukeys) > 5 else all_mukeys
    all_cokeys = await get_cokey_by_mukey(sample_for_cokeys)
    print(f"Found {len(all_cokeys)} components for {len(sample_for_cokeys)} map units")

    # Get only major components
    major_cokeys = await get_cokey_by_mukey(
        sample_for_cokeys, major_components_only=True
    )
    print(f"Found {len(major_cokeys)} major components")

    print("\n" + "=" * 60 + "\n")

    # 8. Error handling demonstration
    print("8. Error handling")
    print("-" * 40)

    try:
        # Try to fetch from a non-existent table
        await fetch_by_keys([123456], "nonexistent_table")
    except Exception as e:
        print(f"Expected error for unknown table: {type(e).__name__}: {e}")

    try:
        # Try to fetch with empty key list
        await fetch_by_keys([], "mapunit")
    except Exception as e:
        print(f"Expected error for empty keys: {type(e).__name__}: {e}")

    try:
        # Try to fetch with invalid keys (should return empty result, not error)
        response = await fetch_by_keys([999999999], "mapunit")
        empty_df = response.to_pandas()
        print(f"Invalid key result: {len(empty_df)} rows (expected 0)")
    except Exception as e:
        print(f"Unexpected error for invalid keys: {type(e).__name__}: {e}")


async def performance_demo():
    """Demonstrate performance characteristics."""
    print("\n=== Performance Demonstration ===\n")

    import time

    # Get a large set of mukeys
    print("Getting mukeys for performance test...")
    mukeys = await get_mukey_by_areasymbol(["CA630", "CA632", "CA644"])

    if len(mukeys) > 100:
        test_mukeys = mukeys[:100]  # Limit for demo

        print(f"Testing fetch performance with {len(test_mukeys)} mukeys")

        chunk_sizes = [10, 50, 100]

        for chunk_size in chunk_sizes:
            start_time = time.time()

            response = await fetch_by_keys(
                test_mukeys,
                "mapunit",
                columns=["mukey", "muname"],
                chunk_size=chunk_size,
            )

            end_time = time.time()
            df = response.to_pandas()

            print(
                f"Chunk size {chunk_size:3d}: {end_time - start_time:.2f}s for {len(df)} records"
            )
    else:
        print("Not enough mukeys for performance demo")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

    # Uncomment to run performance demo
    # asyncio.run(performance_demo())
