"""
soildb Query Templates Examples

Demonstrates using query_templates for common query patterns.
query_templates provides pre-built, discoverable SQL queries for standard tasks.

See also:
- Workflows: Custom Queries → ../workflows.qmd#custom-queries
- API Reference → ../api.qmd
"""

import asyncio

from soildb import SDAClient, fetch_by_keys, query_templates


async def basic_point_query():
    """Get soil components at a specific point using query_templates."""
    print("=== Point Query ===")

    async with SDAClient() as client:
        query = query_templates.query_components_at_point(-93.6, 42.0)
        response = await client.execute(query)
        df = response.to_pandas()

    print(f"Found {len(df)} components")
    if not df.empty:
        print("Component names:", df["compname"].tolist()[:5])
    print()


async def basic_area_query():
    """Get map units for a survey area using query_templates."""
    print("=== Area Query ===")

    async with SDAClient() as client:
        query = query_templates.query_mapunits_by_legend("IA015")
        response = await client.execute(query)
        df = response.to_pandas()

    print(f"Found {len(df)} map units in IA015")
    if not df.empty:
        print("Sample map units:")
        print(df[["mukey", "muname"]].head())
    print()


async def basic_spatial_query():
    """Spatial query using bounding box."""
    print("=== Spatial Query (Bbox) ===")

    async with SDAClient() as client:
        query = query_templates.query_mapunits_intersecting_bbox(
            -93.7, 42.0, -93.6, 42.1
        )
        response = await client.execute(query)
        df = response.to_pandas()

    print(f"Found {len(df)} map units in bounding box")
    if not df.empty:
        print("Sample spatial data:")
        print(df[["mukey", "musym", "muname"]].head())
    print()


async def bulk_fetch_components():
    """Fetch components for multiple map units using fetch_by_keys."""
    print("=== Bulk Fetch Components ===")

    async with SDAClient() as client:
        # First, get some mukeys for a survey area
        query = query_templates.query_mapunits_by_legend("IA015")
        area_response = await client.execute(query)
        area_df = area_response.to_pandas()

    sample_mukeys = area_df["mukey"].tolist()[:5]  # Just first 5
    print(f"Fetching components for {len(sample_mukeys)} map units...")

    # Fetch component data for these mukeys using fetch_by_keys
    response = await fetch_by_keys(
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


async def discover_survey_areas():
    """List available survey areas using query_templates."""
    print("=== Available Survey Areas ===")

    async with SDAClient() as client:
        query = query_templates.query_available_survey_areas()
        response = await client.execute(query)
        df = response.to_pandas()

    print(f"Found {len(df)} survey areas")
    if not df.empty:
        print("Sample areas (first 10):")
        print(df[["areasymbol", "areaname"]].head(10).to_string())
    print()


async def main():
    """Run all query template examples."""
    print("soildb Query Templates Examples")
    print("=" * 50)

    await basic_point_query()
    await basic_area_query()
    await basic_spatial_query()
    await bulk_fetch_components()
    await discover_survey_areas()

    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
