"""
Example 4: Custom Column Configuration

This example demonstrates using CustomColumnConfig to convert data with
user-defined column name mappings. Useful when working with non-standard
data structures or data from external sources that need harmonization.

Use this when:
- You have data with non-standard column names
- You want to map custom column names to SPC structure
- You need to work with external soil data formats
- You want to standardize multiple data sources
"""

import asyncio

from soildb import Query, SDAClient
from soildb.spc_presets import CustomColumnConfig


async def main():
    """Convert data using custom column configuration."""

    print("=" * 60)
    print("Example 4: Custom Column Configuration")
    print("=" * 60)
    print()

    async with SDAClient() as client:
        # Query standard SDA data
        print("Querying horizon data from SDA...")
        query = (
            Query()
            .select(
                "cokey",
                "chkey",
                "hzdept_r",
                "hzdepb_r",
                "claytotal_r",
                "sandtotal_r",
                "om_r",
            )
            .from_("chorizon")
            .order_by("cokey, hzdept_r")
            .limit(100)
        )

        response = await client.execute(query)
        print(f"  Retrieved {len(response)} horizon records")
        print()

        if response.is_empty():
            print("No data available. SDA service may be unavailable.")
            return None

        # Define custom column mappings
        print("Defining custom column configuration...")
        custom_config = CustomColumnConfig(
            description="Custom profile/horizon mapping with descriptive names",
            site_id_col="cokey",
            horizon_id_col="chkey",
            horizon_top_col="hzdept_r",
            horizon_bottom_col="hzdepb_r",
            optional_columns=["claytotal_r", "sandtotal_r", "om_r"],
        )

        print(f"  Site ID column: {custom_config.site_id_col}")
        print(f"  Horizon ID column: {custom_config.horizon_id_col}")
        print(
            f"  Depth columns: {custom_config.horizon_top_col} -> {custom_config.horizon_bottom_col}"
        )
        print(
            f"  Optional columns: {len(custom_config.optional_columns or [])} columns"
        )
        print()

        # Convert using custom config
        print("Converting to SoilProfileCollection with custom config...")
        try:
            spc = response.to_soilprofilecollection(
                preset=custom_config,
                validate_depths=False,  # Don't pre-validate; SoilProfileCollection will validate
                warn_on_defaults=False,
            )

            print("Conversion successful.")
            print()
            print("Results:")
            print(f"  Profiles: {len(spc)}")
            print(f"  Horizons: {len(spc.horizons)}")
            print()

            # Show structure
            print("Profile structure:")
            print(f"  Site ID name: {spc.idname}")
            print(f"  Horizon ID name: {spc.hzidname}")
            print(f"  Site columns: {spc.site.columns.tolist()}")
            print(f"  Horizon columns: {spc.horizons.columns.tolist()}")
            print()

            # Show sample data
            print("Sample horizons (first 5):")
            display_cols = [
                custom_config.site_id_col,
                custom_config.horizon_id_col,
                custom_config.horizon_top_col,
                custom_config.horizon_bottom_col,
                "claytotal_r",
            ]
            available = [col for col in display_cols if col in spc.horizons.columns]
            print(spc.horizons[available].head())

            return spc

        except Exception as e:
            print(f"Note: {type(e).__name__}: {e}")
            print()
            print("This may occur if:")
            print("  - Horizons have depth gaps (not all components have all depths)")
            print("  - Depth values are missing or invalid")
            print("  - Data quality issues exist in the SDA service")
            print()
            print("For production data, consider:")
            print("  - Filtering to components with complete horizon sequences")
            print("  - Cleaning/validating depth values before conversion")
            print("  - Using pandas/polars export for flexible data handling")
            print(f"   Available columns: {response.columns}")
            return None


if __name__ == "__main__":
    spc = asyncio.run(main())
    print()
    print("=" * 60)
    if spc is not None:
        print("Example completed successfully!")
    else:
        print("Example encountered an error.")
    print("=" * 60)
