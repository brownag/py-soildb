"""
Example 3: Using Different Presets

This example demonstrates how to use different presets to convert
the same horizon data to SoilProfileCollection with different column
configurations.

Use this when:
- You want to select which columns to include
- You need different column sets for different analyses
- You're comparing different preset configurations
"""

import asyncio

from soildb import Query, SDAClient
from soildb.spc_presets import get_preset, list_presets


async def main():
    """Demonstrate different presets with the same data."""

    print("=" * 60)
    print("Example 3: Using Different Presets")
    print("=" * 60)
    print()

    async with SDAClient() as client:
        # Query standard horizon data
        print("Querying horizon data...")
        query = Query().select(
            "cokey",
            "chkey",
            "hzdept_r",
            "hzdepb_r",
            "claytotal_r",
            "sandtotal_r",
            "silttotal_r",
            "om_r"
        ).from_(
            "chorizon"
        ).order_by(
            "cokey, hzdept_r"
        ).limit(100)

        response = await client.execute(query)
        print(f"  Retrieved {len(response)} horizon records")
        print()

        if response.is_empty():
            print("No data available. SDA service may be unavailable.")
            return None

        # Show available presets
        print("Available presets:")
        presets = list_presets()
        for preset_name in presets:
            preset = get_preset(preset_name)
            print(f"  - {preset_name}")
            print(f"    Description: {preset.description}")
            if hasattr(preset, 'selected_columns'):
                print(f"    Columns: {len(preset.selected_columns)} columns")
        print()

        # Try converting with standard_sda preset
        print("Converting with 'standard_sda' preset...")
        try:
            spc_std = response.to_soilprofilecollection(
                preset="standard_sda",
                validate_depths=False,  # Disable depth validation for example data
                warn_on_defaults=False
            )

            print("Conversion successful.")
            print(f"  Profiles: {len(spc_std)}")
            print(f"  Horizons: {len(spc_std.horizons)}")
            print(f"  Horizon columns: {spc_std.horizons.columns.tolist()}")
            print()

            # Show statistics
            if len(spc_std.horizons) > 0:
                print("Data statistics:")
                if "claytotal_r" in spc_std.horizons.columns:
                    clay_mean = spc_std.horizons["claytotal_r"].mean()
                    clay_std = spc_std.horizons["claytotal_r"].std()
                    print(f"  Clay: mean={clay_mean:.1f}%, std={clay_std:.1f}%")

                if "sandtotal_r" in spc_std.horizons.columns:
                    sand_mean = spc_std.horizons["sandtotal_r"].mean()
                    print(f"  Sand: mean={sand_mean:.1f}%")

                if "om_r" in spc_std.horizons.columns:
                    om_mean = spc_std.horizons["om_r"].mean()
                    print(f"  Organic matter: mean={om_mean:.2f}%")
                print()

                # Show sample horizons
                print("Sample horizons (first 5):")
                sample_cols = [col for col in ['cokey', 'hzdept_r', 'hzdepb_r', 'claytotal_r', 'sandtotal_r']
                              if col in spc_std.horizons.columns]
                print(spc_std.horizons[sample_cols].head())

            return spc_std

        except Exception as e:
            print(f"Note: {type(e).__name__}: {e}")
            print()
            print("This may occur if:")
            print("  - Horizons have depth gaps in the queried data")
            print("  - Some depth values are missing or invalid")
            print("  - Data quality issues exist in the SDA service")
            print()
            print("For real-world use, consider:")
            print("  - Filtering to profiles with complete horizon coverage")
            print("  - Validating and cleaning depth values")
            print("  - Using the validation functions before conversion")
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
