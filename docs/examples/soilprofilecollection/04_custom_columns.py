"""
Convert non-standard horizon data to SoilProfileCollection with custom columns.

Demonstrates mapping custom or aliased column names to SoilProfileCollection.
Requires optional soilprofilecollection package: pip install 'soildb[soil]'
"""

import asyncio

from soildb import Query, SDAClient, SDAResponse


def handle_missing_spc():
    """Handle missing soilprofilecollection dependency."""
    print("Optional dependency missing: soilprofilecollection")
    print("Install with: pip install 'soildb[soil]'")
    return None


async def main():
    """Convert custom horizon columns to SoilProfileCollection."""

    print("=" * 60)
    print("Example 4: Custom Column Configuration")
    print("=" * 60)
    print()

    async with SDAClient() as client:
        print("Querying horizon data with aliased column names...")
        query = (
            Query()
            .select(
                "cokey AS profile_id",
                "chkey AS horizon_id",
                "hzdept_r AS top_cm",
                "hzdepb_r AS bottom_cm",
                "claytotal_r AS clay_pct",
                "sandtotal_r AS sand_pct",
                "om_r AS organic_matter",
            )
            .from_("chorizon")
            .where("hzdept_r IS NOT NULL AND hzdepb_r IS NOT NULL")
            .order_by("cokey, hzdept_r")
            .limit(100)
        )

        response: SDAResponse = await client.execute(query)
        print(f"Retrieved {len(response)} horizon records")
        print(f"Columns: {response.columns}")
        print()

        if response.is_empty():
            print("No data retrieved.")
            return None

        # Convert using custom column mappings
        print("Converting with custom column mappings...")
        try:
            spc = response.to_soilprofilecollection(
                site_id_col="profile_id",
                hz_id_col="horizon_id",
                hz_top_col="top_cm",
                hz_bot_col="bottom_cm",
            )

            print("Conversion successful.")
            print()
            print("Results:")
            print(f"  Profiles: {len(spc)}")
            print(f"  Horizons: {len(spc.horizons)}")
            print(f"  Site ID name: {spc.idname}")
            print(f"  Horizon ID name: {spc.hzidname}")
            print()

            print("First 5 horizons:")
            cols = ["profile_id", "horizon_id", "top_cm", "bottom_cm", "clay_pct"]
            available = [c for c in cols if c in spc.horizons.columns]
            print(spc.horizons[available].head())

            return spc

        except ImportError:
            return handle_missing_spc()
        except Exception as e:
            print(f"Conversion error: {e}")
            return None


if __name__ == "__main__":
    spc = asyncio.run(main())
    print()
    print("=" * 60)
    if spc is not None:
        print("Example completed successfully!")
    else:
        print("Example finished (conversion skipped).")
    print("=" * 60)
