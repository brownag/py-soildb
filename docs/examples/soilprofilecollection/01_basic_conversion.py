"""
Convert SDA horizon data to a SoilProfileCollection object.

Requires:
- cokey, chkey, hzdept_r, hzdepb_r columns
- optional soilprofilecollection package: pip install 'soildb[soil]'
"""

import asyncio

from soildb import Query, SDAClient, SDAResponse


def handle_missing_spc():
    """Handle missing soilprofilecollection dependency."""
    print("Optional dependency missing: soilprofilecollection")
    print("Install with: pip install 'soildb[soil]'")
    return None


async def main():
    """Convert horizon data to SoilProfileCollection."""

    print("=" * 60)
    print("Example 1: Basic SoilProfileCollection Conversion")
    print("=" * 60)
    print()

    async with SDAClient() as client:
        # Query component horizons
        query = (
            Query()
            .select(
                "cokey",  # Component key (site ID)
                "chkey",  # Component horizon key (horizon ID)
                "hzdept_r",  # Horizon top depth (cm)
                "hzdepb_r",  # Horizon bottom depth (cm)
                "claytotal_r",  # Clay percentage
                "sandtotal_r",  # Sand percentage
                "om_r",  # Organic matter
            )
            .from_("chorizon")
            .order_by("cokey, hzdept_r")
            .limit(100)
        )

        print("Executing query...")
        print("  SELECT: cokey, chkey, hzdept_r, hzdepb_r, claytotal_r, ...")
        print("  FROM: chorizon")
        print("  LIMIT: 100")
        print()

        # Execute query
        response: SDAResponse = await client.execute(query)

        print(f"Query returned {len(response)} horizon records")
        print(f"Columns: {response.columns}")
        print()

        if response.is_empty():
            print("No data retrieved. Try adjusting the query.")
            return None

        # Convert to SoilProfileCollection
        print("Converting to SoilProfileCollection...")
        try:
            spc = response.to_soilprofilecollection()
        except ImportError:
            return handle_missing_spc()

        print("Conversion successful.")
        print()
        print("Results:")
        print(f"  Profiles (unique cokeys): {len(spc)}")
        print(f"  Horizon records: {len(spc.horizons)}")
        print(f"  ID column name: {spc.idname}")
        print(f"  Horizon ID column name: {spc.hzidname}")
        print()

        # Show first few horizons
        print("First 5 horizon records:")
        print(spc.horizons[["cokey", "chkey", "hzdept_r", "hzdepb_r"]].head())
        print()

        return spc


if __name__ == "__main__":
    spc = asyncio.run(main())
    print()
    print("=" * 60)
    if spc is not None:
        print("Example completed successfully!")
    else:
        print("Example finished (conversion skipped).")
    print("=" * 60)
