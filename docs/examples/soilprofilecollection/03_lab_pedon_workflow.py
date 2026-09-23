"""
Convert lab pedon characterization data to SoilProfileCollection.

Queries lab pedon horizons from SDA (lab_layer table) and maps
pedon and layer keys to SoilProfileCollection identifiers.
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
    """Query lab pedons and convert to SoilProfileCollection."""

    print("=" * 60)
    print("Example 3: Lab Pedon Workflow")
    print("=" * 60)
    print()

    async with SDAClient() as client:
        print("Querying lab pedon horizon data...")
        query = (
            Query()
            .select(
                "pedon_key",
                "layer_key",
                "hzn_top",
                "hzn_bot",
                "hzn_desgn",
                "clay_total",
                "sand_total",
                "silt_total",
            )
            .from_("lab_layer")
            .where(
                "layer_type = 'horizon' AND hzn_top IS NOT NULL AND hzn_bot IS NOT NULL"
            )
            .order_by("pedon_key, hzn_top")
            .limit(100)
        )

        response: SDAResponse = await client.execute(query)
        print(f"Retrieved {len(response)} lab horizon records")
        print()

        if response.is_empty():
            print("No data retrieved.")
            return None

        print("Converting lab pedons to SoilProfileCollection...")
        try:
            spc = response.to_soilprofilecollection(
                site_id_col="pedon_key",
                hz_id_col="layer_key",
                hz_top_col="hzn_top",
                hz_bot_col="hzn_bot",
            )
            print("Conversion successful.")
            print(f"  Profiles: {len(spc)}")
            print(f"  Horizons: {len(spc.horizons)}")
            print()

            if len(spc.horizons) > 0:
                print("First 5 lab horizon records:")
                sample_cols = [
                    col
                    for col in [
                        "pedon_key",
                        "layer_key",
                        "hzn_top",
                        "hzn_bot",
                        "hzn_desgn",
                        "clay_total",
                    ]
                    if col in spc.horizons.columns
                ]
                print(spc.horizons[sample_cols].head())

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
