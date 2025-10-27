"""
Example 1: Basic SoilProfileCollection Conversion

This example demonstrates the simplest way to convert SDA query results
to a SoilProfileCollection object using the default preset.

Use this when:
- Working with standard SDA horizon data
- You have cokey, chkey, hzdept_r, hzdepb_r columns
- You want quick conversion without configuration
"""

import asyncio
from soildb import SDAClient, Query


async def main():
    """Convert horizon data to SoilProfileCollection."""
    
    print("=" * 60)
    print("Example 1: Basic SoilProfileCollection Conversion")
    print("=" * 60)
    print()
    
    async with SDAClient() as client:
        # Query component horizons
        query = Query().select(
            "cokey",          # Component key (site ID)
            "chkey",          # Component horizon key (horizon ID)
            "hzdept_r",       # Horizon top depth (cm)
            "hzdepb_r",       # Horizon bottom depth (cm)
            "claytotal_r",    # Clay percentage
            "sandtotal_r",    # Sand percentage
            "om_r"            # Organic matter
        ).from_(
            "chorizon"
        ).order_by(
            "cokey, hzdept_r"
        ).limit(100)
        
        print("Executing query...")
        print(f"  SELECT: cokey, chkey, hzdept_r, hzdepb_r, claytotal_r, ...")
        print(f"  FROM: chorizon")
        print(f"  LIMIT: 100")
        print()
        
        # Execute query
        response = await client.execute(query)
        
        print(f"Query returned {len(response)} horizon records")
        print(f"Columns: {response.columns}")
        print()
        
        if response.is_empty():
            print("⚠️  No data retrieved. Try adjusting the query.")
            return None
        
        # Convert to SoilProfileCollection using default preset
        print("Converting to SoilProfileCollection...")
        spc = response.to_soilprofilecollection()
        
        print("✅ Conversion successful!")
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
        print("Example encountered an error.")
    print("=" * 60)
