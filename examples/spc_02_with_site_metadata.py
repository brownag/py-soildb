"""
Example 2: SoilProfileCollection with Site Metadata

This example shows how to include site-level metadata in the
SoilProfileCollection by merging horizon data with component information.

Use this when:
- You want component information in site slot
- You need properties like comppct_r, taxclname, component names, etc.
- You want to correlate soil properties with component metadata
"""

import asyncio
import pandas as pd
from soildb import SDAClient, Query


async def main():
    """Convert horizon data with component metadata."""
    
    print("=" * 60)
    print("Example 2: SoilProfileCollection with Site Metadata")
    print("=" * 60)
    print()
    
    async with SDAClient() as client:
        # Query with proper join to get both horizons and component metadata
        print("Querying horizon data with component metadata...")
        
        query = Query().select(
            "c.cokey",
            "ch.chkey",
            "ch.hzdept_r",
            "ch.hzdepb_r",
            "ch.claytotal_r",
            "ch.sandtotal_r",
            "ch.om_r",
            "c.compname",
            "c.comppct_r"
        ).from_("chorizon ch").inner_join(
            "component c", "ch.cokey = c.cokey"
        ).order_by(
            "cokey, hzdept_r"
        ).limit(100)
        
        response = await client.execute(query)
        print(f"  Retrieved {len(response)} records")
        print()
        
        if response.is_empty():
            print("⚠️  No data retrieved.")
            return None
        
        # Get all data as dict
        data = response.to_dict()
        
        # Create separate horizon and site DataFrames
        print("Creating site metadata DataFrame...")
        site_df = pd.DataFrame(data).drop_duplicates(subset=['cokey'])
        site_data = site_df[['cokey', 'compname', 'comppct_r']]
        
        print(f"  Site data: {site_data.shape[0]} unique components")
        print(f"  Columns: {site_data.columns.tolist()}")
        print()
        
        print("First 3 components:")
        print(site_data.head(3))
        print()
        
        # Convert to SoilProfileCollection with site data
        print("Converting to SoilProfileCollection with site metadata...")
        try:
            spc = response.to_soilprofilecollection(
                site_data=site_data,
                site_id_col="cokey"
            )
            
            print("✅ Conversion successful!")
            print()
            print("Results:")
            print(f"  Profiles: {len(spc)}")
            print(f"  Horizons: {len(spc.horizons)}")
            print(f"  Site columns: {spc.site.columns.tolist()}")
            print()
            
            # Show site data with component info
            print("Site data with component info:")
            display_cols = [col for col in ['cokey', 'compname', 'comppct_r'] if col in spc.site.columns]
            print(spc.site[display_cols].head())
            print()
            
            # Show horizons for first profile
            if len(spc) > 0:
                first_cokey = spc.site.iloc[0]["cokey"]
                print(f"Horizons for component {first_cokey}:")
                hz_subset = spc.horizons[spc.horizons["cokey"] == first_cokey]
                hz_cols = [col for col in ['chkey', 'hzdept_r', 'hzdepb_r', 'claytotal_r', 'om_r'] 
                          if col in hz_subset.columns]
                print(hz_subset[hz_cols])
            
            return spc
            
        except Exception as e:
            print(f"❌ Conversion error: {e}")
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
