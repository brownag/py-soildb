"""Debug script to identify where fetch_mapunit_struct_by_point hangs."""

import asyncio
import sys
import time


# Add profiling decorator
def profile_async(name: str):
    """Decorator to profile async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            print(f"[{name}] Starting...", file=sys.stderr)
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                print(f"[{name}] Completed in {elapsed:.2f}s", file=sys.stderr)
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"[{name}] Failed after {elapsed:.2f}s: {e}", file=sys.stderr)
                raise
        return wrapper
    return decorator


# Patch the soildb functions to add profiling
async def main():
    """Test with profiling."""
    from soildb import (
        fetch_chorizon_by_cokey,
        fetch_component_by_mukey,
        get_mapunit_by_point,
    )
    from soildb.client import SDAClient

    # Wrap the functions with profiling
    original_get_mapunit = get_mapunit_by_point
    original_fetch_component = fetch_component_by_mukey
    original_fetch_chorizon = fetch_chorizon_by_cokey

    get_mapunit_by_point_profiled = profile_async("get_mapunit_by_point")(original_get_mapunit)
    fetch_component_by_mukey_profiled = profile_async("fetch_component_by_mukey")(original_fetch_component)
    fetch_chorizon_by_cokey_profiled = profile_async("fetch_chorizon_by_cokey")(original_fetch_chorizon)

    # Create a client
    client = SDAClient()

    latitude = 41.5
    longitude = -93.5

    print("\n=== Testing individual components ===\n", file=sys.stderr)

    # Step 1: Get map unit
    print("Step 1: Getting map unit...", file=sys.stderr)
    mu_response = await get_mapunit_by_point_profiled(longitude, latitude, client=client)
    mu_df = mu_response.to_pandas()

    if mu_df.empty:
        print("No map unit found!", file=sys.stderr)
        return

    mukey = str(mu_df.iloc[0]["mukey"])
    print(f"Found mukey: {mukey}", file=sys.stderr)

    # Step 2: Get components
    print(f"\nStep 2: Getting components for mukey {mukey}...", file=sys.stderr)
    comp_response = await fetch_component_by_mukey_profiled(
        mukey,
        columns=[
            "mukey",
            "cokey",
            "compname",
            "comppct_r",
            "majcompflag",
            "localphase",
            "drainagecl",
            "taxclname",
            "hydricrating",
            "compkind",
        ],
        client=client,
    )
    comp_df = comp_response.to_pandas()

    if comp_df.empty:
        print("No components found!", file=sys.stderr)
        return

    print(f"Found {len(comp_df)} components", file=sys.stderr)

    # Step 3: Get horizons for all components
    all_cokeys = comp_df["cokey"].tolist()
    print(f"\nStep 3: Getting horizons for {len(all_cokeys)} components...", file=sys.stderr)
    print(f"Cokeys: {all_cokeys}", file=sys.stderr)

    horizons_response = await fetch_chorizon_by_cokey_profiled(
        all_cokeys,
        columns=[
            "cokey",
            "chkey",
            "hzname",
            "hzdept_r",
            "hzdepb_r",
            "claytotal_l",
            "claytotal_r",
            "claytotal_h",
            "sandtotal_l",
            "sandtotal_r",
            "sandtotal_h",
            "om_l",
            "om_r",
            "om_h",
            "ph1to1h2o_l",
            "ph1to1h2o_r",
            "ph1to1h2o_h",
        ],
        client=client,
    )
    horizons_df = horizons_response.to_pandas()

    print(f"Found {len(horizons_df)} horizons", file=sys.stderr)

    # Step 4: Group by cokey
    print("\nStep 4: Grouping horizons by cokey...", file=sys.stderr)
    horizons_df["cokey"] = horizons_df["cokey"].astype(str)

    for cokey, comp_horizons_df in horizons_df.groupby("cokey"):
        print(f"  Component {cokey}: {len(comp_horizons_df)} horizons", file=sys.stderr)

        # Try iterating rows (this might be where it hangs)
        print("    Iterating rows...", file=sys.stderr)
        for _idx, h_row in comp_horizons_df.iterrows():
            # Check property access
            for prop in ["claytotal_r", "sandtotal_r", "om_r", "ph1to1h2o_r"]:
                if prop in h_row:
                    val = h_row[prop]
                    print(f"      {prop} = {val}", file=sys.stderr)

    print("\n=== All steps completed successfully ===\n", file=sys.stderr)
    await client.close()


if __name__ == "__main__":
    print("Starting debug script...", file=sys.stderr)
    asyncio.run(main())
    print("Debug script completed.", file=sys.stderr)
