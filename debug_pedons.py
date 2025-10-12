#!/usr/bin/env python3
"""
Debug script for fetch_pedon_struct_by_bbox.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from soildb.high_level import fetch_pedon_struct_by_bbox
from soildb.client import SDAClient

async def debug_pedons():
    # Test location from the failing test
    TEST_LAT = 38.5
    TEST_LON = -121.5

    min_x, min_y, max_x, max_y = (
        TEST_LON - 0.1, TEST_LAT - 0.1, TEST_LON + 0.1, TEST_LAT + 0.1
    )

    print(f"Fetching pedons in bbox {min_x}, {min_y}, {max_x}, {max_y}")

    client = SDAClient()
    try:
        pedons = await fetch_pedon_struct_by_bbox(min_x, min_y, max_x, max_y, client=client)

        print(f"Number of pedons: {len(pedons)}")

        if pedons:
            pedon = pedons[0]
            print(f"Type of pedon: {type(pedon)}")
            print(f"Keys: {pedon.keys()}")
            print(f"Site shape: {pedon['site'].shape}")
            print(f"Site columns: {list(pedon['site'].columns)}")
            print(f"Horizons shape: {pedon['horizons'].shape}")
            if not pedon['horizons'].empty:
                print(f"Horizons columns: {list(pedon['horizons'].columns)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_pedons())