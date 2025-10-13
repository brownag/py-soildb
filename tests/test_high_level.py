#!/usr/bin/env python3
"""
Integration tests for high-level functions in soildb.high_level.
"""

import pytest
import pandas as pd
import soildb
from soildb.high_level import (
    fetch_mapunit_struct_by_point,
    fetch_pedon_struct_by_bbox,
    fetch_pedon_struct_by_id,
)
from soildb.models import SoilMapUnit, PedonData

# A known location in California
TEST_LAT = 38.5
TEST_LON = -121.5

# A known lab pedon ID
TEST_PEDON_ID = "S1999NY061001"


@pytest.mark.asyncio
async def test_fetch_mapunit_struct_by_point(sda_client):
    """Test fetching a structured SoilMapUnit by point."""
    print("Testing fetch_mapunit_struct_by_point...")
    try:
        map_unit = await fetch_mapunit_struct_by_point(
            TEST_LAT, TEST_LON, client=sda_client
        )
        assert isinstance(map_unit, SoilMapUnit)
        assert map_unit.map_unit_key is not None
        assert len(map_unit.components) > 0
        assert len(map_unit.components[0].aggregate_horizons) > 0
        print("SUCCESS: fetch_mapunit_struct_by_point returned a valid SoilMapUnit.")
    except soildb.SDAConnectionError as e:
        pytest.fail(f"SDA Connection Error: {e}")
    except soildb.SDAMaintenanceError:
        pytest.skip("SDA is under maintenance.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_fetch_pedon_struct_by_bbox(sda_client):
    """Test fetching structured pedon data by bounding box."""
    print("Testing fetch_pedon_struct_by_bbox...")
    min_x, min_y, max_x, max_y = (
        TEST_LON - 0.1, TEST_LAT - 0.1, TEST_LON + 0.1, TEST_LAT + 0.1
    )
    try:
        pedons = await fetch_pedon_struct_by_bbox(
            min_x, min_y, max_x, max_y, client=sda_client
        )
        assert isinstance(pedons, list)
        if pedons:
            from soildb.models import PedonData
            assert isinstance(pedons[0], PedonData)
            assert hasattr(pedons[0], 'pedon_key')
            assert hasattr(pedons[0], 'horizons')
            assert len(pedons[0].horizons) > 0
            print(f"SUCCESS: fetch_pedon_struct_by_bbox returned {len(pedons)} pedons.")
        else:
            print("No pedons found in the given bbox, which is a valid result.")
    except soildb.SDAConnectionError as e:
        pytest.fail(f"SDA Connection Error: {e}")
    except soildb.SDAMaintenanceError:
        pytest.skip("SDA is under maintenance.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_fetch_pedon_struct_by_id(sda_client):
    """Test fetching structured pedon data by ID."""
    print("Testing fetch_pedon_struct_by_id...")
    try:
        pedon = await fetch_pedon_struct_by_id(TEST_PEDON_ID, client=sda_client)
        from soildb.models import PedonData
        assert isinstance(pedon, PedonData)
        assert hasattr(pedon, 'pedon_key')
        assert hasattr(pedon, 'horizons')
        assert pedon.pedon_id == TEST_PEDON_ID
        assert len(pedon.horizons) > 0
        # Check if a corrected column is present
        horizon = pedon.horizons[0]
        assert (
            horizon.water_content_fifteen_bar is not None
            or horizon.water_content_third_bar is not None
            or horizon.water_content_tenth_bar is not None
        )
        print(
            f"SUCCESS: fetch_pedon_struct_by_id returned a valid PedonData object for ID {TEST_PEDON_ID}."
        )
    except soildb.SDAConnectionError as e:
        pytest.fail(f"SDA Connection Error: {e}")
    except soildb.SDAMaintenanceError:
        pytest.skip("SDA is under maintenance.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")