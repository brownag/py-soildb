#!/usr/bin/env python3
"""
Integration tests for high-level functions in soildb.high_level.
"""

import pytest

import soildb
from soildb.high_level import (
    fetch_mapunit_struct_by_point,
    fetch_pedon_struct_by_bbox,
    fetch_pedon_struct_by_id,
)
from soildb.models import SoilMapUnit

# A known location in Iowa with soil data
TEST_LAT = 42.0
TEST_LON = -93.6

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
        # Check that at least one component has horizons
        has_horizons = any(
            len(comp.aggregate_horizons) > 0 for comp in map_unit.components
        )
        assert has_horizons, (
            f"No components have horizons. Components: {[len(comp.aggregate_horizons) for comp in map_unit.components]}"
        )
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
        TEST_LON - 0.1,
        TEST_LAT - 0.1,
        TEST_LON + 0.1,
        TEST_LAT + 0.1,
    )
    try:
        pedons = await fetch_pedon_struct_by_bbox(
            min_x, min_y, max_x, max_y, client=sda_client
        )
        assert isinstance(pedons, list)
        if pedons:
            from soildb.models import PedonData

            assert isinstance(pedons[0], PedonData)
            assert hasattr(pedons[0], "pedon_key")
            assert hasattr(pedons[0], "horizons")
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
        assert hasattr(pedon, "pedon_key")
        assert hasattr(pedon, "horizons")
        assert pedon.pedon_id == TEST_PEDON_ID
        assert len(pedon.horizons) > 0
        # Check if a corrected column is present
        horizon = pedon.horizons[0]
        assert (
            horizon.water_content_fifteen_bar is not None
            or horizon.water_content_third_bar is not None
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


@pytest.mark.asyncio
async def test_fetch_mapunit_struct_by_point_with_custom_columns(sda_client):
    """Test fetching a structured SoilMapUnit by point with custom columns."""
    print("Testing fetch_mapunit_struct_by_point with custom columns...")
    try:
        # Test with custom component and horizon columns (using known valid columns)
        map_unit = await fetch_mapunit_struct_by_point(
            TEST_LAT,
            TEST_LON,
            component_columns=[
                "cokey",
                "compname",
                "comppct_r",
                "majcompflag",
                "localphase",
                "drainagecl",
                "taxclname",
            ],
            horizon_columns=[
                "chkey",
                "hzname",
                "hzdept_r",
                "hzdepb_r",
                "claytotal_r",
                "sandtotal_r",
                "om_r",
                "ph1to1h2o_r",
            ],
            client=sda_client,
        )
        assert isinstance(map_unit, SoilMapUnit)
        assert map_unit.map_unit_key is not None
        assert len(map_unit.components) > 0

        # Check that extra fields are populated for components
        component = map_unit.components[0]
        assert hasattr(component, "extra_fields")
        # localphase, drainagecl, taxclname should be in extra_fields since they're not in default set
        extra_keys = list(component.extra_fields.keys())
        print(f"Component extra fields: {extra_keys}")

        # Check horizons
        if component.aggregate_horizons:
            horizon = component.aggregate_horizons[0]
            assert hasattr(horizon, "extra_fields")
            # om_r, ph1to1h2o_r should be in extra_fields since they're not in default set
            extra_keys = list(horizon.extra_fields.keys())
            print(f"Horizon extra fields: {extra_keys}")

        # Check metadata tracking
        assert "requested_columns" in map_unit.extra_fields
        assert "default_columns" in map_unit.extra_fields

        print("SUCCESS: fetch_mapunit_struct_by_point with custom columns worked.")
    except soildb.SDAConnectionError as e:
        pytest.fail(f"SDA Connection Error: {e}")
    except soildb.SDAMaintenanceError:
        pytest.skip("SDA is under maintenance.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_fetch_pedon_struct_by_id_with_custom_columns(sda_client):
    """Test fetching structured pedon data by ID with custom horizon columns."""
    print("Testing fetch_pedon_struct_by_id with custom columns...")
    try:
        # Test with custom horizon columns
        pedon = await fetch_pedon_struct_by_id(
            TEST_PEDON_ID,
            horizon_columns=[
                "layer_key",
                "layer_sequence",
                "hzn_desgn",
                "hzn_top",
                "hzn_bot",
                "sand_total",
                "silt_total",
                "clay_total",
                "texture_lab",
                "ph_h2o",
                "total_carbon_ncs",
                "organic_carbon_walkley_black",
                "caco3_lt_2_mm",
                "bulk_density_third_bar",
                "le_third_fifteen_lt2_mm",
                "water_retention_third_bar",
                "water_retention_15_bar",
                "cec7_r",
                "ecec_r",
                "sar_r",  # Custom columns
            ],
            client=sda_client,
        )
        from soildb.models import PedonData

        assert isinstance(pedon, PedonData)
        assert hasattr(pedon, "pedon_key")
        assert hasattr(pedon, "horizons")
        assert pedon.pedon_id == TEST_PEDON_ID
        assert len(pedon.horizons) > 0

        # Check that extra fields are populated for horizons
        horizon = pedon.horizons[0]
        assert hasattr(horizon, "extra_fields")
        # cec7_r, ecec_r, sar_r should be in extra_fields if they exist
        if (
            "cec7_r" in horizon.extra_fields
            or "ecec_r" in horizon.extra_fields
            or "sar_r" in horizon.extra_fields
        ):
            print("SUCCESS: Custom horizon columns found in extra_fields.")

        # Check metadata tracking
        assert "requested_columns" in pedon.extra_fields
        assert "default_columns" in pedon.extra_fields

        print("SUCCESS: fetch_pedon_struct_by_id with custom columns worked.")
    except soildb.SDAConnectionError as e:
        pytest.fail(f"SDA Connection Error: {e}")
    except soildb.SDAMaintenanceError:
        pytest.skip("SDA is under maintenance.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")
