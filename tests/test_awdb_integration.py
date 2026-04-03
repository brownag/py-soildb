"""Integration tests for AWDB integration module."""

import pytest

try:
    from soildb.awdb_integration import (
        estimate_water_availability,
        get_component_water_properties,
        get_recommended_awdb_depths_for_soil,
        get_water_stress_categories,
        get_water_table_depth,
        should_use_awdb_for_water_analysis,
    )

    HAS_AWDB_INTEGRATION = True
except ImportError:
    HAS_AWDB_INTEGRATION = False


@pytest.mark.skipif(not HAS_AWDB_INTEGRATION, reason="awdb_integration not available")
class TestAWDBIntegrationFunctions:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_water_properties_callable(self):
        assert callable(get_component_water_properties)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_water_table_depth_callable(self):
        assert callable(get_water_table_depth)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_water_availability_callable(self):
        assert callable(estimate_water_availability)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stress_categories_callable(self):
        assert callable(get_water_stress_categories)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_use_awdb_callable(self):
        assert callable(should_use_awdb_for_water_analysis)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_depths_callable(self):
        assert callable(get_recommended_awdb_depths_for_soil)
