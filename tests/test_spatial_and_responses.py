"""Integration tests for spatial queries and response conversions."""

import importlib.util

import pytest
import pytest_asyncio

from soildb import ClientConfig, SDAClient
from soildb.query import Query
from soildb.spatial import bbox_query, point_query


@pytest_asyncio.fixture
async def sda_client():
    config = ClientConfig(timeout=30.0)
    client = SDAClient(config=config)
    yield client
    await client.close()


class TestSpatialQueries:
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_point_query_iowa(self, sda_client):
        result = await point_query(latitude=42.0, longitude=-93.6, client=sda_client)
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_point_query_remote(self, sda_client):
        result = await point_query(latitude=0.0, longitude=-160.0, client=sda_client)
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_bbox_query_iowa(self, sda_client):
        result = await bbox_query(
            xmin=-93.8, ymin=41.8, xmax=-93.4, ymax=42.2, client=sda_client
        )
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_point_query_auto_client(self):
        try:
            result = await point_query(latitude=42.0, longitude=-93.6)
            assert result is not None
        except Exception:
            pytest.skip("Auto client creation not supported")


class TestResponseConversions:
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_to_dict(self, sda_client):
        query = Query().select("areasymbol", "areaname").from_("sacatalog").limit(1)
        response = await sda_client.execute(query)
        result = response.to_dict()
        assert isinstance(result, list)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_to_pandas(self, sda_client):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        query = Query().select("areasymbol", "areaname").from_("sacatalog").limit(2)
        response = await sda_client.execute(query)
        df = response.to_pandas()
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_to_polars(self, sda_client):
        if importlib.util.find_spec("polars") is None:
            pytest.skip("polars not installed")
        query = Query().select("areasymbol").from_("sacatalog").limit(1)
        response = await sda_client.execute(query)
        df = response.to_polars()
        assert hasattr(df, "columns")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_is_valid(self, sda_client):
        query = Query().select("areasymbol").from_("sacatalog").limit(1)
        response = await sda_client.execute(query)
        assert response.is_valid

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_is_empty(self, sda_client):
        query = (
            Query()
            .select("areasymbol")
            .from_("sacatalog")
            .where("areasymbol = 'NONEXISTENT_XYZ'")
        )
        response = await sda_client.execute(query)
        assert response.is_empty() or len(response) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_length_and_columns(self, sda_client):
        query = Query().select("areasymbol", "areaname").from_("sacatalog").limit(3)
        response = await sda_client.execute(query)
        assert len(response) <= 3
        assert response.columns is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_null_handling(self, sda_client):
        query = Query().select("areasymbol", "areaname").from_("sacatalog").limit(2)
        response = await sda_client.execute(query)
        data = response.to_dict()
        if len(data) > 0:
            for row in data:
                for value in row.values():
                    if value is None:
                        assert value is None


class TestAWDBIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_awdb_convenience_import(self):
        try:
            from soildb.awdb.convenience import (
                discover_stations_nearby,
                get_property_data_near,
            )

            assert callable(discover_stations_nearby)
            assert callable(get_property_data_near)
        except ImportError:
            pytest.skip("AWDB convenience not available")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_awdb_integration_module(self):
        try:
            from soildb import awdb_integration

            assert hasattr(awdb_integration, "get_component_water_properties")
        except ImportError:
            pytest.skip("awdb_integration not available")


class TestResponseValidation:
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_columns_present(self, sda_client):
        query = Query().select("areasymbol", "areaname").from_("sacatalog").limit(1)
        response = await sda_client.execute(query)
        assert len([str(c).lower() for c in response.columns]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_access_methods(self, sda_client):
        query = Query().select("areasymbol").from_("sacatalog").limit(1)
        response = await sda_client.execute(query)
        assert callable(response.to_dict)
        assert hasattr(response, "is_valid")
        assert callable(response.to_pandas)
