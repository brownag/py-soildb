import pytest

from soildb.client import SDAClient
from soildb.query import QueryBuilder


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_execute_sql():
    query = "SELECT TOP 1 areasymbol, areaname FROM sacatalog"
    async with SDAClient() as client:
        result = await client.execute(query)
        assert len(result) == 1
        assert "areasymbol" in result.columns
        assert "areaname" in result.columns


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(10)
async def test_query_builder_sql():
    query = QueryBuilder.from_sql("SELECT TOP 1 areasymbol, areaname FROM sacatalog")
    async with SDAClient() as client:
        result = await client.execute(query)
        assert len(result) == 1
        assert "areasymbol" in result.columns
        assert "areaname" in result.columns
