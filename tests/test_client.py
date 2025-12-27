import pytest

from soildb import query_templates
from soildb.client import SDAClient


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
    query = query_templates.query_from_sql(
        "SELECT TOP 1 areasymbol, areaname FROM sacatalog"
    )
    async with SDAClient() as client:
        result = await client.execute(query)
        assert len(result) == 1
        assert "areasymbol" in result.columns
        assert "areaname" in result.columns
