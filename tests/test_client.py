from typing import Optional

import pytest

from soildb import query_templates
from soildb.client import SDAClient
from soildb.utils import add_sync_version


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


def test_convenience_function_without_explicit_client_closes_client():
    """
    Test that calling a decorated convenience function via .sync() without an explicit
    client properly creates and closes an internal client.

    This test verifies that the @add_sync_version decorator's client lifecycle management
    works correctly.
    """
    close_tracker = []

    class MockClient:
        async def close(self):
            close_tracker.append("closed")

    @add_sync_version
    async def test_convenience_func(
        symbol: str,
        client: Optional[MockClient] = None,
    ) -> str:
        # Simulates a convenience function that uses the client
        return f"result for {symbol}"

    # Call sync version without explicit client
    result = test_convenience_func.sync("IA015")

    # Verify the function executed successfully
    assert result == "result for IA015"

    # Verify the internal client was closed
    assert len(close_tracker) == 1, (
        "Client should be created and closed automatically when not provided to sync()"
    )
