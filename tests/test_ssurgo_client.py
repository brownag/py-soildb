"""
Tests for SSURGOClient generic SSURGO query builder.

Tests verify that SSURGOClient can construct proper SQL queries
for SSURGO tables and execute them via any backend.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from soildb.backends.ssurgo_client import SSURGOClient
from soildb.response import SDAResponse


class MockBackend:
    """Mock backend for testing SSURGOClient."""

    def __init__(self):
        self.last_query = None
        self.mock_response = MagicMock(spec=SDAResponse)
        self.mock_response.is_empty.return_value = False

    async def execute(self, sql: str) -> SDAResponse:
        """Mock execute method."""
        self.last_query = sql
        return self.mock_response

    async def get_tables(self) -> list:
        """Mock get_tables method."""
        return ["mapunit", "component", "chorizon", "legend"]

    async def get_columns(self, table: str) -> dict:
        """Mock get_columns method."""
        return {"id": "integer", "name": "text"}


class TestSSURGOClientInitialization:
    """Tests for SSURGOClient initialization."""

    def test_ssurgo_client_initialization(self):
        """SSURGOClient should initialize with a backend."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        assert client.backend is mock_backend

    def test_ssurgo_client_has_table_constants(self):
        """SSURGOClient should define table constants."""
        assert SSURGOClient.MAPUNIT_TABLE == "mapunit"
        assert SSURGOClient.COMPONENT_TABLE == "component"
        assert SSURGOClient.CHORIZON_TABLE == "chorizon"


class TestSSURGOClientQueryBuilding:
    """Tests for SQL query building."""

    def test_build_query_with_where_clause(self):
        """_build_query should use custom WHERE clause when provided."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query(
            "mapunit",
            where_clause="muname LIKE 'Miami%'",
        )

        assert "WHERE muname LIKE 'Miami%'" in sql

    def test_build_query_with_single_key(self):
        """_build_query should build IN condition for single value."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("mapunit", primary_key="101")

        assert "WHERE" in sql
        assert "mukey = '101'" in sql or "mukey = 101" in sql

    def test_build_query_with_multiple_keys(self):
        """_build_query should build IN condition for multiple values."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("mapunit", primary_key=[101, 102, 103])

        assert "WHERE" in sql
        assert "mukey IN" in sql

    def test_build_query_with_multiple_string_keys(self):
        """_build_query should handle string values properly."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("mapunit", secondary_key=["IA001A", "IA001B"])

        assert "WHERE" in sql
        assert "musym IN" in sql
        assert "'IA001A'" in sql

    def test_build_query_with_no_filters(self):
        """_build_query should return SELECT * when no filters provided."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("mapunit")

        assert sql == "SELECT * FROM mapunit"

    def test_build_query_component_table(self):
        """_build_query should work with component table."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("component", primary_key=101)

        assert "FROM component" in sql
        assert "WHERE" in sql

    def test_build_in_condition_numeric(self):
        """_build_in_condition should handle numeric values."""
        condition = SSURGOClient._build_in_condition("mukey", [101, 102, 103])

        assert "mukey IN (101,102,103)" in condition

    def test_build_in_condition_string(self):
        """_build_in_condition should handle string values."""
        condition = SSURGOClient._build_in_condition(
            "musym", ["IA001A", "IA001B"]
        )

        assert "musym IN" in condition
        assert "'IA001A'" in condition

    def test_build_in_condition_single_value(self):
        """_build_in_condition should handle single value."""
        condition = SSURGOClient._build_in_condition("mukey", 101)

        assert "mukey = 101" in condition


class TestSSURGOClientMethods:
    """Tests for SSURGOClient high-level methods."""

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_key(self):
        """fetch_mapunit should query by mukey."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_mapunit(mukey=[101, 102])

        assert "mapunit" in mock_backend.last_query
        assert "mukey IN" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_symbol(self):
        """fetch_mapunit should query by musym."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_mapunit(musym=["IA001A", "IA001B"])

        assert "mapunit" in mock_backend.last_query
        assert "musym IN" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_name(self):
        """fetch_mapunit should query by muname."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_mapunit(muname=["Miami", "Cary"])

        assert "mapunit" in mock_backend.last_query
        assert "muname IN" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_mapunit_with_where(self):
        """fetch_mapunit should support custom WHERE clause."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_mapunit(WHERE="muname LIKE 'Miami%'")

        assert "mapunit" in mock_backend.last_query
        assert "muname LIKE 'Miami%'" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_component(self):
        """fetch_component should query component table."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_component(mukey=101)

        assert "component" in mock_backend.last_query
        assert "mukey = 101" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_chorizon(self):
        """fetch_chorizon should query chorizon table."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_chorizon(cokey=101)

        assert "chorizon" in mock_backend.last_query
        assert "cokey = 101" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_fetch_legend(self):
        """fetch_legend should query legend table."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        await client.fetch_legend(areasymbol=["IA001", "IA025"])

        assert "legend" in mock_backend.last_query
        assert "areasymbol IN" in mock_backend.last_query

    @pytest.mark.asyncio
    async def test_get_available_tables(self):
        """get_available_tables should delegate to backend."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        tables = await client.get_available_tables()

        assert "mapunit" in tables
        assert "component" in tables

    @pytest.mark.asyncio
    async def test_get_table_schema(self):
        """get_table_schema should delegate to backend."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        schema = await client.get_table_schema("mapunit")

        assert "id" in schema
        assert "name" in schema


class TestSSURGOClientIntegration:
    """Integration tests for SSURGOClient."""

    @pytest.mark.asyncio
    async def test_ssurgo_client_with_mock_data(self):
        """SSURGOClient should execute queries via backend."""
        # Create mock backend with response
        mock_backend = AsyncMock()
        mock_response = MagicMock(spec=SDAResponse)
        mock_response.is_empty.return_value = False
        mock_response.to_pandas.return_value = MagicMock()
        mock_backend.execute.return_value = mock_response

        client = SSURGOClient(mock_backend)

        # Execute query
        response = await client.fetch_mapunit(mukey=[101, 102])

        # Verify backend was called
        mock_backend.execute.assert_called_once()

        # Verify response was returned
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_ssurgo_client_query_chaining(self):
        """SSURGOClient should support multiple sequential queries."""
        mock_backend = AsyncMock()
        mock_response = MagicMock(spec=SDAResponse)
        mock_backend.execute.return_value = mock_response

        client = SSURGOClient(mock_backend)

        # Execute multiple queries
        response1 = await client.fetch_mapunit(mukey=101)
        response2 = await client.fetch_component(mukey=101)
        response3 = await client.fetch_chorizon(cokey=101)

        # Verify all were executed
        assert mock_backend.execute.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
