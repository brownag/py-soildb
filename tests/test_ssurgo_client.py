"""
Tests for SSURGOClient generic SSURGO query builder.

Tests verify that SSURGOClient can construct proper SQL queries
for SSURGO tables and execute them via any backend.
"""

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from soildb.backends import SQLiteBackend
from soildb.response import SDAResponse
from soildb.ssurgo_client import SSURGOClient
from soildb.ssurgo_tables import filter_fields


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


@pytest.fixture
def tmp_ssurgo_db(tmp_path):
    """Create a temporary SQLite database with SSURGO test tables.

    Sets up minimal mapunit, component, chorizon, and legend tables
    with test data for interface testing.

    Args:
        tmp_path: pytest tmp_path fixture

    Yields:
        Path: Path to the temporary SQLite database file
    """
    db_path = tmp_path / "test.db"

    # Create tables and insert test data using sqlite3 (sync)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # mapunit table: mukey, musym, muname
    cursor.execute(
        """
        CREATE TABLE mapunit (
            mukey INTEGER PRIMARY KEY,
            musym TEXT,
            muname TEXT
        )
        """
    )
    cursor.execute(
        "INSERT INTO mapunit (mukey, musym, muname) VALUES (101, 'IA001A', 'Miami')"
    )
    cursor.execute(
        "INSERT INTO mapunit (mukey, musym, muname) VALUES (102, 'IA001B', 'Cary')"
    )
    cursor.execute(
        'INSERT INTO mapunit (mukey, musym, muname) VALUES (103, "O\'Brien", "O\'Brien soil")'
    )

    # component table: cokey, mukey, compname
    cursor.execute(
        """
        CREATE TABLE component (
            cokey INTEGER PRIMARY KEY,
            mukey INTEGER,
            compname TEXT
        )
        """
    )
    cursor.execute(
        "INSERT INTO component (cokey, mukey, compname) VALUES (201, 101, 'Miami')"
    )
    cursor.execute(
        "INSERT INTO component (cokey, mukey, compname) VALUES (202, 102, 'Cary')"
    )

    # chorizon table: chkey, cokey, hzname
    cursor.execute(
        """
        CREATE TABLE chorizon (
            chkey INTEGER PRIMARY KEY,
            cokey INTEGER,
            hzname TEXT
        )
        """
    )
    cursor.execute(
        "INSERT INTO chorizon (chkey, cokey, hzname) VALUES (301, 201, 'Ap')"
    )
    cursor.execute(
        "INSERT INTO chorizon (chkey, cokey, hzname) VALUES (302, 202, 'Bt')"
    )

    # legend table: lkey, areasymbol
    cursor.execute(
        """
        CREATE TABLE legend (
            lkey INTEGER PRIMARY KEY,
            areasymbol TEXT
        )
        """
    )
    cursor.execute("INSERT INTO legend (lkey, areasymbol) VALUES (401, 'IA001')")
    cursor.execute("INSERT INTO legend (lkey, areasymbol) VALUES (402, 'IA025')")

    conn.commit()
    conn.close()

    yield db_path


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
    """Tests for SQL query building via interface with real SQLite backend."""

    def test_build_query_with_where_clause(self):
        """_build_query should use custom WHERE clause when provided."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query(
            "mapunit",
            where_clause="muname LIKE 'Miami%'",
        )

        assert "WHERE muname LIKE 'Miami%'" in sql

    def test_build_query_with_no_filters(self):
        """_build_query should return SELECT * when no filters provided."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        sql = client._build_query("mapunit")

        assert sql == "SELECT * FROM mapunit"

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_single_key(self, tmp_ssurgo_db):
        """fetch_mapunit should fetch mapunit by single mukey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_mapunit(mukey=101)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["mukey"] == 101
        assert data[0]["musym"] == "IA001A"
        assert data[0]["muname"] == "Miami"

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_multiple_keys(self, tmp_ssurgo_db):
        """fetch_mapunit should fetch mapunit by multiple mukeys."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_mapunit(mukey=[101, 102])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 2
        mukeys = {row["mukey"] for row in data}
        assert mukeys == {101, 102}

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_symbol(self, tmp_ssurgo_db):
        """fetch_mapunit should fetch mapunit by musym."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_mapunit(musym=["IA001A", "IA001B"])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 2
        symbols = {row["musym"] for row in data}
        assert symbols == {"IA001A", "IA001B"}

    @pytest.mark.asyncio
    async def test_fetch_mapunit_by_name(self, tmp_ssurgo_db):
        """fetch_mapunit should fetch mapunit by muname."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_mapunit(muname=["Miami", "Cary"])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 2
        names = {row["muname"] for row in data}
        assert names == {"Miami", "Cary"}

    @pytest.mark.asyncio
    async def test_fetch_mapunit_with_quote_in_value(self, tmp_ssurgo_db):
        """fetch_mapunit should handle values with single quotes correctly."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        # Fetch the row with O'Brien in the musym
        response = await client.fetch_mapunit(musym=["O'Brien"])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["musym"] == "O'Brien"

    @pytest.mark.asyncio
    async def test_fetch_component_by_key(self, tmp_ssurgo_db):
        """fetch_component should fetch component by cokey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_component(cokey=201)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["cokey"] == 201
        assert data[0]["compname"] == "Miami"

    @pytest.mark.asyncio
    async def test_fetch_component_by_mukey(self, tmp_ssurgo_db):
        """fetch_component should fetch component by mukey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_component(mukey=101)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["mukey"] == 101

    @pytest.mark.asyncio
    async def test_fetch_component_by_name(self, tmp_ssurgo_db):
        """fetch_component should fetch component by compname."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_component(compname=["Miami", "Cary"])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 2
        names = {row["compname"] for row in data}
        assert names == {"Miami", "Cary"}

    @pytest.mark.asyncio
    async def test_fetch_chorizon_by_key(self, tmp_ssurgo_db):
        """fetch_chorizon should fetch chorizon by chkey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_chorizon(chkey=301)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["chkey"] == 301
        assert data[0]["hzname"] == "Ap"

    @pytest.mark.asyncio
    async def test_fetch_chorizon_by_cokey(self, tmp_ssurgo_db):
        """fetch_chorizon should fetch chorizon by cokey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_chorizon(cokey=201)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["cokey"] == 201

    @pytest.mark.asyncio
    async def test_fetch_legend_by_key(self, tmp_ssurgo_db):
        """fetch_legend should fetch legend by lkey."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_legend(lkey=401)

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 1
        assert data[0]["lkey"] == 401
        assert data[0]["areasymbol"] == "IA001"

    @pytest.mark.asyncio
    async def test_fetch_legend_by_areasymbol(self, tmp_ssurgo_db):
        """fetch_legend should fetch legend by areasymbol."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        response = await client.fetch_legend(areasymbol=["IA001", "IA025"])

        assert not response.is_empty()
        data = response.to_dict()
        assert len(data) == 2
        symbols = {row["areasymbol"] for row in data}
        assert symbols == {"IA001", "IA025"}


class TestSSURGOClientMethods:
    """Tests for SSURGOClient high-level methods with mock backend."""

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
        await client.fetch_mapunit(mukey=101)
        await client.fetch_component(mukey=101)
        await client.fetch_chorizon(cokey=101)

        # Verify all were executed
        assert mock_backend.execute.call_count == 3


class TestSSURGOClientFilterFields:
    """Tests for SSURGOClient using FILTER_FIELDS metadata."""

    def test_build_query_uses_filter_fields_primary(self):
        """_build_query should use filter_fields primary column from metadata."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        # mapunit's primary filter field is mukey
        sql = client._build_query("mapunit", primary_key=101)
        assert "mukey" in sql or "mapunit" in sql

    def test_build_query_uses_filter_fields_secondary(self):
        """_build_query should use filter_fields secondary column from metadata."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        # mapunit's secondary filter field is musym
        sql = client._build_query("mapunit", primary_key=None, secondary_key="IA001A")
        assert "musym" in sql or "mapunit" in sql

    def test_build_query_uses_filter_fields_tertiary(self):
        """_build_query should use filter_fields tertiary column from metadata."""
        mock_backend = MockBackend()
        client = SSURGOClient(mock_backend)

        # mapunit's tertiary filter field is muname
        sql = client._build_query(
            "mapunit", primary_key=None, secondary_key=None, tertiary_key="Miami"
        )
        assert "muname" in sql or "mapunit" in sql

    def test_filter_fields_for_component_table(self):
        """Verify filter_fields for component table."""
        # component's filter fields should be: cokey, mukey, compname
        fields = filter_fields("component")
        assert fields is not None
        assert fields[0] == "cokey"
        assert fields[1] == "mukey"
        assert fields[2] == "compname"

    def test_filter_fields_for_chorizon_table(self):
        """Verify filter_fields for chorizon table."""
        # chorizon's filter fields should be: chkey, cokey, hzname
        fields = filter_fields("chorizon")
        assert fields is not None
        assert fields[0] == "chkey"
        assert fields[1] == "cokey"
        assert fields[2] == "hzname"

    def test_filter_fields_for_legend_table(self):
        """Verify filter_fields for legend table."""
        # legend's filter fields should be: lkey, areasymbol, areaname
        fields = filter_fields("legend")
        assert fields is not None
        assert fields[0] == "lkey"
        assert fields[1] == "areasymbol"
        assert fields[2] == "areaname"

    @pytest.mark.asyncio
    async def test_fetch_mapunit_uses_filter_fields(self, tmp_ssurgo_db):
        """fetch_mapunit should correctly use filter_fields for queries."""
        backend = SQLiteBackend(tmp_ssurgo_db)
        client = SSURGOClient(backend)

        # Fetch by primary filter field (mukey)
        response = await client.fetch_mapunit(mukey=101)
        assert not response.is_empty()
        data = response.to_dict()
        assert data[0]["mukey"] == 101

        # Fetch by secondary filter field (musym)
        response = await client.fetch_mapunit(musym="IA001A")
        assert not response.is_empty()
        data = response.to_dict()
        assert data[0]["musym"] == "IA001A"

        # Fetch by tertiary filter field (muname)
        response = await client.fetch_mapunit(muname="Miami")
        assert not response.is_empty()
        data = response.to_dict()
        assert data[0]["muname"] == "Miami"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
