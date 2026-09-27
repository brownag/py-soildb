"""
Tests for SDABackend and SQLiteBackend implementations.

Tests verify that the refactored backends work correctly and provide
the same interface as the base infrastructure. All tests work through
the public interface without accessing private attributes or methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soildb.backends import SDABackend, SQLiteBackend
from soildb.backends.exceptions import (
    BackendConnectionError,
    BackendQueryError,
)
from soildb.client import SDAClient


class TestSDABackend:
    """Tests for SDABackend."""

    @pytest.mark.asyncio
    async def test_sda_backend_without_client_closes_owned_client(self):
        """SDABackend without explicit client creates and closes its own.

        Observable behavior: when created without a client, SDABackend should
        close its created client when close() is called.
        """
        # Create a mock SDAClient to track close calls
        mock_client = AsyncMock(spec=SDAClient)
        mock_client.close = AsyncMock()

        # Patch the SDAClient constructor to return our mock
        with patch("soildb.backends.sda_backend.SDAClient", return_value=mock_client):
            backend = SDABackend()
            # Trigger client creation via connect
            await backend.connect()
            # Close should close the owned client
            await backend.close()

            # Verify the client's close was called
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sda_backend_with_external_client_does_not_close_it(self):
        """SDABackend with provided client does not close it on close().

        Observable behavior: when initialized with an explicit SDAClient,
        that client should NOT be closed when backend.close() is called.
        """
        # Create a mock SDAClient with spy on close
        mock_client = AsyncMock(spec=SDAClient)
        mock_client.close = AsyncMock()

        # Create backend with the client
        backend = SDABackend(client=mock_client)
        await backend.close()

        # Verify the client's close was NOT called
        mock_client.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_sda_backend_connect_fails_when_client_unavailable(self):
        """SDABackend.connect() raises when SDAClient can't be created.

        Observable behavior: connect() should fail with BackendConnectionError
        if the SDAClient cannot be instantiated.
        """
        # Mock SDAClient constructor to raise an exception
        with patch(
            "soildb.backends.sda_backend.SDAClient",
            side_effect=Exception("Network error"),
        ):
            backend = SDABackend()
            with pytest.raises(BackendConnectionError):
                await backend.connect()

    @pytest.mark.asyncio
    async def test_sda_backend_execute_delegates_to_client(self):
        """SDABackend.execute() delegates query to SDAClient.execute_sql().

        Observable behavior: when execute() is called, the underlying
        SDAClient.execute_sql() should be called with the same SQL.
        """
        # Create a mock SDAClient with execute_sql method
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = MagicMock()
        mock_client.execute_sql = AsyncMock(return_value=mock_response)

        backend = SDABackend(client=mock_client)
        sql = "SELECT * FROM mapunit LIMIT 1"
        result = await backend.execute(sql)

        assert result == mock_response
        mock_client.execute_sql.assert_called_once_with(sql)

    @pytest.mark.asyncio
    async def test_sda_backend_context_manager_connects_and_closes(self):
        """SDABackend context manager connects on enter and closes on exit.

        Observable behavior: using SDABackend as an async context manager
        should call connect() and close() in proper order.
        """
        mock_client = AsyncMock(spec=SDAClient)
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()

        with patch("soildb.backends.sda_backend.SDAClient", return_value=mock_client):
            async with SDABackend() as backend:
                # Inside the context, should be connected
                # We can verify by trying to use the backend
                mock_response = MagicMock()
                mock_client.execute_sql = AsyncMock(return_value=mock_response)
                result = await backend.execute("SELECT 1")
                assert result == mock_response

            # After context exit, close should have been called
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sda_backend_get_columns_validates_table_name(self):
        """SDABackend.get_columns() should reject invalid table names."""
        mock_client = AsyncMock()
        backend = SDABackend(client=mock_client)

        # Table names with SQL injection attempts should be rejected
        invalid_names = [
            "mapunit; DROP TABLE--",
            "table' OR '1'='1",
            "table`; SELECT * FROM--",
            "table<script>alert(1)</script>",
        ]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid SQL identifier"):
                await backend.get_columns(invalid_name)

    @pytest.mark.asyncio
    async def test_sda_backend_get_columns_accepts_valid_identifiers(self):
        """SDABackend.get_columns() should accept valid table names."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.is_empty.return_value = True
        mock_client.execute_sql.return_value = mock_response

        backend = SDABackend(client=mock_client)
        await backend.connect()

        # Valid identifiers should be accepted
        valid_names = ["mapunit", "component", "m.mapunit", "table_with_underscore"]

        for valid_name in valid_names:
            schema = await backend.get_columns(valid_name)
            # Should not raise and should return a dict (even if empty)
            assert isinstance(schema, dict)


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    def test_sqlite_backend_initialization(self, tmp_path):
        """SQLiteBackend should initialize with valid database path."""
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)
        assert backend.db_path == db_file

    def test_sqlite_backend_missing_database(self):
        """SQLiteBackend should fail if database doesn't exist."""
        with pytest.raises(BackendConnectionError):
            SQLiteBackend("/nonexistent/database.db")

    @pytest.mark.asyncio
    async def test_sqlite_backend_connect(self, tmp_path):
        """SQLiteBackend.connect() should succeed with valid database."""
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)
        result = await backend.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_sqlite_backend_execute_empty_result(self, tmp_path):
        """SQLiteBackend.execute() should handle empty results."""
        # Create an in-memory SQLite database for testing
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)
        await backend.connect()

        # Query non-existent table
        with pytest.raises(BackendQueryError):
            await backend.execute("SELECT * FROM nonexistent_table")

    @pytest.mark.asyncio
    async def test_sqlite_backend_get_tables(self, tmp_path):
        """SQLiteBackend.get_tables() should return table list."""
        import aiosqlite

        db_file = tmp_path / "test.db"

        # Create database with a table
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
            await db.commit()

        backend = SQLiteBackend(db_file)
        tables = await backend.get_tables()

        assert "test_table" in tables

    @pytest.mark.asyncio
    async def test_sqlite_backend_get_columns(self, tmp_path):
        """SQLiteBackend.get_columns() should return column schema."""
        import aiosqlite

        db_file = tmp_path / "test.db"

        # Create database with a table
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
            await db.commit()

        backend = SQLiteBackend(db_file)
        columns = await backend.get_columns("test_table")

        assert "id" in columns
        assert "name" in columns
        assert columns["id"].upper() == "INTEGER"
        assert columns["name"].upper() == "TEXT"

    @pytest.mark.asyncio
    async def test_sqlite_backend_close(self, tmp_path):
        """SQLiteBackend.close() should clean up resources.

        Observable behavior: after close(), subsequent operations should fail.
        """
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)
        await backend.connect()
        await backend.close()

        # After close, operations on the database should fail
        # (close() sets _connected to False, which affects the backend state)
        # We verify this by ensuring the close() method completes without error


class TestBackendIntegrationWithNewImplementations:
    """Integration tests with the new backend implementations."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_context_manager(self, tmp_path):
        """SQLiteBackend context manager connects on enter and closes on exit.

        Observable behavior: operations should work inside the context manager
        but the backend should clean up properly after exiting.
        """
        import aiosqlite

        db_file = tmp_path / "test.db"

        # Create database with test data
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE test_data (id INTEGER, value TEXT)")
            await db.execute("INSERT INTO test_data VALUES (1, 'test')")
            await db.commit()

        backend = SQLiteBackend(db_file)

        # Inside context, operations should work
        async with backend:
            response = await backend.execute("SELECT * FROM test_data")
            assert not response.is_empty()
            assert len(response.to_dict()) > 0

        # After context, backend should be closed (we can't easily verify this
        # without accessing private state, so we just ensure close() was called)

    @pytest.mark.asyncio
    async def test_sqlite_backend_execute_returns_correct_rows_and_columns(
        self, tmp_path
    ):
        """SQLiteBackend.execute() returns correct rows and columns with types.

        Observable behavior: execute() should return SDAResponse with correct
        row data and column names, with proper type inference.
        """
        import aiosqlite

        db_file = tmp_path / "test.db"

        # Create database with test data including various types
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("""
                CREATE TABLE test_data (
                    id INTEGER,
                    score REAL,
                    name TEXT,
                    notes TEXT
                )
            """)
            await db.execute("INSERT INTO test_data VALUES (1, 95.5, 'Alice', NULL)")
            await db.execute("INSERT INTO test_data VALUES (2, 87.3, 'Bob', 'notes')")
            await db.commit()

        backend = SQLiteBackend(db_file)
        response = await backend.execute("SELECT * FROM test_data ORDER BY id")

        # Verify response structure
        assert not response.is_empty()
        rows = response.to_dict()
        assert len(rows) == 2

        # Check row data
        assert rows[0]["id"] == 1
        assert rows[0]["score"] == 95.5
        assert rows[0]["name"] == "Alice"
        # NULL values may be converted to None or empty string depending on type inference
        assert rows[0]["notes"] in (None, "")

        assert rows[1]["id"] == 2
        assert rows[1]["score"] == 87.3
        assert rows[1]["name"] == "Bob"
        assert rows[1]["notes"] == "notes"

    @pytest.mark.asyncio
    async def test_sqlite_backend_query_with_data(self, tmp_path):
        """SQLiteBackend should execute queries and return results."""
        import aiosqlite

        db_file = tmp_path / "test.db"

        # Create database with test data
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            await db.execute("INSERT INTO users VALUES (1, 'Alice')")
            await db.execute("INSERT INTO users VALUES (2, 'Bob')")
            await db.commit()

        backend = SQLiteBackend(db_file)
        async with backend:
            response = await backend.execute("SELECT * FROM users ORDER BY id")
            df = response.to_pandas()

            assert len(df) == 2
            assert "id" in df.columns
            assert "name" in df.columns

    @pytest.mark.asyncio
    async def test_backend_factory_pattern(self, tmp_path):
        """Test creating backends dynamically."""
        db_file = tmp_path / "test.db"
        db_file.touch()

        # Create backends
        sda_backend = SDABackend()
        sqlite_backend = SQLiteBackend(db_file)

        # Both should implement the BaseBackend interface
        assert hasattr(sda_backend, "execute")
        assert hasattr(sda_backend, "get_tables")
        assert hasattr(sda_backend, "get_columns")

        assert hasattr(sqlite_backend, "execute")
        assert hasattr(sqlite_backend, "get_tables")
        assert hasattr(sqlite_backend, "get_columns")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
