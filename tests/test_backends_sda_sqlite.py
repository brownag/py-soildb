"""
Tests for SDABackend and SQLiteBackend implementations.

Tests verify that the refactored backends work correctly and provide
the same interface as the base infrastructure.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from soildb.backends import SDABackend, SQLiteBackend
from soildb.backends.exceptions import BackendConnectionError, BackendQueryError, BackendSchemaError


class TestSDABackend:
    """Tests for SDABackend."""

    @pytest.mark.asyncio
    async def test_sda_backend_initialization(self):
        """SDABackend should initialize without client."""
        backend = SDABackend()
        assert backend._client is None
        assert backend._owned_client is True

    @pytest.mark.asyncio
    async def test_sda_backend_with_client(self):
        """SDABackend should accept existing client."""
        mock_client = MagicMock()
        backend = SDABackend(client=mock_client)
        assert backend._client is mock_client
        assert backend._owned_client is False

    @pytest.mark.asyncio
    async def test_sda_backend_connect_fails_without_internet(self):
        """SDABackend.connect() should fail if client can't be created."""
        backend = SDABackend()

        # Mock the _get_client method to fail
        with patch.object(backend, '_get_client') as mock_get:
            mock_get.side_effect = Exception("Network error")

            with pytest.raises(BackendConnectionError):
                await backend.connect()

    @pytest.mark.asyncio
    async def test_sda_backend_execute_query(self):
        """SDABackend.execute() should delegate to SDAClient."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.is_empty.return_value = False
        mock_client.execute_sql.return_value = mock_response

        backend = SDABackend(client=mock_client)
        await backend.connect()

        result = await backend.execute("SELECT * FROM table")

        assert result == mock_response
        mock_client.execute_sql.assert_called_once_with("SELECT * FROM table")

    @pytest.mark.asyncio
    async def test_sda_backend_close(self):
        """SDABackend.close() should close owned client."""
        mock_client = AsyncMock()
        backend = SDABackend(client=mock_client)
        backend._owned_client = True

        await backend.close()

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sda_backend_close_with_external_client(self):
        """SDABackend.close() should not close external client."""
        mock_client = AsyncMock()
        backend = SDABackend(client=mock_client)
        backend._owned_client = False

        await backend.close()

        # Should not call close on external client
        mock_client.close.assert_not_called()


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
        assert backend._connected is True

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
        """SQLiteBackend.close() should clean up resources."""
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)
        await backend.connect()
        await backend.close()

        assert backend._connected is False


class TestBackendIntegrationWithNewImplementations:
    """Integration tests with the new backend implementations."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_context_manager(self, tmp_path):
        """SQLiteBackend should work as async context manager."""
        db_file = tmp_path / "test.db"
        db_file.touch()

        backend = SQLiteBackend(db_file)

        async with backend:
            assert backend._connected is True

        assert backend._connected is False

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
        assert hasattr(sda_backend, 'execute')
        assert hasattr(sda_backend, 'get_tables')
        assert hasattr(sda_backend, 'get_columns')

        assert hasattr(sqlite_backend, 'execute')
        assert hasattr(sqlite_backend, 'get_tables')
        assert hasattr(sqlite_backend, 'get_columns')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
