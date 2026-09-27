"""
Unit tests for backend infrastructure components.

Tests cover:
- BaseBackend abstract interface
- SchemaIntrospector schema discovery
- BackendError exception hierarchy
"""

import os
import sqlite3
import tempfile

import pytest

from soildb.backends import (
    BackendConnectionError,
    BackendError,
    BackendQueryError,
    BackendSchemaError,
    BaseBackend,
    ColumnInfo,
    DatabaseTableSchema,
    SchemaIntrospector,
    SQLiteBackend,
)
from soildb.response import SDAResponse


class MockBackend(BaseBackend):
    """Mock backend for testing abstract interface."""

    def __init__(self, config=None, should_fail=False, fail_on=None):
        super().__init__(config)
        self.should_fail = should_fail
        self.fail_on = fail_on or []
        self.connected = False
        self.executed_queries = []
        self.closed = False

    async def __aenter__(self):
        """Support async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager."""
        await self.close()
        return False

    async def connect(self) -> bool:
        if self.should_fail and "connect" in self.fail_on:
            raise BackendConnectionError(
                "Mock connection failed", details="test failure"
            )
        self.connected = True
        return True

    async def execute(self, sql: str) -> SDAResponse:
        if self.should_fail and "execute" in self.fail_on:
            raise BackendQueryError(f"Mock query failed: {sql}", details="test failure")
        self.executed_queries.append(sql)

        # Return mock response as dictionary
        response_dict = {
            "Table": [
                ["id", "name"],
                ["int", "varchar"],
                [1, "test1"],
                [2, "test2"],
            ]
        }
        return SDAResponse(response_dict)

    async def get_tables(self) -> list[str]:
        if self.should_fail and "get_tables" in self.fail_on:
            raise BackendSchemaError("Mock get_tables failed", details="test failure")
        return ["table1", "table2", "table3"]

    async def get_columns(self, table_name: str) -> dict[str, str]:
        if self.should_fail and "get_columns" in self.fail_on:
            raise BackendSchemaError(
                f"Mock get_columns failed for {table_name}", details="test failure"
            )
        return {"id": "int", "name": "varchar", "created_at": "datetime"}

    async def close(self) -> None:
        self.closed = True


class TestBackendError:
    """Tests for exception hierarchy."""

    def test_backend_error_is_exception(self):
        """BackendError should be an Exception."""
        error = BackendError("test error")
        assert isinstance(error, Exception)

    def test_backend_connection_error(self):
        """BackendConnectionError should have proper inheritance."""
        error = BackendConnectionError("connection failed", details="test details")
        assert isinstance(error, BackendError)
        assert isinstance(error, Exception)
        assert "Failed to connect to data backend" in str(error)

    def test_backend_query_error(self):
        """BackendQueryError should have proper inheritance."""
        error = BackendQueryError("query failed")
        assert isinstance(error, BackendError)
        assert isinstance(error, Exception)

    def test_backend_schema_error(self):
        """BackendSchemaError should have proper inheritance."""
        error = BackendSchemaError("schema introspection failed")
        assert isinstance(error, BackendError)
        assert isinstance(error, Exception)

    def test_backend_error_with_cause(self):
        """BackendError should support chaining with __cause__."""
        original = ValueError("original error")
        backend_error = BackendQueryError("wrapper error")
        backend_error.__cause__ = original
        assert backend_error.__cause__ is original


class TestBaseBackend:
    """Tests for BaseBackend abstract interface."""

    @pytest.mark.asyncio
    async def test_context_manager_interface(self):
        """BaseBackend should support async context manager."""
        backend = MockBackend()
        async with backend:
            assert backend.connected is True
        assert backend.closed is True

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        """Backend should connect and close properly."""
        backend = MockBackend()
        assert backend.connected is False

        connected = await backend.connect()
        assert connected is True
        assert backend.connected is True

        await backend.close()
        assert backend.closed is True

    @pytest.mark.asyncio
    async def test_execute_returns_sda_response(self):
        """execute() should return SDAResponse."""
        backend = MockBackend()
        await backend.connect()

        response = await backend.execute("SELECT * FROM table1")
        assert isinstance(response, SDAResponse)

    @pytest.mark.asyncio
    async def test_execute_many_concurrent(self):
        """execute_many() should run queries concurrently."""
        backend = MockBackend()
        await backend.connect()

        queries = [
            "SELECT * FROM table1",
            "SELECT * FROM table2",
            "SELECT * FROM table3",
        ]
        responses = await backend.execute_many(queries)

        assert len(responses) == 3
        assert all(isinstance(r, SDAResponse) for r in responses)
        assert len(backend.executed_queries) == 3

    @pytest.mark.asyncio
    async def test_get_tables(self):
        """get_tables() should return list of table names."""
        backend = MockBackend()
        await backend.connect()

        tables = await backend.get_tables()
        assert tables == ["table1", "table2", "table3"]

    @pytest.mark.asyncio
    async def test_get_columns(self):
        """get_columns() should return dict of column names and types."""
        backend = MockBackend()
        await backend.connect()

        columns = await backend.get_columns("table1")
        assert columns == {"id": "int", "name": "varchar", "created_at": "datetime"}

    @pytest.mark.asyncio
    async def test_connection_error_propagation(self):
        """Connection errors should propagate properly."""
        backend = MockBackend(should_fail=True, fail_on=["connect"])

        with pytest.raises(BackendConnectionError):
            await backend.connect()

    @pytest.mark.asyncio
    async def test_query_error_propagation(self):
        """Query errors should propagate properly."""
        backend = MockBackend(should_fail=True, fail_on=["execute"])
        await backend.connect()

        with pytest.raises(BackendQueryError):
            await backend.execute("SELECT * FROM table1")

    @pytest.mark.asyncio
    async def test_schema_error_propagation(self):
        """Schema errors should propagate properly."""
        backend = MockBackend(should_fail=True, fail_on=["get_tables"])
        await backend.connect()

        with pytest.raises(BackendSchemaError):
            await backend.get_tables()


class TestSchemaIntrospector:
    """Tests for SchemaIntrospector schema discovery."""

    @pytest.mark.asyncio
    async def test_introspect_table(self):
        """introspect_table() should get table schema."""
        backend = MockBackend()
        await backend.connect()

        schema = await SchemaIntrospector.introspect_table(backend, "table1")

        assert isinstance(schema, DatabaseTableSchema)
        assert schema.name == "table1"
        assert "id" in schema.columns
        assert "name" in schema.columns
        assert "created_at" in schema.columns

    @pytest.mark.asyncio
    async def test_introspect_table_column_info(self):
        """Introspected columns should have ColumnInfo."""
        backend = MockBackend()
        await backend.connect()

        schema = await SchemaIntrospector.introspect_table(backend, "table1")

        assert isinstance(schema.columns["id"], ColumnInfo)
        assert schema.columns["id"].name == "id"
        assert schema.columns["id"].type == "int"

    @pytest.mark.asyncio
    async def test_introspect_table_geometry_detection(self):
        """Introspector should detect geometry columns."""
        backend = MockBackend()

        # Mock get_columns to return geometry column
        async def mock_get_columns(table_name):
            return {"id": "int", "geom": "geometry"}

        backend.get_columns = mock_get_columns

        schema = await SchemaIntrospector.introspect_table(backend, "table1")

        assert schema.geometry_column == "geom"
        assert schema.is_spatial is True

    @pytest.mark.asyncio
    async def test_introspect_database(self):
        """introspect_database() should get all table schemas."""
        backend = MockBackend()
        await backend.connect()

        schemas = await SchemaIntrospector.introspect_database(backend)

        assert isinstance(schemas, dict)
        assert len(schemas) >= 3  # At least the 3 mock tables
        for schema in schemas.values():
            assert isinstance(schema, DatabaseTableSchema)

    @pytest.mark.asyncio
    async def test_introspect_database_skip_errors(self):
        """introspect_database() should skip tables that fail."""
        backend = MockBackend()

        # Mock get_tables to return multiple tables
        async def mock_get_tables():
            return ["table1", "table2", "failing_table"]

        # Mock get_columns to fail on specific table
        call_count = 0

        async def mock_get_columns(table_name):
            nonlocal call_count
            call_count += 1
            if table_name == "failing_table":
                raise BackendSchemaError("This table is broken")
            return {"id": "int"}

        backend.get_tables = mock_get_tables
        backend.get_columns = mock_get_columns

        schemas = await SchemaIntrospector.introspect_database(backend)

        # Should have 2 schemas (skipped the failing one)
        assert len(schemas) == 2
        assert "failing_table" not in schemas

    @pytest.mark.asyncio
    async def test_is_spatial_type_detection(self):
        """_is_spatial_type() should detect spatial types."""
        assert SchemaIntrospector._is_spatial_type("GEOMETRY") is True
        assert SchemaIntrospector._is_spatial_type("geometry") is True
        assert SchemaIntrospector._is_spatial_type("GEOGRAPHY") is True
        assert SchemaIntrospector._is_spatial_type("POINT") is True
        assert SchemaIntrospector._is_spatial_type("POLYGON") is True
        assert SchemaIntrospector._is_spatial_type("int") is False
        assert SchemaIntrospector._is_spatial_type("varchar") is False


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""

    def test_column_info_basic(self):
        """ColumnInfo should store column metadata."""
        col = ColumnInfo(name="id", type="int")
        assert col.name == "id"
        assert col.type == "int"
        assert col.nullable is True
        assert col.primary_key is False
        assert col.spatial is False

    def test_column_info_with_options(self):
        """ColumnInfo should support optional attributes."""
        col = ColumnInfo(
            name="geom",
            type="geometry",
            nullable=False,
            primary_key=False,
            spatial=True,
        )
        assert col.name == "geom"
        assert col.nullable is False
        assert col.spatial is True

    def test_column_info_is_geometry(self):
        """is_geometry property should detect geometry columns."""
        geom_col = ColumnInfo(name="geom", type="geometry")
        assert geom_col.is_geometry is True

        int_col = ColumnInfo(name="id", type="int")
        assert int_col.is_geometry is False

        blob_col = ColumnInfo(name="data", type="blob")
        assert blob_col.is_geometry is False  # BLOB without spatial flag

        # BLOB with spatial flag should be geometry (WKB)
        wkb_col = ColumnInfo(name="geom", type="blob", spatial=True)
        assert wkb_col.is_geometry is True


class TestDatabaseTableSchema:
    """Tests for DatabaseTableSchema dataclass."""

    def test_table_schema_basic(self):
        """DatabaseTableSchema should store table metadata."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "name": ColumnInfo(name="name", type="varchar"),
        }
        schema = DatabaseTableSchema(name="users", columns=columns)

        assert schema.name == "users"
        assert len(schema.columns) == 2
        assert schema.is_spatial is False

    def test_table_schema_with_geometry(self):
        """DatabaseTableSchema should detect spatial tables."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "geom": ColumnInfo(name="geom", type="geometry", spatial=True),
        }
        schema = DatabaseTableSchema(
            name="features",
            columns=columns,
            geometry_column="geom",
        )

        assert schema.is_spatial is True
        assert schema.geometry_column == "geom"

    def test_table_schema_column_names(self):
        """column_names property should return list of column names."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "name": ColumnInfo(name="name", type="varchar"),
        }
        schema = DatabaseTableSchema(name="users", columns=columns)

        names = schema.column_names
        assert set(names) == {"id", "name"}

    def test_table_schema_column_types(self):
        """column_types property should return type mapping."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "name": ColumnInfo(name="name", type="varchar"),
        }
        schema = DatabaseTableSchema(name="users", columns=columns)

        types = schema.column_types
        assert types == {"id": "int", "name": "varchar"}


class TestSQLiteBackendInterface:
    """Interface tests for SQLite backend type inference."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_infers_types_correctly(self):
        """SQLite backend should infer and preserve types in SDAResponse."""
        # Create a temporary SQLite database with mixed types
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create test table and insert data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE test_types (
                    int_col INTEGER,
                    float_col REAL,
                    str_col TEXT,
                    nullable_col TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO test_types (int_col, float_col, str_col, nullable_col)
                VALUES (42, 3.14, 'hello', NULL),
                       (100, 2.71, 'world', 'value')
                """
            )
            conn.commit()
            conn.close()

            # Execute via SQLite backend
            backend = SQLiteBackend(db_path)
            response = await backend.execute("SELECT * FROM test_types")

            # Verify response types
            data = response.to_dict()
            assert len(data) == 2

            # Check first row: (42, 3.14, 'hello', NULL)
            row1 = data[0]
            assert row1["int_col"] == 42
            assert isinstance(row1["int_col"], int)
            assert row1["float_col"] == pytest.approx(3.14)
            assert isinstance(row1["float_col"], float)
            assert row1["str_col"] == "hello"
            assert isinstance(row1["str_col"], str)
            # NULL values may be converted to empty string or None depending on type
            assert row1["nullable_col"] in (None, "")

            # Check second row: (100, 2.71, 'world', 'value')
            row2 = data[1]
            assert row2["int_col"] == 100
            assert isinstance(row2["int_col"], int)
            assert row2["float_col"] == pytest.approx(2.71)
            assert isinstance(row2["float_col"], float)
            assert row2["str_col"] == "world"
            assert isinstance(row2["str_col"], str)
            assert row2["nullable_col"] == "value"
            assert isinstance(row2["nullable_col"], str)

        finally:
            # Clean up
            os.unlink(db_path)


# Integration test combining multiple components
class TestBackendIntegration:
    """Integration tests combining multiple backend components."""

    @pytest.mark.asyncio
    async def test_full_backend_workflow(self):
        """Test complete workflow: connect -> introspect -> execute -> adapt."""
        backend = MockBackend()

        async with backend:
            # Get schema
            schema = await SchemaIntrospector.introspect_table(backend, "test_table")
            assert isinstance(schema, DatabaseTableSchema)

            # Execute query
            response = await backend.execute("SELECT * FROM test_table")
            assert isinstance(response, SDAResponse)

            # Convert to DataFrame
            df = response.to_pandas()
            assert len(df) > 0

    @pytest.mark.asyncio
    async def test_multi_table_schema_discovery(self):
        """Test discovering schema for multiple tables."""
        backend = MockBackend()

        async with backend:
            # Get all tables
            tables = await backend.get_tables()
            assert len(tables) > 0

            # Introspect all tables
            schemas = await SchemaIntrospector.introspect_database(backend)
            assert len(schemas) > 0

            # Verify structure
            for _table_name, schema in schemas.items():
                assert schema.name in tables
                assert len(schema.columns) > 0

    @pytest.mark.asyncio
    async def test_response_combination_workflow(self):
        """Test combining multiple response objects with SDAResponse.concat."""
        backend = MockBackend()

        async with backend:
            # Execute multiple queries
            responses = await backend.execute_many(
                ["SELECT * FROM table1", "SELECT * FROM table2"]
            )

            # Combine responses using concat
            combined = SDAResponse.concat(responses)
            df = combined.to_pandas()

            # Should have combined results
            assert len(df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
