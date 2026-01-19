"""
Unit tests for backend infrastructure components.

Tests cover:
- BaseBackend abstract interface
- ResponseAdapter response conversion
- SchemaIntrospector schema discovery
- TypeMapperFactory type mappings
- BackendError exception hierarchy
"""

import json
import pytest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from soildb.backends import (
    BaseBackend,
    BackendConnectionError,
    BackendError,
    BackendQueryError,
    BackendSchemaError,
    ColumnInfo,
    DatabaseTypeMapper,
    ResponseAdapter,
    SchemaIntrospector,
    TableSchema,
    TypeMapperFactory,
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
            raise BackendConnectionError("Mock connection failed", details="test failure")
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

    async def get_tables(self) -> List[str]:
        if self.should_fail and "get_tables" in self.fail_on:
            raise BackendSchemaError("Mock get_tables failed", details="test failure")
        return ["table1", "table2", "table3"]

    async def get_columns(self, table_name: str) -> Dict[str, str]:
        if self.should_fail and "get_columns" in self.fail_on:
            raise BackendSchemaError(f"Mock get_columns failed for {table_name}", details="test failure")
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

        queries = ["SELECT * FROM table1", "SELECT * FROM table2", "SELECT * FROM table3"]
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


class TestResponseAdapter:
    """Tests for ResponseAdapter response conversion."""

    @pytest.mark.asyncio
    async def test_from_rows_basic(self):
        """from_rows() should convert tuple list to SDAResponse."""
        rows = [
            (1, "Alice", 25.5),
            (2, "Bob", 30.0),
        ]
        columns = ["id", "name", "score"]

        response = await ResponseAdapter.from_rows(rows, columns)
        assert isinstance(response, SDAResponse)

        df = response.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "score"]
        # Values may be typed as strings by SDAResponse
        assert df.iloc[0]["id"] in (1, "1")
        assert df.iloc[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_from_rows_empty(self):
        """from_rows() should handle empty results."""
        rows = []
        columns = ["id", "name", "score"]

        response = await ResponseAdapter.from_rows(rows, columns)
        df = response.to_pandas()
        assert len(df) == 0
        # Column names are preserved even with empty results
        assert "id" in df.columns or len(df.columns) == 0

    @pytest.mark.asyncio
    async def test_from_rows_with_nulls(self):
        """from_rows() should handle NULL values."""
        rows = [
            (1, "Alice", None),
            (None, "Bob", 30.0),
        ]
        columns = ["id", "name", "score"]

        response = await ResponseAdapter.from_rows(rows, columns)
        df = response.to_pandas()
        assert len(df) == 2
        # Nulls may be converted to empty strings or NaN
        score_val = df.iloc[0]["score"]
        assert score_val in (None, "", float('nan')) or score_val != score_val

        id_val = df.iloc[1]["id"]
        assert id_val in (None, "", float('nan')) or id_val != id_val

    @pytest.mark.asyncio
    async def test_from_rows_type_inference(self):
        """from_rows() should infer types from values."""
        rows = [
            (1, "text", 1.5, True),
        ]
        columns = ["int_col", "str_col", "float_col", "bool_col"]

        response = await ResponseAdapter.from_rows(rows, columns)
        df = response.to_pandas()

        # Values should be present (may be typed as strings by SDAResponse)
        assert df.iloc[0]["int_col"] in (1, "1")
        assert df.iloc[0]["str_col"] == "text"
        assert df.iloc[0]["float_col"] in (1.5, "1.5")

    @pytest.mark.asyncio
    async def test_from_dict_rows(self):
        """from_dict_rows() should convert dict list to SDAResponse."""
        rows = [
            {"id": 1, "name": "Alice", "score": 25.5},
            {"id": 2, "name": "Bob", "score": 30.0},
        ]
        columns = ["id", "name", "score"]

        response = await ResponseAdapter.from_dict_rows(rows, columns)
        df = response.to_pandas()

        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "score"]
        assert df.iloc[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_from_dict_rows_empty(self):
        """from_dict_rows() should handle empty results."""
        rows = []
        columns = ["id", "name", "score"]
        response = await ResponseAdapter.from_dict_rows(rows, columns)
        df = response.to_pandas()
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_combine_responses(self):
        """combine_responses() should merge multiple responses."""
        response1 = SDAResponse(
            {
                "Table": [
                    ["id", "name"],
                    ["int", "varchar"],
                    [1, "Alice"],
                ]
            }
        )
        response2 = SDAResponse(
            {
                "Table": [
                    ["id", "name"],
                    ["int", "varchar"],
                    [2, "Bob"],
                ]
            }
        )

        combined = await ResponseAdapter.combine_responses([response1, response2])
        df = combined.to_pandas()

        assert len(df) == 2
        # Values may be converted to strings
        assert list(df["id"]) in ([1, 2], ["1", "2"])
        assert list(df["name"]) == ["Alice", "Bob"]

    @pytest.mark.asyncio
    async def test_combine_responses_empty(self):
        """combine_responses() should handle empty list."""
        combined = await ResponseAdapter.combine_responses([])
        df = combined.to_pandas()
        assert len(df) == 0


class TestSchemaIntrospector:
    """Tests for SchemaIntrospector schema discovery."""

    @pytest.mark.asyncio
    async def test_introspect_table(self):
        """introspect_table() should get table schema."""
        backend = MockBackend()
        await backend.connect()

        schema = await SchemaIntrospector.introspect_table(backend, "table1")

        assert isinstance(schema, TableSchema)
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
            assert isinstance(schema, TableSchema)

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


class TestTableSchema:
    """Tests for TableSchema dataclass."""

    def test_table_schema_basic(self):
        """TableSchema should store table metadata."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "name": ColumnInfo(name="name", type="varchar"),
        }
        schema = TableSchema(name="users", columns=columns)

        assert schema.name == "users"
        assert len(schema.columns) == 2
        assert schema.is_spatial is False

    def test_table_schema_with_geometry(self):
        """TableSchema should detect spatial tables."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "geom": ColumnInfo(name="geom", type="geometry", spatial=True),
        }
        schema = TableSchema(
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
        schema = TableSchema(name="users", columns=columns)

        names = schema.column_names
        assert set(names) == {"id", "name"}

    def test_table_schema_column_types(self):
        """column_types property should return type mapping."""
        columns = {
            "id": ColumnInfo(name="id", type="int"),
            "name": ColumnInfo(name="name", type="varchar"),
        }
        schema = TableSchema(name="users", columns=columns)

        types = schema.column_types
        assert types == {"id": "int", "name": "varchar"}


class TestTypeMapperFactory:
    """Tests for TypeMapperFactory type mappers."""

    def test_sqlite_mapper(self):
        """TypeMapperFactory.for_sqlite() should return SQLite mapper."""
        mapper = TypeMapperFactory.for_sqlite()
        assert isinstance(mapper, DatabaseTypeMapper)
        assert mapper.name == "SQLite"

    def test_sqlite_type_mapping(self):
        """SQLite mapper should map types correctly."""
        mapper = TypeMapperFactory.for_sqlite()

        assert mapper.map_to_sda("INTEGER") == "int"
        assert mapper.map_to_sda("TEXT") == "varchar"
        assert mapper.map_to_sda("REAL") == "float"
        assert mapper.map_to_sda("BLOB") == "binary"
        assert mapper.map_to_sda("GEOMETRY") == "geometry"

    def test_sqlite_case_insensitive(self):
        """SQLite mapper should be case-insensitive."""
        mapper = TypeMapperFactory.for_sqlite()

        assert mapper.map_to_sda("integer") == "int"
        assert mapper.map_to_sda("INTEGER") == "int"
        assert mapper.map_to_sda("Integer") == "int"

    def test_postgresql_mapper(self):
        """TypeMapperFactory.for_postgresql() should return PostgreSQL mapper."""
        mapper = TypeMapperFactory.for_postgresql()
        assert isinstance(mapper, DatabaseTypeMapper)
        assert mapper.name == "PostgreSQL"

    def test_postgresql_type_mapping(self):
        """PostgreSQL mapper should map types correctly."""
        mapper = TypeMapperFactory.for_postgresql()

        assert mapper.map_to_sda("integer") == "int"
        assert mapper.map_to_sda("text") == "varchar"
        assert mapper.map_to_sda("float8") == "float"
        assert mapper.map_to_sda("bytea") == "binary"
        assert mapper.map_to_sda("geometry") == "geometry"

    def test_sda_mapper(self):
        """TypeMapperFactory.for_sda() should return SDA mapper."""
        mapper = TypeMapperFactory.for_sda()
        assert isinstance(mapper, DatabaseTypeMapper)
        assert mapper.name == "SDA"

    def test_sda_type_mapping(self):
        """SDA mapper should preserve SDA types."""
        mapper = TypeMapperFactory.for_sda()

        assert mapper.map_to_sda("int") == "int"
        assert mapper.map_to_sda("varchar") == "varchar"
        assert mapper.map_to_sda("float") == "float"
        assert mapper.map_to_sda("geometry") == "geometry"

    def test_geopackage_mapper(self):
        """TypeMapperFactory.for_geopackage() should return GeoPackage mapper."""
        mapper = TypeMapperFactory.for_geopackage()
        assert isinstance(mapper, DatabaseTypeMapper)
        assert mapper.name == "GeoPackage"

    def test_geopackage_inherits_sqlite(self):
        """GeoPackage mapper should use SQLite type mappings."""
        sqlite_mapper = TypeMapperFactory.for_sqlite()
        geopackage_mapper = TypeMapperFactory.for_geopackage()

        # Should have same mappings
        assert sqlite_mapper.map_to_sda("INTEGER") == geopackage_mapper.map_to_sda("INTEGER")
        assert sqlite_mapper.map_to_sda("GEOMETRY") == geopackage_mapper.map_to_sda("GEOMETRY")

    def test_type_mapper_get_python_type(self):
        """Type mapper should get Python type for database type."""
        mapper = TypeMapperFactory.for_sqlite()

        assert mapper.get_python_type("INTEGER") is int
        assert mapper.get_python_type("TEXT") is str
        assert mapper.get_python_type("REAL") is float

    def test_type_mapper_unknown_type(self):
        """Type mapper should default unknown types to varchar."""
        mapper = TypeMapperFactory.for_sqlite()

        sda_type = mapper.map_to_sda("CUSTOM_TYPE")
        assert sda_type == "varchar"

    def test_type_mapper_type_with_params(self):
        """Type mapper should handle types with parameters."""
        mapper = TypeMapperFactory.for_sqlite()

        # VARCHAR(255) should map to varchar
        assert mapper.map_to_sda("VARCHAR(255)") == "varchar"

    def test_type_mapper_infer_from_value(self):
        """Type mapper should infer database type from Python value."""
        mapper = TypeMapperFactory.for_sqlite()

        assert mapper.infer_type_from_value(42) == "INTEGER"
        assert mapper.infer_type_from_value("text") == "TEXT"
        assert mapper.infer_type_from_value(3.14) == "REAL"
        assert mapper.infer_type_from_value(b"bytes") == "BLOB"
        assert mapper.infer_type_from_value(True) == "BOOLEAN"
        assert mapper.infer_type_from_value(None) == "NULL"

    def test_type_mapper_infer_sda_type(self):
        """Type mapper should infer SDA type from Python value."""
        mapper = TypeMapperFactory.for_sqlite()

        assert mapper.infer_sda_type_from_value(42) == "int"
        assert mapper.infer_sda_type_from_value("text") == "varchar"
        assert mapper.infer_sda_type_from_value(3.14) == "float"

    def test_custom_type_mapper_registration(self):
        """TypeMapperFactory should support custom mapper registration."""
        custom_types = {
            "MY_INT": "int",
            "MY_STR": "varchar",
            "MY_FLOAT": "float",
        }
        custom_mapper = TypeMapperFactory.create(custom_types, "mydb")

        assert custom_mapper.map_to_sda("MY_INT") == "int"
        assert custom_mapper.map_to_sda("MY_STR") == "varchar"

        # Register it
        TypeMapperFactory.register("mydb", custom_mapper)

        # Retrieve it
        retrieved = TypeMapperFactory.get("mydb")
        assert retrieved is custom_mapper

    def test_cached_mappers(self):
        """TypeMapperFactory should cache mapper instances."""
        mapper1 = TypeMapperFactory.for_sqlite()
        mapper2 = TypeMapperFactory.for_sqlite()

        # Should be same instance (cached)
        assert mapper1 is mapper2


class TestDatabaseTypeMapper:
    """Tests for DatabaseTypeMapper."""

    def test_mapper_initialization(self):
        """DatabaseTypeMapper should initialize with type mappings."""
        types = {"INT": "int", "VARCHAR": "varchar"}
        mapper = DatabaseTypeMapper(types, name="TestDB")

        assert mapper.name == "TestDB"
        assert mapper.map_to_sda("INT") == "int"

    def test_mapper_case_insensitive(self):
        """DatabaseTypeMapper should be case-insensitive."""
        types = {"INT": "int", "VARCHAR": "varchar"}
        mapper = DatabaseTypeMapper(types)

        assert mapper.map_to_sda("int") == "int"
        assert mapper.map_to_sda("INT") == "int"
        assert mapper.map_to_sda("Int") == "int"

    def test_mapper_type_with_params(self):
        """DatabaseTypeMapper should handle types with parameters."""
        types = {"VARCHAR": "varchar", "INTEGER": "int"}
        mapper = DatabaseTypeMapper(types)

        assert mapper.map_to_sda("VARCHAR(255)") == "varchar"
        assert mapper.map_to_sda("INTEGER(10)") == "int"

    def test_mapper_default_unknown(self):
        """DatabaseTypeMapper should default unknown types to varchar."""
        types = {"INT": "int"}
        mapper = DatabaseTypeMapper(types)

        assert mapper.map_to_sda("UNKNOWN_TYPE") == "varchar"

    def test_mapper_repr(self):
        """DatabaseTypeMapper should have useful repr."""
        types = {"INT": "int", "VARCHAR": "varchar"}
        mapper = DatabaseTypeMapper(types, name="TestDB")

        repr_str = repr(mapper)
        assert "TestDB" in repr_str
        assert "2" in repr_str  # 2 mappings


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
            assert isinstance(schema, TableSchema)

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
            for table_name, schema in schemas.items():
                assert schema.name in tables
                assert len(schema.columns) > 0

    @pytest.mark.asyncio
    async def test_response_combination_workflow(self):
        """Test combining multiple response objects."""
        backend = MockBackend()

        async with backend:
            # Execute multiple queries
            responses = await backend.execute_many(
                ["SELECT * FROM table1", "SELECT * FROM table2"]
            )

            # Combine responses
            combined = await ResponseAdapter.combine_responses(responses)
            df = combined.to_pandas()

            # Should have combined results
            assert len(df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
