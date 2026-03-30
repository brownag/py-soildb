"""
Unified backend infrastructure for multi-database support.

This module provides the foundational components for supporting multiple database
backends (SDA, SQLite, GeoPackage, PostgreSQL, etc.) in a consistent way.

## Architecture

The backend layer consists of:

1. **BaseBackend** - Abstract base class defining the unified interface
2. **ResponseAdapter** - Converts any database result to SDAResponse
3. **SchemaIntrospector** - Database-agnostic schema discovery
4. **TypeMapperFactory** - Database-specific type mappings
5. **BackendError** - Semantic exception hierarchy

## Usage

```python
from soildb.backends import BaseBackend, ResponseAdapter, SchemaIntrospector

# Create a backend implementation
backend = SQLiteBackend("path/to/database.sqlite")

# Connect and get schema
async with backend:
    tables = await backend.get_tables()
    schema = await SchemaIntrospector.introspect_table(backend, "my_table")

    # Execute query and get SDAResponse
    response = await backend.execute("SELECT * FROM my_table")
    df = response.to_pandas()
```

## Implementing a New Backend

To add support for a new database system:

1. Create a backend class inheriting from BaseBackend
2. Implement required methods: connect(), execute(), get_tables(), get_columns()
3. Use ResponseAdapter to convert results to SDAResponse
4. Use SchemaIntrospector for schema discovery
5. Use TypeMapperFactory to get the appropriate type mapper

Example:

```python
from soildb.backends import (
    BaseBackend,
    ResponseAdapter,
    SchemaIntrospector,
    TypeMapperFactory,
    BackendConnectionError,
)

class MyDatabaseBackend(BaseBackend):
    def __init__(self, connection_string, config=None):
        super().__init__(config)
        self.connection_string = connection_string
        self.connection = None
        self.type_mapper = TypeMapperFactory.get("mydatabase")

    async def connect(self):
        try:
            self.connection = await mydatabase.connect(self.connection_string)
            return True
        except Exception as e:
            raise BackendConnectionError(f"Failed to connect: {e}") from e

    async def execute(self, sql):
        rows = await self.connection.fetch(sql)
        columns = [col[0] for col in rows.description]
        response = await ResponseAdapter.from_rows(rows, columns, self.type_mapper)
        return response

    async def get_tables(self):
        rows = await self.connection.fetch("SELECT table_name FROM information_schema.tables")
        return [row['table_name'] for row in rows]

    async def get_columns(self, table_name):
        rows = await self.connection.fetch(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
        )
        return {row['column_name']: row['data_type'] for row in rows}
```

## Backend Implementations

- **SDABackend** - Soil Data Access web service (HTTP API)
- **SQLiteBackend** - Local SQLite database snapshots
- **GeoPackageBackend** - OGC GeoPackage format (spatial SQLite)
- **PostgreSQLBackend** - PostgreSQL with PostGIS (future)

## Configuration

All backends accept an optional `ClientConfig` for:
- Timeouts and retry behavior
- Custom headers/authentication
- Performance tuning
- Logging levels

```python
from soildb import ClientConfig

config = ClientConfig(
    timeout=30,
    max_retries=3,
    chunk_size=1000,
)

backend = SQLiteBackend("database.sqlite", config=config)
```

## Error Handling

Backend exceptions follow a semantic hierarchy:

- **BackendError** - Base class for all backend errors
- **BackendConnectionError** - Connection failures
- **BackendQueryError** - Query execution failures
- **BackendSchemaError** - Schema introspection failures

```python
from soildb.backends import BackendConnectionError, BackendQueryError

try:
    async with backend:
        result = await backend.execute(sql)
except BackendConnectionError as e:
    # Handle connection issue
    pass
except BackendQueryError as e:
    # Handle query error
    pass
```
"""

from soildb.backends.base import BaseBackend
from soildb.backends.exceptions import (
    BackendConnectionError,
    BackendError,
    BackendQueryError,
    BackendSchemaError,
)
from soildb.backends.geopackage_backend import GeoPackageBackend
from soildb.backends.response_adapter import ResponseAdapter
from soildb.backends.schema import ColumnInfo, SchemaIntrospector, TableSchema
from soildb.backends.sda_backend import SDABackend
from soildb.backends.sqlite_backend import SQLiteBackend
from soildb.backends.type_mapper import DatabaseTypeMapper, TypeMapperFactory

__all__ = [
    # Base classes
    "BaseBackend",
    # Backend implementations
    "SDABackend",
    "SQLiteBackend",
    "GeoPackageBackend",
    # Response conversion
    "ResponseAdapter",
    # Schema introspection
    "SchemaIntrospector",
    "TableSchema",
    "ColumnInfo",
    # Type mapping
    "TypeMapperFactory",
    "DatabaseTypeMapper",
    # Exceptions
    "BackendError",
    "BackendConnectionError",
    "BackendQueryError",
    "BackendSchemaError",
]
