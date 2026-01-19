"""Database-specific type mappers for backend result conversion.

This module provides type mapping between different database systems and
the unified SDA type system. Each backend can declare which database it uses,
and get an appropriate type mapper for converting results.

## Type Mapping Hierarchy

Each database type mapper converts native types to SDA types:
- SQLite: INTEGER -> int, TEXT -> varchar, REAL -> float, BLOB -> binary
- PostgreSQL: integer -> int, text -> varchar, float8 -> float, geometry -> geometry
- SDA: Already in SDA types (no mapping needed)

The ResponseAdapter then uses the default TypeMap to convert from SDA types
to Python types, enabling unified response handling.

## Usage

```python
from soildb.backends.type_mapper import TypeMapperFactory

# Get mapper for SQLite
sqlite_mapper = TypeMapperFactory.for_sqlite()
python_type = sqlite_mapper.get_python_type("INTEGER")  # Returns int

# Get mapper for PostgreSQL
pg_mapper = TypeMapperFactory.for_postgresql()
python_type = pg_mapper.get_python_type("integer")  # Returns int

# Unknown types
sda_type = sqlite_mapper.map_to_sda("custom_type")  # Returns "varchar"
```

## Database-Specific Behaviors

### SQLite
- Limited type system (NULL, INTEGER, REAL, TEXT, BLOB)
- Case-insensitive type names
- All types represented as strings in introspection
- Geometry stored as BLOB (WKB) or TEXT (WKT)

### PostgreSQL
- Rich type system with user-defined types
- Geometry/Geography types with PostGIS
- Array types, JSON/JSONB, ranges, etc.
- Type schema includes namespace (e.g., "public"."geometry")

### GeoPackage
- Based on SQLite with OGC conventions
- Geometry column detection via gpkg_geometry_columns
- All constraints in SQLite format
- Standard SQLite type system

### SDA
- SQL Server types (int, varchar, float, datetime, etc.)
- Geometry as VARCHAR(MAX) containing WKT
- All types from standard SQL Server type system
"""

from typing import Any, Callable, Dict, Optional, Type

from soildb.type_conversion import TypeMap, TypeProcessor


class DatabaseTypeMapper:
    """
    Maps database-native types to SDA types and Python types.

    Each database has its own type system. This class handles the mapping
    from native types to the unified SDA type system, which the ResponseAdapter
    then uses with the default TypeMap for Python conversion.
    """

    def __init__(
        self,
        native_to_sda: Dict[str, str],
        name: str = "unknown",
    ):
        """
        Initialize a database type mapper.

        Args:
            native_to_sda: Mapping from database type name to SDA type name
            name: Display name for this mapper (e.g., "SQLite", "PostgreSQL")
        """
        self.name = name
        self._native_to_sda = {k.lower(): v.lower() for k, v in native_to_sda.items()}
        self._default_sda_map = TypeMap.default()

    def map_to_sda(self, native_type: str) -> str:
        """
        Map a database-native type to an SDA type.

        Args:
            native_type: Database-native type name (case-insensitive)

        Returns:
            SDA type name (e.g., "int", "varchar", "float")
            Defaults to "varchar" for unknown types
        """
        # Try exact match first
        sda_type = self._native_to_sda.get(native_type.lower())
        if sda_type:
            return sda_type

        # Try partial match (e.g., "INTEGER(10)" -> "INTEGER")
        native_base = native_type.lower().split("(")[0].strip()
        sda_type = self._native_to_sda.get(native_base)
        if sda_type:
            return sda_type

        # Default to varchar for unknown types
        return "varchar"

    def get_python_type(self, native_type: str) -> Type:
        """
        Get the Python type for a database-native type.

        Args:
            native_type: Database-native type name

        Returns:
            Python type (int, str, float, bool, datetime, bytes)
        """
        sda_type = self.map_to_sda(native_type)
        return self._default_sda_map.get_python_type(sda_type)

    def infer_type_from_value(self, value: Any) -> str:
        """
        Infer a database type from a Python value.

        Args:
            value: Python value from database result

        Returns:
            Inferred database native type name
        """
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, bytes):
            return "BLOB"
        else:
            return "TEXT"

    def infer_sda_type_from_value(self, value: Any) -> str:
        """
        Infer an SDA type from a Python value.

        Args:
            value: Python value from database result

        Returns:
            Inferred SDA type name
        """
        native_type = self.infer_type_from_value(value)
        return self.map_to_sda(native_type)

    def __repr__(self) -> str:
        """String representation."""
        return f"DatabaseTypeMapper({self.name}, mappings={len(self._native_to_sda)})"


class TypeMapperFactory:
    """
    Factory for creating database-specific type mappers.

    This provides standardized type mappers for common database systems,
    and allows registration of custom mappers for other systems.
    """

    # SQLite type mappings
    _SQLITE_TYPES = {
        # NULL - handled separately
        "null": "varchar",  # Treated as NULL
        # Integer types
        "integer": "int",
        "int": "int",
        "tinyint": "int",
        "smallint": "int",
        "mediumint": "int",
        "bigint": "int",
        "int2": "int",
        "int4": "int",
        "int8": "int",
        # Float types
        "real": "float",
        "float": "float",
        "double": "float",
        "numeric": "float",
        "decimal": "float",
        # Boolean (SQLite stores as 0/1)
        "boolean": "bit",
        "bool": "bit",
        # Text types
        "text": "varchar",
        "varchar": "varchar",
        "char": "varchar",
        "character": "varchar",
        "nchar": "varchar",
        "nvarchar": "varchar",
        # Blob types
        "blob": "binary",
        "bytea": "binary",
        # Date/Time
        "datetime": "datetime",
        "date": "date",
        "time": "time",
        "timestamp": "timestamp",
        # Spatial (OGC GeoPackage standard)
        "geometry": "geometry",
        "geography": "geography",
        "geom": "geometry",
    }

    # PostgreSQL type mappings
    _POSTGRESQL_TYPES = {
        # Integer types
        "integer": "int",
        "int": "int",
        "int2": "int",
        "int4": "int",
        "int8": "int",
        "smallint": "int",
        "bigint": "int",
        "serial": "int",
        "serial2": "int",
        "serial4": "int",
        "serial8": "int",
        # Float types
        "real": "float",
        "float4": "float",
        "float8": "float",
        "double precision": "float",
        "numeric": "float",
        "decimal": "float",
        # Boolean
        "boolean": "bit",
        "bool": "bit",
        # Text types
        "text": "varchar",
        "varchar": "varchar",
        "char": "varchar",
        "character": "varchar",
        "character varying": "varchar",
        "name": "varchar",
        # Bytea
        "bytea": "binary",
        # Date/Time
        "datetime": "datetime",
        "timestamp": "datetime",
        "timestamp without time zone": "datetime",
        "timestamp with time zone": "datetime",
        "timestamptz": "datetime",
        "date": "date",
        "time": "time",
        "time without time zone": "time",
        "time with time zone": "time",
        "timetz": "time",
        # JSON
        "json": "varchar",
        "jsonb": "varchar",
        # Arrays (treat as text)
        "integer[]": "varchar",
        "text[]": "varchar",
        "float8[]": "varchar",
        # Ranges
        "int4range": "varchar",
        "int8range": "varchar",
        "numrange": "varchar",
        "daterange": "varchar",
        "tsrange": "varchar",
        "tstzrange": "varchar",
        # Spatial (PostGIS)
        "geometry": "geometry",
        "geography": "geography",
        "point": "geometry",
        "line": "geometry",
        "lseg": "geometry",
        "box": "geometry",
        "path": "geometry",
        "polygon": "geometry",
        "circle": "geometry",
        # UUID
        "uuid": "varchar",
        "uniqueidentifier": "varchar",
    }

    # SDA (SQL Server) type mappings - already in SDA format
    _SDA_TYPES = {
        # Integer types
        "int": "int",
        "integer": "int",
        "tinyint": "int",
        "smallint": "int",
        "bigint": "int",
        # Boolean
        "bit": "bit",
        # Float types
        "float": "float",
        "real": "float",
        "double": "float",
        "decimal": "float",
        "numeric": "float",
        "money": "float",
        "smallmoney": "float",
        # String types
        "varchar": "varchar",
        "nvarchar": "varchar",
        "char": "varchar",
        "nchar": "varchar",
        "text": "varchar",
        "ntext": "varchar",
        # Date/Time
        "datetime": "datetime",
        "datetime2": "datetime",
        "smalldatetime": "datetime",
        "date": "date",
        "time": "time",
        "timestamp": "timestamp",
        # Spatial
        "geometry": "geometry",
        "geography": "geography",
        # Binary
        "varbinary": "binary",
        "binary": "binary",
        "image": "binary",
        # Other
        "uniqueidentifier": "varchar",
        "xml": "varchar",
    }

    # GeoPackage type mappings (inherits from SQLite)
    _GEOPACKAGE_TYPES = _SQLITE_TYPES.copy()

    _mappers: Dict[str, DatabaseTypeMapper] = {}

    @classmethod
    def for_sqlite(cls) -> DatabaseTypeMapper:
        """Get the SQLite type mapper."""
        if "sqlite" not in cls._mappers:
            cls._mappers["sqlite"] = DatabaseTypeMapper(
                cls._SQLITE_TYPES, name="SQLite"
            )
        return cls._mappers["sqlite"]

    @classmethod
    def for_postgresql(cls) -> DatabaseTypeMapper:
        """Get the PostgreSQL type mapper."""
        if "postgresql" not in cls._mappers:
            cls._mappers["postgresql"] = DatabaseTypeMapper(
                cls._POSTGRESQL_TYPES, name="PostgreSQL"
            )
        return cls._mappers["postgresql"]

    @classmethod
    def for_postgis(cls) -> DatabaseTypeMapper:
        """Get the PostGIS (PostgreSQL with spatial) type mapper.

        PostGIS uses PostgreSQL types with additional spatial support.
        """
        return cls.for_postgresql()

    @classmethod
    def for_sda(cls) -> DatabaseTypeMapper:
        """Get the SDA (SQL Server) type mapper."""
        if "sda" not in cls._mappers:
            cls._mappers["sda"] = DatabaseTypeMapper(cls._SDA_TYPES, name="SDA")
        return cls._mappers["sda"]

    @classmethod
    def for_geopackage(cls) -> DatabaseTypeMapper:
        """Get the GeoPackage type mapper.

        GeoPackage is OGC-compliant SQLite with spatial support.
        It uses SQLite types with conventions for geometry columns.
        """
        if "geopackage" not in cls._mappers:
            cls._mappers["geopackage"] = DatabaseTypeMapper(
                cls._GEOPACKAGE_TYPES, name="GeoPackage"
            )
        return cls._mappers["geopackage"]

    @classmethod
    def register(cls, name: str, mapper: DatabaseTypeMapper) -> None:
        """
        Register a custom database type mapper.

        Args:
            name: Name for this mapper (e.g., "my_database")
            mapper: DatabaseTypeMapper instance

        Example:
            >>> custom_types = {"CUSTOM_INT": "int", "CUSTOM_STR": "varchar"}
            >>> mapper = DatabaseTypeMapper(custom_types, name="MyDB")
            >>> TypeMapperFactory.register("mydb", mapper)
            >>> mapper = TypeMapperFactory.get("mydb")
        """
        cls._mappers[name.lower()] = mapper

    @classmethod
    def get(cls, name: str) -> Optional[DatabaseTypeMapper]:
        """
        Get a registered mapper by name.

        Args:
            name: Mapper name (case-insensitive)

        Returns:
            DatabaseTypeMapper or None if not found
        """
        return cls._mappers.get(name.lower())

    @classmethod
    def create(
        cls,
        native_to_sda: Dict[str, str],
        name: str = "custom",
    ) -> DatabaseTypeMapper:
        """
        Create a custom type mapper for a database system.

        Args:
            native_to_sda: Mapping from native type names to SDA type names
            name: Display name for this mapper

        Returns:
            New DatabaseTypeMapper instance

        Example:
            >>> custom_types = {
            ...     "MY_INT": "int",
            ...     "MY_STR": "varchar",
            ...     "MY_FLOAT": "float",
            ... }
            >>> mapper = TypeMapperFactory.create(custom_types, "mydb")
            >>> mapper.get_python_type("MY_INT")
            <class 'int'>
        """
        return DatabaseTypeMapper(native_to_sda, name=name)


__all__ = [
    "DatabaseTypeMapper",
    "TypeMapperFactory",
]
