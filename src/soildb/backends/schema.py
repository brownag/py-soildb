"""Schema introspection and representation.

Provides unified way to represent and query table schemas across
different database backends.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ColumnInfo:
    """Information about a single table column.

    Attributes:
        name: Column name
        type: Database type (e.g., 'INTEGER', 'VARCHAR', 'GEOMETRY')
        nullable: Whether column allows NULL values
        primary_key: Whether column is part of primary key
        spatial: Whether column contains spatial data
    """

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    spatial: bool = False

    @property
    def is_geometry(self) -> bool:
        """Check if column is spatial geometry."""
        if self.spatial:
            return True
        # Only BLOB with spatial flag set should be considered geometry
        type_upper = self.type.upper()
        return type_upper in (
            "GEOMETRY",
            "GEOGRAPHY",
            "GEOM",
            "SHAPE",
        )


@dataclass
class TableSchema:
    """Complete schema information for a table.

    Attributes:
        name: Table name
        columns: Dict of column_name -> ColumnInfo
        primary_key: Primary key column name(s)
        geometry_column: Name of geometry column if spatial, else None
        spatial_index: Type of spatial index if present (e.g., 'R-tree', 'GiST')
    """

    name: str
    columns: Dict[str, ColumnInfo]
    primary_key: Optional[str] = None
    geometry_column: Optional[str] = None
    spatial_index: Optional[str] = None

    @property
    def is_spatial(self) -> bool:
        """Check if table has spatial data."""
        return self.geometry_column is not None

    @property
    def column_names(self) -> list:
        """Get list of all column names."""
        return list(self.columns.keys())

    @property
    def column_types(self) -> Dict[str, str]:
        """Get dict of column_name -> type."""
        return {name: col.type for name, col in self.columns.items()}


class SchemaIntrospector:
    """Extract schema information from different database types.

    This is another key infrastructure piece. By centralizing schema
    introspection here, each backend only needs to implement
    simple get_columns() method - we handle the rest.
    """

    @staticmethod
    async def introspect_table(
        backend,  # BaseBackend instance
        table_name: str,
    ) -> TableSchema:
        """Get complete schema for a table.

        Args:
            backend: Backend instance with get_columns() method
            table_name: Name of table to introspect

        Returns:
            TableSchema with full column information
        """
        columns_dict = await backend.get_columns(table_name)

        # Convert to ColumnInfo objects
        columns = {}
        geometry_col = None

        for col_name, col_type in columns_dict.items():
            col_info = ColumnInfo(
                name=col_name,
                type=col_type,
                spatial=SchemaIntrospector._is_spatial_type(col_type),
            )
            columns[col_name] = col_info

            # Track geometry column
            if col_info.is_geometry:
                geometry_col = col_name

        return TableSchema(
            name=table_name,
            columns=columns,
            geometry_column=geometry_col,
        )

    @staticmethod
    async def introspect_database(
        backend,  # BaseBackend instance
    ) -> Dict[str, TableSchema]:
        """Get schema for all tables in database.

        Args:
            backend: Backend instance with get_tables() method

        Returns:
            Dict of table_name -> TableSchema
        """
        tables = await backend.get_tables()

        schemas = {}
        for table_name in tables:
            try:
                schemas[table_name] = await SchemaIntrospector.introspect_table(
                    backend, table_name
                )
            except Exception:
                # Skip tables we can't introspect
                continue

        return schemas

    @staticmethod
    def _is_spatial_type(db_type: str) -> bool:
        """Check if database type is spatial.

        Args:
            db_type: Database type string

        Returns:
            True if type represents spatial data
        """
        db_type_upper = db_type.upper()
        spatial_keywords = (
            "GEOMETRY",
            "GEOGRAPHY",
            "GEOM",
            "SHAPE",
            "POINT",
            "POLYGON",
            "LINESTRING",
            "MULTIPOINT",
            "MULTIPOLYGON",
            "MULTILINESTRING",
            "GEOMETRYCOLLECTION",
        )
        return any(keyword in db_type_upper for keyword in spatial_keywords)
