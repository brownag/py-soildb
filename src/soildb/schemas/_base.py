"""
Base classes for schema system.

Provides core schema infrastructure: ColumnSchema and TableSchema.
This module is shared by all schema definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ColumnSchema:
    """Schema definition for a single column.

    Attributes:
        name: Column name in the database
        type_hint: Python type hint for the column
        processor: Function to process/convert the column value
        default: Whether this is a default/commonly used column
        field_name: Dataclass field name (None = goes to extra_fields)
        required: Whether column is required for valid data
        description: Human-readable column description
    """

    name: str
    type_hint: Any
    processor: Callable[[Any], Any]
    default: bool = False
    field_name: Optional[str] = None
    required: bool = False
    description: str = ""


@dataclass
class TableSchema:
    """Schema definition for a table/entity type.

    Attributes:
        name: Table name in the database
        columns: Mapping of column name to ColumnSchema
        base_fields: Fixed fields for dataclass (e.g., lists, containers)
        version: Schema version for tracking updates
    """

    name: str
    columns: Dict[str, ColumnSchema]
    base_fields: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def get_default_columns(self) -> List[str]:
        """Get list of default column names."""
        return [col.name for col in self.columns.values() if col.default]

    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        return [col.name for col in self.columns.values() if col.required]

    def process_row(
        self, row: Any, requested_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a data row according to the schema.

        Args:
            row: Row data (dict-like with index)
            requested_columns: Which columns to process (defaults to default columns)

        Returns:
            Dictionary with processed data ready for dataclass initialization
        """
        result = dict(self.base_fields)
        extra_fields = {}

        # Determine which columns to process
        columns_to_process = requested_columns or self.get_default_columns()

        for col_name in columns_to_process:
            if col_name in row.index and col_name in self.columns:
                schema = self.columns[col_name]
                raw_value = row[col_name]

                # Apply processor
                processed_value = schema.processor(raw_value)

                # Map to field or extra_fields
                if schema.field_name:
                    result[schema.field_name] = processed_value
                else:
                    extra_fields[col_name] = processed_value
            elif col_name in row.index and col_name not in self.columns:
                # Pass through unknown-but-requested columns into extra_fields
                extra_fields[col_name] = row[col_name]

        result["extra_fields"] = extra_fields
        return result

    def get_column_by_field_name(self, field_name: str) -> Optional[ColumnSchema]:
        """Get column schema by dataclass field name."""
        for col in self.columns.values():
            if col.field_name == field_name:
                return col
        return None

    def get_columns_for_field(self, field_name: str) -> List[str]:
        """Get all database columns that map to a specific field."""
        return [
            col.name for col in self.columns.values() if col.field_name == field_name
        ]
