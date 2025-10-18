"""
Automatic T-SQL type inference for schema generation.

This module provides automatic generation of ColumnSchema entries from SDA metadata
with correct Python type hints, appropriate processors, and proper null handling.
"""

from typing import Dict, Tuple, Type, Callable, Optional
from .schema_system import ColumnSchema, TableSchema, SCHEMAS

# T-SQL type mapping dictionary
T_SQL_TYPE_MAP = {
    # String types
    'varchar': ('str', 'to_str', False),
    'char': ('str', 'to_str', False),
    'text': ('str', 'to_str', False),
    'nvarchar': ('str', 'to_str', False),
    
    # Numeric types
    'int': ('int', 'to_optional_int', True),
    'bigint': ('int', 'to_optional_int', True),
    'float': ('float', 'to_optional_float', True),
    'real': ('float', 'to_optional_float', True),
    'numeric': ('float', 'to_optional_float', True),
    'decimal': ('float', 'to_optional_float', True),
    
    # Boolean
    'bit': ('bool', 'to_bool', False),
    
    # Date/Time (placeholder for now)
    'datetime': ('datetime', 'to_datetime', True),
    'datetime2': ('datetime', 'to_datetime', True),
    'date': ('date', 'to_date', True),
}
def infer_column_type(tsql_type: str) -> Tuple[Type, Callable, bool]:
    """Map T-SQL type to Python type hint, processor, and optional flag.

    Args:
        tsql_type: Raw T-SQL type string (e.g., 'varchar(255)', 'int')

    Returns:
        Tuple of (python_type, processor_function, is_optional)
    """
    # Parse base type from T-SQL type string
    base_type = tsql_type.split('(')[0].lower()

    if base_type in T_SQL_TYPE_MAP:
        type_name, processor_name, is_optional = T_SQL_TYPE_MAP[base_type]

        # Import the processor function dynamically
        from . import type_processors
        processor = getattr(type_processors, processor_name)

        # Get the Python type
        if type_name == 'str':
            python_type = str
        elif type_name == 'int':
            python_type = int
        elif type_name == 'float':
            python_type = float
        elif type_name == 'bool':
            python_type = bool
        elif type_name == 'datetime':
            from datetime import datetime
            python_type = datetime
        elif type_name == 'date':
            from datetime import date
            python_type = date
        else:
            python_type = str  # fallback

        # Make optional if needed
        if is_optional:
            from typing import Optional
            python_type = Optional[python_type]

        return python_type, processor, is_optional

    # Default fallback
    from . import type_processors
    return str, type_processors.to_str, False


def create_schema_from_sda_metadata(response, table_name: str) -> TableSchema:
    """Create a schema by extracting type info from SDA response metadata.

    Args:
        response: SDAResponse object with metadata
        table_name: Name of the table to create schema for

    Returns:
        Complete TableSchema with inferred column definitions
    """
    columns = {}

    # Extract metadata from response
    if hasattr(response, '_raw_data') and 'Table' in response._raw_data:
        table_data = response._raw_data['Table']
        if len(table_data) >= 2:
            column_names = table_data[0]
            metadata = table_data[1]

            for i, col_name in enumerate(column_names):
                if i < len(metadata):
                    # Parse metadata like "ColumnOrdinal=0,DataTypeName=varchar"
                    metadata_str = metadata[i]
                    if "DataTypeName=" in metadata_str:
                        type_part = metadata_str.split("DataTypeName=")[1]
                        tsql_type = (
                            type_part.split(",")[0] if "," in type_part else type_part
                        )

                        # Infer Python type and processor
                        python_type, processor, is_optional = infer_column_type(tsql_type)

                        # Create ColumnSchema
                        columns[col_name.lower()] = ColumnSchema(
                            name=col_name.lower(),
                            type_hint=python_type,
                            processor=processor,
                            default=True,  # Mark all auto-generated columns as default
                            description=f"Auto-inferred from T-SQL type: {tsql_type}"
                        )

    return TableSchema(
        name=table_name,
        columns=columns,
    )


def auto_register_schema(response, table_name: str) -> bool:
    """Automatically register a schema in SCHEMAS dict based on response.

    Args:
        response: SDAResponse with metadata
        table_name: Table name to register

    Returns:
        True if schema was registered, False otherwise
    """
    if table_name in SCHEMAS:
        return False  # Already exists

    try:
        schema = create_schema_from_sda_metadata(response, table_name)
        if schema.columns:  # Only register if we found columns
            SCHEMAS[table_name] = schema
            return True
    except Exception:
        pass  # Silently fail for now

    return False