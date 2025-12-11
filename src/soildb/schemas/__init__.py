"""
Schema system module.

Provides modular, lazy-loaded schema definitions for SSURGO tables.
Reduces monolithic schema definitions from 900+ lines to ~200 lines
by splitting into separate files and lazy-loading on demand.

Schema Structure:
- _base.py: Core classes (ColumnSchema, TableSchema)
- _registry.py: Lazy-loading registry and schema access functions
- mapunit.py, component.py, chorizon.py, pedon.py, property.py, spatial.py: Individual table schemas

Usage:
    from soildb.schemas import get_schema, list_available_schemas

    schema = get_schema("mapunit")
    available = list_available_schemas()

Schema Versioning:
    Each schema file includes SCHEMA_VERSION and LAST_UPDATED for tracking changes
"""

from ._base import ColumnSchema, TableSchema
from ._registry import (
    get_chorizon_schema,
    get_component_schema,
    get_mapunit_schema,
    get_pedon_horizon_schema,
    get_pedon_schema,
    get_schema,
    list_available_schemas,
)

__all__ = [
    # Base classes
    "ColumnSchema",
    "TableSchema",
    # Registry functions
    "get_schema",
    "list_available_schemas",
    # Convenience functions for commonly-used schemas
    "get_mapunit_schema",
    "get_component_schema",
    "get_chorizon_schema",
    "get_pedon_schema",
    "get_pedon_horizon_schema",
]
