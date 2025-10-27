"""
Horizon property table schema definition.

Defines the schema for horizon property tables.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import to_optional_float, to_str
from ._base import ColumnSchema, TableSchema

SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Horizon property schema
HORIZON_PROPERTY_SCHEMA = TableSchema(
    name="horizon_property",
    version=SCHEMA_VERSION,
    base_fields={
        "extra_fields": {},
    },
    columns={
        "property_name": ColumnSchema(
            "property_name",
            str,
            to_str,
            default=True,
            field_name="property_name",
            required=True,
            description="Property name",
        ),
        "rv": ColumnSchema(
            "rv",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="rv",
            description="Representative value",
        ),
        "low": ColumnSchema(
            "low",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="low",
            description="Low value",
        ),
        "high": ColumnSchema(
            "high",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="high",
            description="High value",
        ),
        "unit": ColumnSchema(
            "unit",
            str,
            to_str,
            default=True,
            field_name="unit",
            description="Unit of measurement",
        ),
    },
)
