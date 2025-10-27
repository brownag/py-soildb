"""
Spatial table schema definitions.

Defines schemas for spatial tables like mupolygon.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import to_optional_int, to_optional_str
from ._base import ColumnSchema, TableSchema

SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Map unit polygon schema
MUPOLYGON_SCHEMA = TableSchema(
    name="mupolygon",
    version=SCHEMA_VERSION,
    base_fields={
        "extra_fields": {},
    },
    columns={
        "mukey": ColumnSchema(
            "mukey",
            str,
            str,
            default=True,
            field_name="map_unit_key",
            required=True,
            description="Mapunit key",
        ),
        "musym": ColumnSchema(
            "musym",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="map_unit_symbol",
            description="Mapunit symbol",
        ),
        "areasymbol": ColumnSchema(
            "areasymbol",
            str,
            str,
            default=True,
            field_name="area_symbol",
            description="Area/survey symbol",
        ),
        "spatialversion": ColumnSchema(
            "spatialversion",
            Optional[int],
            to_optional_int,
            default=True,
            field_name="spatial_version",
            description="Spatial data version",
        ),
    },
)
