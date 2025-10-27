"""
Mapunit table schema definition.

Defines the schema for SSURGO mapunit table including related metadata.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import to_optional_str, to_str
from ._base import ColumnSchema, TableSchema

# Schema version tracking
SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Mapunit schema - soil map unit table
MAPUNIT_SCHEMA = TableSchema(
    name="mapunit",
    version=SCHEMA_VERSION,
    base_fields={
        "components": [],
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
            description="Unique mapunit identifier",
        ),
        "muname": ColumnSchema(
            "muname",
            str,
            to_str,
            default=True,
            field_name="map_unit_name",
            description="Mapunit name",
        ),
        "musym": ColumnSchema(
            "musym",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="map_unit_symbol",
            description="Mapunit symbol (e.g., IA109A)",
        ),
        "lkey": ColumnSchema(
            "lkey",
            str,
            str,
            default=True,
            field_name="survey_area_symbol",
            description="Legend key (survey area identifier)",
        ),
        "areaname": ColumnSchema(
            "areaname",
            str,
            to_str,
            default=True,
            field_name="survey_area_name",
            description="Survey area name",
        ),
    },
)

# Soil map unit schema - simplified mapunit variant
SOIL_MAP_UNIT_SCHEMA = TableSchema(
    name="soil_map_unit",
    version=SCHEMA_VERSION,
    base_fields={
        "components": [],
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
        ),
        "muname": ColumnSchema(
            "muname", str, to_str, default=True, field_name="map_unit_name"
        ),
        "musym": ColumnSchema(
            "musym",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="map_unit_symbol",
        ),
        "lkey": ColumnSchema(
            "lkey", str, str, default=True, field_name="survey_area_symbol"
        ),
        "areaname": ColumnSchema(
            "areaname", str, to_str, default=True, field_name="survey_area_name"
        ),
    },
)
