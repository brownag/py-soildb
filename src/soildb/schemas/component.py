"""
Component table schema definition.

Defines the schema for SSURGO component table.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import to_optional_float, to_optional_str, to_str
from ._base import ColumnSchema, TableSchema

SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Component schema - soil component table
COMPONENT_SCHEMA = TableSchema(
    name="component",
    version=SCHEMA_VERSION,
    base_fields={
        "aggregate_horizons": [],
        "extra_fields": {},
    },
    columns={
        "cokey": ColumnSchema(
            "cokey",
            str,
            str,
            default=True,
            field_name="component_key",
            required=True,
            description="Unique component identifier",
        ),
        "compname": ColumnSchema(
            "compname",
            str,
            to_str,
            default=True,
            field_name="component_name",
            description="Component name",
        ),
        "comppct_r": ColumnSchema(
            "comppct_r",
            float,
            to_optional_float,
            default=True,
            field_name="component_percentage",
            description="Component percentage",
        ),
        "majcompflag": ColumnSchema(
            "majcompflag",
            bool,
            lambda x: str(x).lower() == "yes",
            default=True,
            field_name="is_major_component",
            description="Is major component flag",
        ),
        "taxclname": ColumnSchema(
            "taxclname",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="taxonomic_class",
            description="Taxonomic classification",
        ),
        "drainagecl": ColumnSchema(
            "drainagecl",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="drainage_class",
            description="Drainage class",
        ),
        "localphase": ColumnSchema(
            "localphase",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="local_phase",
            description="Local phase designation",
        ),
        "hydricrating": ColumnSchema(
            "hydricrating",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="hydric_rating",
            description="Hydric rating",
        ),
        "compkind": ColumnSchema(
            "compkind",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="component_kind",
            description="Component kind",
        ),
    },
)

# Map unit component schema - component variant with additional context
MAP_UNIT_COMPONENT_SCHEMA = TableSchema(
    name="map_unit_component",
    version=SCHEMA_VERSION,
    base_fields={
        "aggregate_horizons": [],
        "extra_fields": {},
    },
    columns={
        "cokey": ColumnSchema(
            "cokey",
            str,
            str,
            default=True,
            field_name="component_key",
            required=True,
        ),
        "compname": ColumnSchema(
            "compname", str, to_str, default=True, field_name="component_name"
        ),
        "comppct_r": ColumnSchema(
            "comppct_r",
            float,
            to_optional_float,
            default=True,
            field_name="component_percentage",
        ),
        "majcompflag": ColumnSchema(
            "majcompflag",
            bool,
            lambda x: str(x).lower() == "yes",
            default=True,
            field_name="is_major_component",
        ),
        "taxclname": ColumnSchema(
            "taxclname",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="taxonomic_class",
        ),
        "drainagecl": ColumnSchema(
            "drainagecl",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="drainage_class",
        ),
        "localphase": ColumnSchema(
            "localphase",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="local_phase",
        ),
        "hydricrating": ColumnSchema(
            "hydricrating",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="hydric_rating",
        ),
        "compkind": ColumnSchema(
            "compkind",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="component_kind",
        ),
    },
)
