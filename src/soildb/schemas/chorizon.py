"""
Component horizon (chorizon) table schema definition.

Defines the schema for SSURGO chorizon table.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import to_optional_float, to_str
from ._base import ColumnSchema, TableSchema

SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Component horizon schema
CHORIZON_SCHEMA = TableSchema(
    name="chorizon",
    version=SCHEMA_VERSION,
    base_fields={
        "properties": [],
        "extra_fields": {},
    },
    columns={
        "chkey": ColumnSchema(
            "chkey",
            str,
            str,
            default=True,
            field_name="horizon_key",
            required=True,
            description="Unique component horizon identifier",
        ),
        "hzname": ColumnSchema(
            "hzname",
            str,
            to_str,
            default=True,
            field_name="horizon_name",
            description="Horizon designation (e.g., Ap, Bw)",
        ),
        "hzdept_r": ColumnSchema(
            "hzdept_r",
            float,
            to_optional_float,
            default=True,
            field_name="top_depth",
            description="Horizon top depth in cm",
        ),
        "hzdepb_r": ColumnSchema(
            "hzdepb_r",
            float,
            to_optional_float,
            default=True,
            field_name="bottom_depth",
            description="Horizon bottom depth in cm",
        ),
        # Property columns
        "claytotal_r": ColumnSchema(
            "claytotal_r",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="Clay percentage",
        ),
        "sandtotal_r": ColumnSchema(
            "sandtotal_r",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="Sand percentage",
        ),
        "om_r": ColumnSchema(
            "om_r",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="Organic matter percentage",
        ),
        "ph1to1h2o_r": ColumnSchema(
            "ph1to1h2o_r",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="pH in water",
        ),
    },
)

# Aggregate horizon schema - simplified variant
AGGREGATE_HORIZON_SCHEMA = TableSchema(
    name="aggregate_horizon",
    version=SCHEMA_VERSION,
    base_fields={
        "properties": [],
        "extra_fields": {},
    },
    columns={
        "chkey": ColumnSchema(
            "chkey", str, str, default=True, field_name="horizon_key", required=True
        ),
        "hzname": ColumnSchema(
            "hzname", str, to_str, default=True, field_name="horizon_name"
        ),
        "hzdept_r": ColumnSchema(
            "hzdept_r",
            float,
            to_optional_float,
            default=True,
            field_name="top_depth",
        ),
        "hzdepb_r": ColumnSchema(
            "hzdepb_r",
            float,
            to_optional_float,
            default=True,
            field_name="bottom_depth",
        ),
    },
)
