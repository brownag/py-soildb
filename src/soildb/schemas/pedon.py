"""
Pedon and pedon horizon table schema definitions.

Defines schemas for SSURGO pedon (site) and pedon_horizon (layer) tables.
Version: 1.0 (Last updated: 2025-10-27)
"""

from typing import Optional

from ..type_processors import (
    to_optional_float,
    to_optional_int,
    to_optional_str,
    to_str,
)
from ._base import ColumnSchema, TableSchema

SCHEMA_VERSION = "1.0"
LAST_UPDATED = "2025-10-27"

# Pedon (site) schema
PEDON_SCHEMA = TableSchema(
    name="pedon",
    version=SCHEMA_VERSION,
    base_fields={
        "horizons": [],
        "extra_fields": {},
    },
    columns={
        "pedon_key": ColumnSchema(
            "pedon_key",
            str,
            str,
            default=True,
            field_name="pedon_key",
            required=True,
            description="Unique pedon identifier",
        ),
        "upedonid": ColumnSchema(
            "upedonid",
            str,
            str,
            default=True,
            field_name="pedon_id",
            required=True,
            description="User pedon ID",
        ),
        "corr_name": ColumnSchema(
            "corr_name",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="taxonname",
            description="Correlated soil series name",
        ),
        "latitude_decimal_degrees": ColumnSchema(
            "latitude_decimal_degrees",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="latitude",
            description="Latitude in decimal degrees",
        ),
        "longitude_decimal_degrees": ColumnSchema(
            "longitude_decimal_degrees",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="longitude",
            description="Longitude in decimal degrees",
        ),
        "taxonname": ColumnSchema(
            "taxonname",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="taxclname",
            description="Taxonomic class name",
        ),
    },
)

# Pedon horizon (layer) schema
PEDON_HORIZON_SCHEMA = TableSchema(
    name="pedon_horizon",
    version=SCHEMA_VERSION,
    base_fields={
        "extra_fields": {},
    },
    columns={
        "pedon_key": ColumnSchema(
            "pedon_key",
            str,
            str,
            default=True,
            field_name="pedon_key",
            required=True,
            description="Pedon identifier",
        ),
        "layer_key": ColumnSchema(
            "layer_key",
            str,
            str,
            default=True,
            field_name="layer_key",
            required=True,
            description="Unique layer/horizon identifier",
        ),
        "layer_sequence": ColumnSchema(
            "layer_sequence",
            Optional[int],
            to_optional_int,
            default=True,
            field_name="layer_sequence",
            description="Layer sequence number",
        ),
        "hzn_desgn": ColumnSchema(
            "hzn_desgn",
            str,
            to_str,
            default=True,
            field_name="horizon_name",
            description="Horizon designation",
        ),
        "hzn_top": ColumnSchema(
            "hzn_top",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="top_depth",
            description="Horizon top depth in cm",
        ),
        "hzn_bot": ColumnSchema(
            "hzn_bot",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="bottom_depth",
            description="Horizon bottom depth in cm",
        ),
        "sand_total": ColumnSchema(
            "sand_total",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="sand_total",
            description="Total sand percentage",
        ),
        "silt_total": ColumnSchema(
            "silt_total",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="silt_total",
            description="Total silt percentage",
        ),
        "clay_total": ColumnSchema(
            "clay_total",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="clay_total",
            description="Total clay percentage",
        ),
        "texture_lab": ColumnSchema(
            "texture_lab",
            Optional[str],
            to_optional_str,
            default=True,
            field_name="texture_lab",
            description="Laboratory texture classification",
        ),
        "ph_h2o": ColumnSchema(
            "ph_h2o",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="ph_h2o",
            description="pH in water",
        ),
        "total_carbon_ncs": ColumnSchema(
            "total_carbon_ncs",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="Total carbon from NCS",
        ),
        "organic_carbon_walkley_black": ColumnSchema(
            "organic_carbon_walkley_black",
            Optional[float],
            to_optional_float,
            default=True,
            field_name=None,
            description="Organic carbon (Walkley-Black method)",
        ),
        "organic_carbon": ColumnSchema(
            "organic_carbon",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="organic_carbon",
            description="Organic carbon (computed or direct)",
        ),
        "caco3_lt_2_mm": ColumnSchema(
            "caco3_lt_2_mm",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="calcium_carbonate",
            description="Calcium carbonate equivalent",
        ),
        "bulk_density_third_bar": ColumnSchema(
            "bulk_density_third_bar",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="bulk_density_third_bar",
            description="Bulk density at 1/3 bar",
        ),
        "le_third_fifteen_lt2_mm": ColumnSchema(
            "le_third_fifteen_lt2_mm",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="le_third_fifteen_lt2_mm",
            description="Liquid limit between 1/3 and 15 bar",
        ),
        "water_retention_third_bar": ColumnSchema(
            "water_retention_third_bar",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="water_content_third_bar",
            description="Water retention at 1/3 bar",
        ),
        "water_retention_15_bar": ColumnSchema(
            "water_retention_15_bar",
            Optional[float],
            to_optional_float,
            default=True,
            field_name="water_content_fifteen_bar",
            description="Water retention at 15 bar",
        ),
    },
)
