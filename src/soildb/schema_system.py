"""
Schema-driven column mapping system for flexible data structures.

This module provides a completely automatic system for mapping database columns
to dataclass fields with minimal hardcoded logic.
"""

from dataclasses import dataclass, field, make_dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Type, Union
import pandas as pd


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    type_hint: Type
    processor: Callable[[Any], Any]
    default: bool = False
    field_name: Optional[str] = None  # Maps to dataclass field, None = extra_fields
    required: bool = False
    description: str = ""


@dataclass
class TableSchema:
    """Schema definition for a table/entity type."""
    name: str
    columns: Dict[str, ColumnSchema]
    base_fields: Dict[str, Any] = field(default_factory=dict)  # Fixed fields for dataclass

    def get_default_columns(self) -> List[str]:
        """Get list of default column names."""
        return [col.name for col in self.columns.values() if col.default]

    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        return [col.name for col in self.columns.values() if col.required]

    def process_row(self, row: pd.Series, requested_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a data row according to the schema."""
        result = dict(self.base_fields)  # Start with base fields
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

        result['extra_fields'] = extra_fields
        return result


# Define processors
def to_optional_float(value: Any) -> Optional[float]:
    """Convert to float, returning None if NaN."""
    return float(value) if pd.notna(value) else None

def to_optional_int(value: Any) -> Optional[int]:
    """Convert to int, returning None if NaN."""
    return int(value) if pd.notna(value) else None

def to_str(value: Any) -> str:
    """Convert to string."""
    return str(value) if pd.notna(value) else ""

def to_optional_str(value: Any) -> Optional[str]:
    """Convert to string or None."""
    return str(value) if pd.notna(value) else None

# Schema definitions
SCHEMAS = {
    "mapunit": TableSchema(
        name="mapunit",
        base_fields={
            'components': [],
            'extra_fields': {},
        },
        columns={
            "mukey": ColumnSchema("mukey", str, str, default=True, field_name="map_unit_key", required=True),
            "muname": ColumnSchema("muname", str, to_str, default=True, field_name="map_unit_name"),
            "musym": ColumnSchema("musym", Optional[str], to_optional_str, default=True, field_name="map_unit_symbol"),
            "lkey": ColumnSchema("lkey", str, str, default=True, field_name="survey_area_symbol"),
            "areaname": ColumnSchema("areaname", str, to_str, default=True, field_name="survey_area_name"),
            # Additional columns can be added dynamically
        }
    ),

    "component": TableSchema(
        name="component",
        base_fields={
            'aggregate_horizons': [],
            'extra_fields': {},
        },
        columns={
            "cokey": ColumnSchema("cokey", str, str, default=True, field_name="component_key", required=True),
            "compname": ColumnSchema("compname", str, to_str, default=True, field_name="component_name"),
            "comppct_r": ColumnSchema("comppct_r", float, to_optional_float, default=True, field_name="component_percentage"),
            "majcompflag": ColumnSchema("majcompflag", bool, lambda x: str(x).lower() == "yes", default=True, field_name="is_major_component"),
            "taxclname": ColumnSchema("taxclname", Optional[str], to_optional_str, default=True, field_name="taxonomic_class"),
            "drainagecl": ColumnSchema("drainagecl", Optional[str], to_optional_str, default=True, field_name="drainage_class"),
            "localphase": ColumnSchema("localphase", Optional[str], to_optional_str, default=True, field_name="local_phase"),
            "hydricrating": ColumnSchema("hydricrating", Optional[str], to_optional_str, default=True, field_name="hydric_rating"),
            "compkind": ColumnSchema("compkind", Optional[str], to_optional_str, default=True, field_name="component_kind"),
        }
    ),

    "chorizon": TableSchema(
        name="chorizon",
        base_fields={
            'properties': [],
            'extra_fields': {},
        },
        columns={
            "chkey": ColumnSchema("chkey", str, str, default=True, field_name="horizon_key", required=True),
            "hzname": ColumnSchema("hzname", str, to_str, default=True, field_name="horizon_name"),
            "hzdept_r": ColumnSchema("hzdept_r", float, to_optional_float, default=True, field_name="top_depth"),
            "hzdepb_r": ColumnSchema("hzdepb_r", float, to_optional_float, default=True, field_name="bottom_depth"),
            # Property columns
            "claytotal_r": ColumnSchema("claytotal_r", Optional[float], to_optional_float, default=True, field_name=None),  # Goes to properties
            "sandtotal_r": ColumnSchema("sandtotal_r", Optional[float], to_optional_float, default=True, field_name=None),
            "om_r": ColumnSchema("om_r", Optional[float], to_optional_float, default=True, field_name=None),
            "ph1to1h2o_r": ColumnSchema("ph1to1h2o_r", Optional[float], to_optional_float, default=True, field_name=None),
        }
    ),

    "pedon": TableSchema(
        name="pedon",
        base_fields={
            'horizons': [],
            'extra_fields': {},
        },
        columns={
            "pedon_key": ColumnSchema("pedon_key", str, str, default=True, field_name="pedon_key", required=True),
            "upedonid": ColumnSchema("upedonid", str, str, default=True, field_name="pedon_id", required=True),
            "corr_name": ColumnSchema("corr_name", Optional[str], to_optional_str, default=True, field_name="series"),
            "latitude_decimal_degrees": ColumnSchema("latitude_decimal_degrees", Optional[float], to_optional_float, default=True, field_name="latitude"),
            "longitude_decimal_degrees": ColumnSchema("longitude_decimal_degrees", Optional[float], to_optional_float, default=True, field_name="longitude"),
            "taxonname": ColumnSchema("taxonname", Optional[str], to_optional_str, default=True, field_name="soil_classification"),
        }
    ),

    "pedon_horizon": TableSchema(
        name="pedon_horizon",
        base_fields={
            'extra_fields': {},
        },
        columns={
            "pedon_key": ColumnSchema("pedon_key", str, str, default=True, field_name="pedon_key", required=True),
            "layer_key": ColumnSchema("layer_key", str, str, default=True, field_name="layer_key", required=True),
            "layer_sequence": ColumnSchema("layer_sequence", Optional[int], to_optional_int, default=True, field_name="layer_sequence"),
            "hzn_desgn": ColumnSchema("hzn_desgn", str, to_str, default=True, field_name="horizon_name"),
            "hzn_top": ColumnSchema("hzn_top", Optional[float], to_optional_float, default=True, field_name="top_depth"),
            "hzn_bot": ColumnSchema("hzn_bot", Optional[float], to_optional_float, default=True, field_name="bottom_depth"),
            "sand_total": ColumnSchema("sand_total", Optional[float], to_optional_float, default=True, field_name="sand_total"),
            "silt_total": ColumnSchema("silt_total", Optional[float], to_optional_float, default=True, field_name="silt_total"),
            "clay_total": ColumnSchema("clay_total", Optional[float], to_optional_float, default=True, field_name="clay_total"),
            "texture_lab": ColumnSchema("texture_lab", Optional[str], to_optional_str, default=True, field_name="texture_lab"),
            "ph_h2o": ColumnSchema("ph_h2o", Optional[float], to_optional_float, default=True, field_name="ph_h2o"),
            "total_carbon_ncs": ColumnSchema("total_carbon_ncs", Optional[float], to_optional_float, default=True, field_name=None),  # Processed
            "organic_carbon_walkley_black": ColumnSchema("organic_carbon_walkley_black", Optional[float], to_optional_float, default=True, field_name=None),
            "organic_carbon": ColumnSchema("organic_carbon", Optional[float], to_optional_float, default=True, field_name="organic_carbon"),  # Computed from carbon sources
            "caco3_lt_2_mm": ColumnSchema("caco3_lt_2_mm", Optional[float], to_optional_float, default=True, field_name="calcium_carbonate"),
            "bulk_density_third_bar": ColumnSchema("bulk_density_third_bar", Optional[float], to_optional_float, default=True, field_name="bulk_density_third_bar"),
            "le_third_fifteen_lt2_mm": ColumnSchema("le_third_fifteen_lt2_mm", Optional[float], to_optional_float, default=True, field_name="le_third_fifteen_lt2_mm"),
            "water_retention_tenth_bar": ColumnSchema("water_retention_tenth_bar", Optional[float], to_optional_float, default=True, field_name="water_content_tenth_bar"),
            "water_retention_third_bar": ColumnSchema("water_retention_third_bar", Optional[float], to_optional_float, default=True, field_name="water_content_third_bar"),
            "water_retention_15_bar": ColumnSchema("water_retention_15_bar", Optional[float], to_optional_float, default=True, field_name="water_content_fifteen_bar"),
        }
    ),

    "horizon_property": TableSchema(
        name="horizon_property",
        base_fields={
            'extra_fields': {},
        },
        columns={
            "property_name": ColumnSchema("property_name", str, to_str, default=True, field_name="property_name", required=True),
            "rv": ColumnSchema("rv", Optional[float], to_optional_float, default=True, field_name="rv"),
            "low": ColumnSchema("low", Optional[float], to_optional_float, default=True, field_name="low"),
            "high": ColumnSchema("high", Optional[float], to_optional_float, default=True, field_name="high"),
            "unit": ColumnSchema("unit", str, to_str, default=True, field_name="unit"),
        }
    ),

    "aggregate_horizon": TableSchema(
        name="aggregate_horizon",
        base_fields={
            'properties': [],
            'extra_fields': {},
        },
        columns={
            "chkey": ColumnSchema("chkey", str, str, default=True, field_name="horizon_key", required=True),
            "hzname": ColumnSchema("hzname", str, to_str, default=True, field_name="horizon_name"),
            "hzdept_r": ColumnSchema("hzdept_r", float, to_optional_float, default=True, field_name="top_depth"),
            "hzdepb_r": ColumnSchema("hzdepb_r", float, to_optional_float, default=True, field_name="bottom_depth"),
        }
    ),

    "map_unit_component": TableSchema(
        name="map_unit_component",
        base_fields={
            'aggregate_horizons': [],
            'extra_fields': {},
        },
        columns={
            "cokey": ColumnSchema("cokey", str, str, default=True, field_name="component_key", required=True),
            "compname": ColumnSchema("compname", str, to_str, default=True, field_name="component_name"),
            "comppct_r": ColumnSchema("comppct_r", float, to_optional_float, default=True, field_name="component_percentage"),
            "majcompflag": ColumnSchema("majcompflag", bool, lambda x: str(x).lower() == "yes", default=True, field_name="is_major_component"),
            "taxclname": ColumnSchema("taxclname", Optional[str], to_optional_str, default=True, field_name="taxonomic_class"),
            "drainagecl": ColumnSchema("drainagecl", Optional[str], to_optional_str, default=True, field_name="drainage_class"),
            "localphase": ColumnSchema("localphase", Optional[str], to_optional_str, default=True, field_name="local_phase"),
            "hydricrating": ColumnSchema("hydricrating", Optional[str], to_optional_str, default=True, field_name="hydric_rating"),
            "compkind": ColumnSchema("compkind", Optional[str], to_optional_str, default=True, field_name="component_kind"),
        }
    ),

    "soil_map_unit": TableSchema(
        name="soil_map_unit",
        base_fields={
            'components': [],
            'extra_fields': {},
        },
        columns={
            "mukey": ColumnSchema("mukey", str, str, default=True, field_name="map_unit_key", required=True),
            "muname": ColumnSchema("muname", str, to_str, default=True, field_name="map_unit_name"),
            "musym": ColumnSchema("musym", Optional[str], to_optional_str, default=True, field_name="map_unit_symbol"),
            "lkey": ColumnSchema("lkey", str, str, default=True, field_name="survey_area_symbol"),
            "areaname": ColumnSchema("areaname", str, to_str, default=True, field_name="survey_area_name"),
        }
    ),
}


def create_dynamic_dataclass(schema: TableSchema, name: str, base_class: Optional[Type] = None) -> Type:
    """Create a dataclass dynamically from a schema.

    Args:
        schema: The table schema to create the dataclass from
        name: Name for the new dataclass
        base_class: Optional base class to inherit from (for complex models)
    """
    from dataclasses import Field
    
    fields = []

    # Add base fields
    for fname, default_value in schema.base_fields.items():
        if fname == 'extra_fields':
            fields.append((fname, Dict[str, Any], field(default_factory=dict)))
        elif isinstance(default_value, list):
            fields.append((fname, List[Any], field(default_factory=list)))
        else:
            fields.append((fname, type(default_value), default_value))

    # Add schema-defined fields with proper defaults for Optional types
    for col_schema in schema.columns.values():
        if col_schema.field_name and col_schema.field_name not in [f[0] for f in fields]:
            # For Optional types or fields, use None as default if not explicitly set
            default_val = None
            # Check if type hint is Optional by looking at string representation
            type_str = str(col_schema.type_hint)
            if 'Optional' in type_str or 'Union' in type_str or 'None' in type_str:
                default_val = None
            elif col_schema.field_name == 'unit':
                # Special case: unit field defaults to ""
                default_val = ""
            else:
                default_val = None
            
            fields.append((col_schema.field_name, col_schema.type_hint, default_val))

    # Define utility methods
    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)

    # Create the dataclass
    if base_class:
        # If we have a base class, create a subclass
        base_dict = {}
        for fname, ftype, default_val in fields:
            if isinstance(default_val, Field):
                # Handle field() objects
                base_dict[fname] = default_val
            else:
                base_dict[fname] = field(default=default_val)

        # Add methods directly to class dict
        base_dict['get_extra_field'] = get_extra_field
        base_dict['has_extra_field'] = has_extra_field
        base_dict['list_extra_fields'] = list_extra_fields
        base_dict['to_dict'] = to_dict

        # Create the class
        DynamicClass = type(name, (base_class,), base_dict)
        return DynamicClass
    else:
        # Use make_dataclass for the basic structure
        dataclass_fields = []
        for fname, ftype, default_val in fields:
            if isinstance(default_val, Field):
                dataclass_fields.append((fname, ftype, default_val))
            else:
                dataclass_fields.append((fname, ftype, default_val))
        
        DynamicClass = make_dataclass(name, dataclass_fields)
        
        # Add methods to the class
        DynamicClass.get_extra_field = get_extra_field
        DynamicClass.has_extra_field = has_extra_field
        DynamicClass.list_extra_fields = list_extra_fields
        DynamicClass.to_dict = to_dict
        
        return DynamicClass


def add_column_to_schema(table_name: str, column_schema: ColumnSchema):
    """Add a new column to an existing schema."""
    if table_name in SCHEMAS:
        SCHEMAS[table_name].columns[column_schema.name] = column_schema


def get_schema(table_name: str) -> Optional[TableSchema]:
    """Get schema for a table."""
    return SCHEMAS.get(table_name)


# Create dynamic dataclasses from schemas
PedonHorizon = create_dynamic_dataclass(SCHEMAS["pedon_horizon"], "PedonHorizon")
HorizonProperty = create_dynamic_dataclass(SCHEMAS["horizon_property"], "HorizonProperty")
AggregateHorizon = create_dynamic_dataclass(SCHEMAS["aggregate_horizon"], "AggregateHorizon")
MapUnitComponent = create_dynamic_dataclass(SCHEMAS["map_unit_component"], "MapUnitComponent")
SoilMapUnit = create_dynamic_dataclass(SCHEMAS["soil_map_unit"], "SoilMapUnit")


# Export all dynamically created models
__all__ = [
    "PedonHorizon",
    "HorizonProperty",
    "AggregateHorizon",
    "MapUnitComponent",
    "SoilMapUnit",
    "SCHEMAS",
    "ColumnSchema",
    "TableSchema",
    "create_dynamic_dataclass",
    "add_column_to_schema",
    "get_schema",
]
