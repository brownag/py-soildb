#!/usr/bin/env python3
"""
Demonstration of the schema-driven column mapping system.
"""

import pandas as pd

from soildb.schema_system import (
    SCHEMAS,
    ColumnSchema,
    add_column_to_schema,
    create_dynamic_dataclass,
    get_schema,
)


def demo_schema_system():
    """Demonstrate the schema system functionality."""

    print("=== Schema-Driven Column Mapping System Demo ===\n")

    # 1. Show existing schemas
    print("1. Available Schemas:")
    for name, schema in SCHEMAS.items():
        default_cols = schema.get_default_columns()
        required_cols = schema.get_required_columns()
        print(f"   {name}: {len(schema.columns)} columns ({len(default_cols)} default, {len(required_cols)} required)")

    print()

    # 2. Create dynamic dataclass
    print("2. Creating Dynamic Dataclass:")
    MapUnitSchema = get_schema("mapunit")
    if MapUnitSchema:
        MapUnitClass = create_dynamic_dataclass(MapUnitSchema, "DynamicMapUnit")
        print(f"   Created class: {MapUnitClass.__name__}")
        print(f"   Fields: {[f.name for f in MapUnitClass.__dataclass_fields__.values()]}")

    print()

    # 3. Add custom column
    print("3. Adding Custom Column:")
    add_column_to_schema("component", ColumnSchema(
        name="custom_slope",
        type_hint=float,
        processor=lambda x: float(x) if pd.notna(x) else None,
        default=False,
        field_name=None  # Goes to extra_fields
    ))
    print("   Added 'custom_slope' column to component schema")

    print()

    # 4. Process sample data
    print("4. Processing Sample Data:")
    sample_row = pd.Series({
        "mukey": "123456",
        "muname": "Test Soil",
        "musym": "TS",
        "lkey": "CA001",
        "areaname": "Test Area",
        "unknown_col": "extra data"
    })

    if MapUnitSchema:
        processed = MapUnitSchema.process_row(sample_row)
        print(f"   Raw data: {dict(sample_row)}")
        print(f"   Processed: {processed}")

        # Create object
        mapunit = MapUnitClass(**processed)
        print(f"   Created object: {mapunit.map_unit_key} - {mapunit.map_unit_name}")
        print(f"   Extra fields: {mapunit.extra_fields}")

    print()

    # 5. Show component schema with custom column
    print("5. Component Schema with Custom Column:")
    ComponentSchema = get_schema("component")
    if ComponentSchema:
        print(f"   Columns: {list(ComponentSchema.columns.keys())}")
        print(f"   Default: {ComponentSchema.get_default_columns()}")
        print(f"   Required: {ComponentSchema.get_required_columns()}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_schema_system()
