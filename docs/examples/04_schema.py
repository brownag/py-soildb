"""
Example demonstrating schema inspection and usage.

This example shows how to use the schema_system module to work with
column schemas for different SSURGO tables.
"""

import asyncio

from soildb import SDAClient
from soildb.convenience import get_mapunit_by_areasymbol
from soildb.fetch import fetch_by_keys, get_mukey_by_areasymbol
from soildb.schema_system import SCHEMAS, get_schema


async def main():
    """Demonstrate schema system functionality."""
    async with SDAClient() as client:
        print("=== Schema System Example ===\n")

        # Get some mukeys for testing
        print("1. Getting mukeys for survey area CA630...")
        mukeys = await get_mukey_by_areasymbol(["CA630"], client=client)
        test_mukeys = mukeys[:3]  # Use just a few for demo
        print(f"   Found {len(mukeys)} mukeys, using {len(test_mukeys)} for demo\n")

        # Inspect available schemas
        print("2. Available schemas in registry:")
        schema_names = list(SCHEMAS.keys())
        print(f"   {len(schema_names)} schemas available")
        for schema_name in schema_names[:10]:
            print(f"     - {schema_name}")
        if len(schema_names) > 10:
            print(f"     ... and {len(schema_names) - 10} more")

        print()

        # Fetch mapunits and inspect schema
        print("3. Fetching mapunits from CA630...")
        mapunit_response = await get_mapunit_by_areasymbol("CA630", client=client)

        print(f"   Response has {len(mapunit_response.data)} rows")
        print(f"   Columns available: {len(mapunit_response.columns)}")
        print(f"   Sample columns: {mapunit_response.columns[:5]}")

        # Get the mapunit schema
        mapunit_schema = get_schema("mapunit")
        if mapunit_schema:
            print(
                f"   Mapunit schema has {len(mapunit_schema.columns)} defined columns\n"
            )

            # Show schema details for a few columns
            print("4. Schema details for selected mapunit columns:")
            for col_name, col_schema in list(mapunit_schema.columns.items())[:5]:
                required = "required" if col_schema.required else "optional"
                type_hint = col_schema.type_hint or "Any"
                print(f"     {col_name}: {type_hint} ({required})")

            if len(mapunit_schema.columns) > 5:
                print(f"     ... and {len(mapunit_schema.columns) - 5} more columns")
        else:
            print("   Mapunit schema not found in registry")

        print()

        # Fetch components using fetch_by_keys (the modern API)
        print("5. Fetching components using fetch_by_keys...")
        component_response = await fetch_by_keys(
            test_mukeys,
            table="component",
            key_column="mukey",
            client=client,
        )

        print(f"   Response has {len(component_response.data)} rows")
        print(f"   Columns: {component_response.columns[:5]}...")

        # Get the component schema
        component_schema = get_schema("component")
        if component_schema:
            print(f"   Component schema has {len(component_schema.columns)} columns")
            print(f"   Default columns: {component_schema.get_default_columns()[:5]}")
        else:
            print("   Component schema not found in registry")

        print()

        # Convert to pandas and show data types
        print("6. Data types in converted DataFrame:")
        df = mapunit_response.to_pandas()
        print(f"   DataFrame shape: {df.shape}")
        print("   Data types:")
        for col_name, dtype in list(df.dtypes.items())[:5]:
            print(f"     {col_name}: {dtype}")
        if len(df.dtypes) > 5:
            print(f"     ... and {len(df.dtypes) - 5} more columns")

        print("\n=== Example Complete ===")
        print("\nKey takeaways:")
        print("- Schemas provide column type information")
        print("- Use get_schema(table_name) to inspect a table's schema")
        print("- SCHEMAS dict contains all registered schemas")
        print("- Schema info is used for type conversion in to_pandas()/to_polars()")


if __name__ == "__main__":
    asyncio.run(main())
