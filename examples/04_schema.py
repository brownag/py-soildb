"""
Example demonstrating automatic T-SQL type inference and schema generation.

This example shows how to use the auto_schema parameter in fetch functions
to automatically create ColumnSchema entries from SDA metadata.
"""

import asyncio

from soildb import SDAClient
from soildb.convenience import get_mapunit_by_areasymbol
from soildb.fetch import fetch_component_by_mukey, get_mukey_by_areasymbol
from soildb.schema_system import SCHEMAS, get_schema


async def main():
    """Demonstrate auto_schema functionality."""
    async with SDAClient() as client:
        print("=== T-SQL Type Inference Example ===\n")

        # Get some mukeys for testing
        print("1. Getting mukeys for survey area CA630...")
        mukeys = await get_mukey_by_areasymbol(["CA630"], client=client)
        test_mukeys = mukeys[:3]  # Use just a few for demo
        print(f"   Found {len(mukeys)} mukeys, using {len(test_mukeys)} for demo\n")

        # Remove existing schemas to demonstrate auto-registration
        print("2. Clearing existing schemas...")
        schemas_to_clear = ["component", "mapunit"]
        for schema_name in schemas_to_clear:
            if schema_name in SCHEMAS:
                del SCHEMAS[schema_name]
                print(f"   Cleared {schema_name} schema")

        print(f"   Remaining schemas: {list(SCHEMAS.keys())}\n")

        # Fetch components with auto_schema
        print("3. Fetching components with auto_schema=True...")
        component_response = await fetch_component_by_mukey(
            test_mukeys, auto_schema=True, client=client
        )

        print(f"   Response has {len(component_response.data)} rows")
        print(f"   Columns: {component_response.columns[:5]}...")  # Show first 5

        # Check if schema was auto-registered
        component_schema = get_schema("component")
        if component_schema:
            print(
                f"    Component schema auto-registered with {len(component_schema.columns)} columns"
            )
            print(
                f"   Default columns: {component_schema.get_default_columns()[:5]}..."
            )
        else:
            print("    Component schema not found")

        print()

        # Fetch mapunits with auto_schema
        print("4. Fetching mapunits with auto_schema=True...")
        mapunit_response = await get_mapunit_by_areasymbol(
            "CA630", auto_schema=True, client=client
        )

        print(f"   Response has {len(mapunit_response.data)} rows")
        print(f"   Columns: {mapunit_response.columns[:5]}...")  # Show first 5

        # Check if schema was auto-registered
        mapunit_schema = get_schema("mapunit")
        if mapunit_schema:
            print(
                f"    Mapunit schema auto-registered with {len(mapunit_schema.columns)} columns"
            )
            print(f"   Default columns: {mapunit_schema.get_default_columns()[:5]}...")
        else:
            print("    Mapunit schema not found")

        print()

        # Demonstrate schema inspection
        print("5. Inspecting auto-generated component schema...")
        if component_schema:
            print("   Column details:")
            for col_name, col_schema in list(component_schema.columns.items())[:5]:
                print(
                    f"     {col_name}: {col_schema.type_hint} (required: {col_schema.required})"
                )
            if len(component_schema.columns) > 5:
                print(f"     ... and {len(component_schema.columns) - 5} more columns")

        print()

        # Demonstrate that subsequent calls don't re-register
        print("6. Testing that auto_schema doesn't re-register existing schemas...")
        initial_schema_count = len(component_schema.columns) if component_schema else 0

        # Call again with auto_schema=True (result not used, just testing schema registration)
        await fetch_component_by_mukey(
            test_mukeys[:1],
            auto_schema=True,
            client=client,  # Just one mukey this time
        )

        final_schema_count = len(component_schema.columns) if component_schema else 0
        if initial_schema_count == final_schema_count:
            print("    Schema not re-registered (counts match)")
        else:
            print(
                f"    Schema was re-registered ({initial_schema_count} -> {final_schema_count})"
            )

        print("\n=== Example Complete ===")
        print("\nThe auto_schema feature automatically:")
        print("- Infers Python types from T-SQL metadata")
        print("- Creates ColumnSchema entries with appropriate processors")
        print("- Registers schemas in the SCHEMAS registry")
        print("- Handles nullability and default values correctly")
        print("- Only registers once per table (no duplicates)")


if __name__ == "__main__":
    asyncio.run(main())
