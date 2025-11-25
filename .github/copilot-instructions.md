# AI Coding Agent Instructions for py-soildb

## Project Overview
py-soildb is an async Python client for the USDA-NRCS Soil Data Access (SDA) web service. It provides SQL query building, spatial queries, and bulk data fetching with pandas/polars export support.

## Architecture
- **Core**: `SDAClient` (httpx-based async HTTP client) executes `Query` objects against SDA endpoints
- **Data Hierarchy**: Mapunit -> Component -> Horizon (see `ColumnSets` in `query.py` for standard column groups)
- **Async First**: All public APIs are async; use `asyncio.run()` or nest_asyncio in Jupyter
- **Key Modules**:
  - `client.py`: HTTP client with retry logic
  - `query.py`: Fluent SQL builder with sanitization
  - `fetch.py`: Bulk fetching with automatic pagination
  - `spatial.py`: Geometry-based queries
  - `convenience.py`: High-level functions for common tasks

## Developer Workflows
- **Setup**: `make install` (installs dev dependencies)
- **Test**: `make test` (pytest with asyncio)
- **Lint/Format**: `make lint` (ruff + mypy), `make format` (ruff)
- **Docs**: `make docs` (Quarto build)
- **Build**: `make build` (hatchling)
- **Virtual Env**: Source `venv/bin/activate` before running

## Conventions
- **Async Everywhere**: Use `async def`, `await` for all SDA operations
- **Query Building**: Chain methods on `Query()` objects, e.g., `.select().from_().where()`
- **Error Handling**: Catch `SoilDBError` subclasses (`SDAConnectionError`, `SDAQueryError`, etc.)
- **Spatial**: Use WKT strings for geometries in `spatial_query()`
- **Bulk Operations**: Use `fetch_by_keys()` with `chunk_size` for large datasets
- **Imports**: `from soildb import SDAClient, Query, spatial_query`

## Examples
```python
# Basic query
async with SDAClient() as client:
    query = Query().select("mukey", "muname").from_("mapunit").where("areasymbol = 'IA109'")
    result = await client.execute(query)
    df = result.to_pandas()

# Spatial query
response = await spatial_query("POINT (-93.6 42.0)", "mupolygon", "intersects")

# Bulk fetch
mukeys = await get_mukey_by_areasymbol(["IA109", "IA113"])
components = await fetch_component_by_mukey(mukeys)
```</content>
<parameter name="filePath">/home/andrew/workspace/soilmcp/upstream/py-soildb/.github/copilot-instructions.md