# AI Coding Agent Instructions for py-soildb

## Project Overview

py-soildb is an async Python client for USDA soil data services. It wraps three
remote APIs (SDA, AWDB, Henry) and supports local backends (SQLite, GeoPackage).
Key capabilities: SQL query building, spatial queries, bulk fetching with
automatic pagination, multi-format export (pandas/polars/SoilProfileCollection),
and lab pedon data access via the LDM subsystem.

**Status**: Alpha (v0.x). The high-level API surface is stabilizing but may still
change. Lower-level modules (Query builder, response, spatial) are more stable.
Breaking changes are acceptable during this period, but should be deliberate—
especially in the lower-level API.

See `README.md` for user-facing overview. See `CONTRIBUTING.md` for contributor
setup. See `docs/examples/` for runnable code samples.

## Architecture

### Data Hierarchy
- SSURGO: Legend (survey area) → Mapunit → Component → Horizon (chorizon)
- Lab Data (KSSL): Pedon Site → Pedon Horizon → Physical/Chemical properties

### API Tiers (prefer higher tiers when possible)

1. **High-level** (`high_level.py`) — returns nested dataclass objects
   (`SoilMapUnit` → `MapUnitComponent` → `AggregateHorizon`). Relationships
   are pre-fetched. Functions: `fetch_ssurgo_mapunit_by_point`,
   `fetch_labpedon_by_bbox`, `fetch_labpedon_by_id`.

2. **Mid-level** (`convenience.py`, `fetch.py`) — returns `SDAResponse`
   (export to DataFrame/dict/GeoDataFrame). Functions: `get_mapunit_by_*`,
   `get_sacatalog`, `fetch_by_keys`, `get_mukey_by_areasymbol`.

3. **Low-level** (`query.py`, `spatial.py`, `query_templates.py`) — raw
   `Query` builder, `spatial_query()`, template functions. Returns
   `SDAResponse` via `client.execute()`.

### Module Organization

Core:
- **base_client.py** — `BaseDataAccessClient` (ABC) and `ClientConfig` dataclass; all clients inherit from this
- **client.py** — `SDAClient` (httpx-based async HTTP client with retry logic)
- **query.py** — `Query` fluent SQL builder, `ColumnSets` for standard column groups
- **response.py** — `SDAResponse` with `.to_pandas()`, `.to_polars()`, `.to_dict()`, `.to_geodataframe()`, `.to_soilprofilecollection()`
- **spatial.py** — `spatial_query()`, `point_query()`, `bbox_query()`, `SpatialQueryBuilder`
- **exceptions.py** — `SoilDBError` → `SDANetworkError`, `SDAQueryError`, `SDAMaintenanceError`, etc.
- **sanitization.py** — input sanitization for SQL queries

Data access:
- **fetch.py** — `fetch_by_keys()` (primary bulk fetcher), key lookup helpers, `QueryPresets`
- **convenience.py** — `get_mapunit_by_*`, `get_sacatalog`, `get_lab_pedon*`
- **high_level.py** — nested-object API (`fetch_ssurgo_mapunit_by_point`, etc.)
- **query_templates.py** — factory functions for common SDA queries

Subsystems:
- **ldm/** — Lab Data Model client (`LDMClient`), query builder, table definitions, own exception hierarchy
- **backends/** — multi-database abstraction: `BaseBackend`, `SDABackend`, `SQLiteBackend`, `GeoPackageBackend`, `SSURGOClient`, `ResponseAdapter`
- **awdb/** — `AWDBClient` for SCAN/SNOTEL monitoring data; convenience functions: `discover_stations`, `discover_stations_nearby`, `get_property_data_near`, `get_soil_moisture_by_depth`, `station_sensors`, `station_available_properties`, `station_sensor_depths`
- **henry/** — `HenryClient` for Henry Mount Soil Climate Database (temporary NRCS sensor deployments)
- **wss.py** — `WSSClient` for downloading SSURGO/STATSGO ZIP archives from Web Soil Survey

Schema & types:
- **schemas/** — modular, lazy-loaded table schema definitions (`TableSchema`, `ColumnSchema`, registry)
- **schema_system.py** — dataclass definitions for nested objects (`SoilMapUnit`, `MapUnitComponent`, etc.)
- **type_conversion.py** — `TypeMap`, `TypeProcessor`, `convert_value`
- **type_processors.py** — individual type processor functions
- **spc_presets.py** — `ColumnConfig` presets for SoilProfileCollection conversion
- **spc_validator.py** — validation for SoilProfileCollection data

Infrastructure:
- **awdb_integration.py** — combined SDA + AWDB soil water availability workflows
- **metadata.py** — `SurveyMetadata` parsing from fgdcmetadata column
- **utils.py** — `@add_sync_version` decorator (sync bridge pattern)

### Sync Wrapper Pattern
Async functions decorated with `@add_sync_version` get a `.sync()` method:
```python
result = await get_mapunit_by_areasymbol("IA109")        # async
result = get_mapunit_by_areasymbol.sync("IA109")         # sync
```
The bridge auto-creates and closes the appropriate client if the function
has an optional `client` parameter with a type annotation.

## Developer Workflows

Run `make help` for all available commands. Key ones:

```bash
make install          # Dev install with all extras
make test             # pytest
make test-cov         # Coverage report
make lint             # ruff check + mypy
make lint-fix         # Auto-fix (pass RUFF_ARGS="--unsafe-fixes" for unsafe)
make format           # ruff format
make docs             # Build Quarto docs (includes validation)
make build            # Build distribution
make security         # bandit + safety
```

### Key Tools
- **Testing**: pytest + pytest-asyncio + pytest-httpx (mocking)
- **Linting**: ruff (linter + formatter) + mypy (type checking)
- **Documentation**: Quarto + quartodoc (API docs from docstrings)
- **Build**: hatchling
- **Dependencies**: httpx (async HTTP), aiosqlite (async SQLite)
- **Optional**: pandas, polars, geopandas, shapely, soilprofilecollection

## Conventions

### Async-First
- All public APIs are async; sync access via `.sync()` method
- Context manager pattern: `async with SDAClient() as client:`
- No blocking operations in async functions

### Error Handling
Exception hierarchy rooted at `SoilDBError`. Subsystem-specific:
- SDA: `SDANetworkError` (with `SDAConnectionError`, `SDATimeoutError`, `SDAMaintenanceError`), `SDAQueryError`, `SDAResponseError`
- LDM: `LDMError` → `LDMBackendError`, `LDMQueryError`, `LDMParameterError`, `LDMTableError`, `LDMResponseError`
- AWDB: `AWDBError` → `AWDBConnectionError`, `AWDBQueryError`
- Backends: `BackendError` → `BackendConnectionError`, `BackendQueryError`, `BackendSchemaError`
- WSS: `WSSDownloadError`

### Code Style
- Docstrings: NumPy format with Examples section
- Type hints: PEP 484 throughout (target Python ≥3.9)
- Module docstrings: brief description + architecture notes
- See `ColumnSets` in `query.py` for standard column group constants

### Testing
```bash
pytest tests/test_query.py -v       # Specific file
pytest -m "not integration"         # Skip network-dependent tests
pytest --timeout=30                 # With timeout
```
Integration tests require network access to SDA/AWDB services.

## Key Patterns

See `docs/examples/` for complete runnable examples covering basic queries,
spatial queries, metadata, schema, AWDB, query builder, fetch, and WSS download.

### Client Lifecycle
Always use context managers:
```python
async with SDAClient() as client:
    result = await client.execute(query)
```

### ClientConfig Presets
```python
from soildb import ClientConfig
config = ClientConfig.reliable()  # 120s timeout, 5 retries
# Also: .default() (60s, 3 retries), .fast() (30s, 1 retry)
```

### Bulk Fetch
Use `fetch_by_keys()` for any key-based bulk retrieval. It handles chunking,
concurrency, and retry with halved chunk size on failure.

### LDM (Lab Data)
`LDMClient(dsn=None)` — pass no DSN for SDA remote, or a path for local SQLite.
Two-stage queries: discover pedon_keys, then fetch layer data.

### Multi-Backend
```python
from soildb.backends import SQLiteBackend, SSURGOClient
async with SQLiteBackend("path/to/soil.db") as backend:
    client = SSURGOClient(backend)
    mapunits = await client.fetch_mapunit(areasymbol="IA109")
```

## Troubleshooting

- **SDA maintenance**: daily window ~12:45–1:00 AM Central Time
- **Timeout**: use `ClientConfig(timeout=120.0)` or `ClientConfig.reliable()`
- **No results**: verify areasymbol with `get_sacatalog()`, coordinates are WGS84 lon/lat
- **Type conversion**: check `TypeMap` in `type_conversion.py`; override via `type_map` parameter
</content>
<parameter name="filePath">/home/andrew/workspace/soilmcp/upstream/py-soildb/.github/copilot-instructions.md