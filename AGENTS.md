# AGENTS.md

Navigation map and infrastructure guide for py-soildb agents.

**Project**: Async Python client for USDA soil data services (SDA, LDM, AWDB, Henry, WSS)  
**Status**: Alpha (v0.x) — lower-level APIs (Query, response, spatial) change less frequently than high-level convenience functions  
**Language**: Python ≥3.9 | **Build**: hatchling | **Test**: pytest + pytest-asyncio + pytest-httpx

## Quick Start

Essential commands (extract from `Makefile`):

```bash
make install              # Install with dev extras
make install-prod         # Install production dependencies only
make test                 # Run all unit tests (pytest)
make test-cov             # Run tests with coverage report (HTML + terminal)
make test-integration     # Run integration tests (requires network)
make lint-fix             # Auto-fix linting (ruff + mypy)
make format               # Format code
make format-check         # Check formatting without changes
make security             # Run security checks (bandit/safety)
make docs                 # Build Quarto documentation
make docs-serve           # Serve docs with live reload (watch mode)
make examples-validate    # Validate example scripts (import check)
make examples-test        # Run selected example scripts (network-dependent)
make all                  # Full pipeline (clean → install → lint-fix → test → build → docs → validate-examples)
```

Common pytest patterns:
```bash
pytest tests/test_query.py -v                # Single test file
pytest -m "not integration" -v               # Skip network-dependent tests
pytest tests/test_query.py::test_name -v    # Single test
pytest tests/ -v --cov=soildb               # Run with coverage
```

Setup: See `CONTRIBUTING.md` for detailed environment setup.

## Find Things

### Code Structure

```
src/soildb/
├── __init__.py              # Public API re-exports (__all__ list)
├── base_client.py           # BaseDataAccessClient and ClientConfig
├── client.py                # SDAClient (async HTTP to NRCS web service)
├── chunked.py               # Paginated query execution with batching (internal)
├── query.py                 # Query builder (fluent interface for SQL)
├── query_templates.py       # Pre-built query templates for common tasks
├── response.py              # SDAResponse (DataFrame/dict/GeoDataFrame export)
├── spatial.py               # Spatial filtering (point/bbox queries)
├── fetch.py                 # Bulk key-based queries with pagination
├── convenience.py           # Single/simple queries
├── high_level.py            # Complex workflows returning nested dataclasses
├── type_conversion.py       # Type mapping (SQL → Python)
├── schema_system.py         # Schema metadata system
├── metadata.py              # Survey metadata parsing and filtering
├── sanitization.py          # Input validation and SQL injection prevention
├── utils.py                 # Shared utility functions
├── wss.py                   # Web Soil Survey data download
├── ssurgo_client.py         # SSURGO data client (queries and metadata)
├── ssurgo_tables.py         # SSURGO table schemas
├── exceptions.py            # SoilDBError hierarchy
├── ldm/                     # Lab Data Model (KSSL pedon data)
│   ├── client.py            # LDMClient (multi-backend support)
│   ├── query_builder.py     # SQL query builder for lab data
│   ├── tables.py            # Lab data table schemas
│   ├── exceptions.py        # LDMError hierarchy
│   └── __init__.py
├── awdb/                    # AWDB/SCAN/SNOTEL monitoring data
│   ├── client.py            # AWDBClient
│   ├── convenience.py       # High-level AWDB queries
│   ├── models.py            # Data models (StationInfo, TimeSeriesDataPoint, etc.)
│   ├── exceptions.py        # AWDBError hierarchy
│   └── __init__.py
├── henry/                   # Henry (Mount Soil) climate database
│   ├── client.py            # HenryClient
│   ├── convenience.py       # High-level Henry queries
│   ├── models.py            # Data models
│   ├── utils.py             # Henry-specific utilities
│   ├── exceptions.py        # HenryError hierarchy
│   └── __init__.py
├── backends/                # Multi-database backends
│   ├── base.py              # BaseBackend interface
│   ├── sda_backend.py       # SDA web service backend
│   ├── sqlite_backend.py    # SQLite file backend
│   ├── geopackage_backend.py # GeoPackage vector backend
│   ├── schema.py            # Backend schema utilities
│   ├── exceptions.py        # BackendError hierarchy
│   └── __init__.py
├── schemas/                 # Table schemas with type metadata
│   ├── _base.py             # Base schema class
│   ├── pedon.py             # Lab pedon schema
│   ├── chorizon.py          # Component horizon schema
│   ├── component.py         # Map unit component schema
│   ├── mapunit.py           # Map unit schema
│   ├── property.py          # Soil property schema
│   ├── spatial.py           # Spatial data schema
│   ├── _registry.py         # Schema registry
│   └── __init__.py
└── py.typed                 # PEP 561 type hints marker
```

Key files by task:

| Task | File(s) |
|------|---------|
| Add public API | `__init__.py` (`__all__` list) |
| Fix query bugs | `query.py`, `query_templates.py` |
| Add query templates | `query_templates.py` |
| Spatial queries | `spatial.py` |
| Bulk fetch logic | `fetch.py` |
| Chunked queries | `chunked.py` |
| Type conversion | `type_conversion.py` |
| Schema system | `schema_system.py`, `schemas/*.py` |
| Input validation | `sanitization.py` |
| Response export | `response.py` |
| Survey metadata | `metadata.py` |
| Exceptions | `exceptions.py`, `ldm/exceptions.py`, `awdb/exceptions.py`, `henry/exceptions.py`, `backends/exceptions.py` |
| Async client | `client.py`, `base_client.py` |
| LDM workflows | `ldm/*.py` |
| AWDB workflows | `awdb/*.py` |
| Henry workflows | `henry/*.py` |
| SSURGO data access | `ssurgo_client.py`, `ssurgo_tables.py` |
| Backend support | `backends/*.py` |
| Web Soil Survey | `wss.py` |
| Utilities | `utils.py` |

### Tests

```
tests/
├── conftest.py                        # pytest fixtures and configuration
├── test_query.py                      # Query builder tests
├── test_query_templates.py            # Query template interface tests
├── test_fetch.py                      # Bulk fetch and QueryPresets tests
├── test_chunked.py                    # Paginated query execution tests
├── test_spatial.py                    # Spatial query interface tests
├── test_spatial_and_responses.py      # Spatial queries and response exports
├── test_spatial_integration.py        # Spatial integration tests (marked @pytest.mark.integration)
├── test_response.py                   # SDAResponse export tests
├── test_response_concat.py            # Response concatenation tests
├── test_public_api.py                 # Public API exports (__all__ list)
├── test_client.py                     # SDAClient tests
├── test_type_conversion.py            # Type mapping tests
├── test_metadata.py                   # Survey metadata parsing tests
├── test_sanitization.py               # Input validation tests
├── test_sync.py                       # Sync wrapper decorator tests
├── test_integration.py                # Integration tests (marked @pytest.mark.integration)
├── test_ssurgo_client.py              # SSURGO client tests
├── test_ssurgo_tables.py              # SSURGO table schema tests
├── test_wss.py                        # Web Soil Survey download tests
├── test_ldm_imports.py                # LDM module import tests
├── test_ldm_client.py                 # LDMClient tests
├── test_ldm_exceptions.py             # LDM exception handling
├── test_ldm_query_builder.py          # LDM SQL query builder tests
├── test_ldm_tables.py                 # LDM table schema tests
├── test_ldm_backend_execution.py      # LDM backend execution tests
├── test_lab_pedon_lookup.py           # Lab pedon lookup tests
├── test_awdb.py                       # AWDB client tests
├── test_henry.py                      # Henry climate database tests
├── test_high_level.py                 # High-level workflow tests (marked @pytest.mark.integration)
├── test_backends_infrastructure.py    # Multi-backend infrastructure tests
├── test_backends_sda_sqlite.py        # SDA/SQLite backend tests
└── test_geopackage_backend.py         # GeoPackage backend tests
```

Run via:
```bash
pytest tests/<file>.py -v              # Single test file
pytest -m "not integration" -v         # Skip network-dependent tests
pytest tests/ -v --cov=soildb          # All tests with coverage
```

### Documentation

- `README.md` — API overview and quick examples
- `CONTRIBUTING.md` — Setup, PR guidelines, code conventions
- `AGENTS.md` — This file: agent navigation guide
- `docs/` — Quarto source documentation (build with `make docs`)
  - `docs/examples/` — Runnable code samples (client lifecycle, spatial, bulk fetch, LDM, AWDB, Henry, etc.)
  - `docs/*.qmd` — Quarto markdown files (validated via `make docs-validate`)
- `scripts/` — Utility scripts
  - `validate_examples.py` — Import validation for example scripts
  - `awdb_health_check.py` — AWDB service health monitoring
- `pyproject.toml` — Dependencies, build config, test config, project metadata
- `.github/` — GitHub workflows (if present)

## Workflows

### Data Hierarchy

USDA soil data is hierarchical:

- **SSURGO**: Survey area (legend) → Map unit (mukey) → Component (cokey) → Horizon (chorizonkey)
- **Lab Data (KSSL)**: Pedon (pedon_key) → Horizon → Physical/Chemical properties
- **Monitoring (AWDB/Henry)**: Station (site_id) → Sensor (variable) → Time series by depth

### API Tiers (use higher tier for simplicity)

1. **High-level** (`high_level.py`): Nested dataclasses with pre-fetched relationships
   - Examples: `fetch_ssurgo_mapunit_by_point()`, `fetch_labpedon_by_bbox()`
   - Returns complex nested structures with relationships resolved

2. **Mid-level** (`fetch.py`, `convenience.py`, subsystem `**/convenience.py`): `SDAResponse` (exports to DataFrame/dict/GeoDataFrame)
   - Examples: `fetch_by_keys()`, `get_mapunit_by_areasymbol()`, `get_sacatalog()`
   - Returns `SDAResponse` with flexible export options
   - Subsystems: `awdb.convenience`, `henry.convenience` (similar pattern)

3. **Low-level** (`query.py`, `query_templates.py`, `spatial.py`): Manual SQL + fluent Query builder
   - Use when mid-level functions don't fit
   - Includes pre-built templates for common queries

### Schema & Metadata System

- **Schemas** (`schemas/`): Table definitions with type metadata
  - Pedon, component, horizon, map unit, property schemas
  - Use for introspection, validation, and type mapping
- **Metadata** (`metadata.py`): Survey-level metadata parsing and filtering
  - Parse SSURGO survey metadata, search by keywords, filter by bbox
  - Use `SurveyMetadata` to explore available data sources

### Multi-Backend Support

LDM queries can run against multiple backends via `LDMClient`:
- **SDA backend**: NRCS web service (default, requires network)
- **SQLite backend**: Local database file (portable, offline)
- **GeoPackage backend**: Vector data in standardized format

Backend selection is automatic when a database path is provided.

### Async/Sync Pattern

All public functions are async. Sync access via `.sync()` method (auto-manages event loop):

```python
# Async
result = await get_mapunit_by_areasymbol("IA109")

# Sync (for scripts, interactive use)
result = get_mapunit_by_areasymbol.sync("IA109")
```

Sync wrapper created by `@add_sync_version` decorator.

### Exception Handling

Catch subsystem-specific exceptions rooted at `SoilDBError`:

- **SDA**: `SDANetworkError` (connection, timeout, maintenance), `SDAQueryError`, `SDAResponseError`
- **LDM**: `LDMError` → backend, query, parameter, table, response errors
- **AWDB**: `AWDBError` → connection and query errors
- **Backends**: `BackendError` → connection, query, schema errors
- **WSS**: `WSSDownloadError`

### Testing Pattern

Unit tests mock HTTP responses via `pytest-httpx` (no real SDA calls). Integration tests marked with `@pytest.mark.integration` (network-dependent, skipped by default).

Example test:
```python
@pytest.mark.asyncio
async def test_something(client, httpx_mock):
    httpx_mock.add_response(...)
    result = await client.execute(...)
    assert result...
```

## Infrastructure

### Configuration & Clients

**SDAClient** lifecycle (all APIs use this pattern):
```python
async with SDAClient(config=ClientConfig(timeout=120.0, retries=5)) as client:
    result = await client.execute_sql(sql)
```

**LDMClient** (Lab Data Model) supports multiple backends:
```python
# Auto-detect from database path
async with LDMClient(database_path="/path/to/data.db") as client:
    result = await client.query_by_location(...)
```

**Multi-backend support**: Backends auto-selected via `backend_name` or detected from database path.

**SDA maintenance window**: ~12:45–1:00 AM US Central Time. Use `ClientConfig.reliable()` (120s timeout, 5 retries) for transient timeouts.

### Subsystem Clients

Each subsystem (AWDB, Henry, LDM) has its own client with independent configuration:
- **AWDBClient** — SCAN/SNOTEL stations (USDA water/snow monitoring)
- **HenryClient** — Mount Soil climate data
- **LDMClient** — Lab Data Model (KSSL pedon data)

All follow the same async context manager pattern as SDAClient.

### Code Style

- **Type hints**: Full PEP 484 (target Python ≥3.9, run mypy)
- **Docstrings**: Google style (`Args:` / `Returns:` / `Raises:` / `Example:`)
- **Line length**: 88 characters (ruff)
- **Linting**: `ruff check` + `mypy` (run via `make lint-fix`)
- **Formatting**: `ruff format` (run via `make format`)
- **Imports**: Organize per black/isort standards (enforced by ruff)
- **Pre-commit**: Use `make pre-commit-install` to set up hooks

### Dependencies

**Core runtime**:
- `httpx>=0.24.0` — async HTTP client
- `aiosqlite>=0.19.0` — async SQLite driver

**Development**:
- `pytest>=7.0`, `pytest-asyncio`, `pytest-httpx`, `pytest-cov`, `pytest-timeout`
- `ruff>=0.1.0` — linting and formatting
- `mypy>=1.0.0` — static type checking
- `pre-commit>=3.0.0` — git hooks

**Documentation**:
- `quartodoc>=0.9.0` — API documentation extraction
- `quarto>=0.1.0` — document renderer

**Security**:
- `bandit>=1.7.0` — security issue scanner
- `safety>=2.3.0` — dependency vulnerability checker

**Optional features**:
- **DataFrames**: `pandas>=1.5.0`, `polars>=0.18.0`
- **Spatial**: `geopandas>=0.13.0`, `shapely>=2.0.0`
- **Jupyter**: `jupyter>=1.0.0`, `ipython>=8.0.0`, `nest-asyncio>=1.5.0`
- **Soil profiles**: `soilprofilecollection>=0.2.0`

See `pyproject.toml` for full list and exact version specs.

## References

- [[Code Conventions & Patterns]](CONTRIBUTING.md) — Setup, PR guidelines, test patterns
- [[User Guide & Examples]](README.md) — API overview and quick-start code
- [[Runnable Examples]](docs/examples/) — Client lifecycle, spatial, bulk fetch, LDM, AWDB
- [[Full Documentation]](docs/) — Build with `make docs`, serve with `make docs-serve`
