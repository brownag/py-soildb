# AGENTS.md

Navigation map and infrastructure guide for py-soildb agents.

**Project**: Async Python client for USDA soil data services (SDA, LDM, AWDB, Henry, WSS)  
**Status**: Alpha (v0.x) — lower-level APIs (Query, response, spatial) change less frequently than high-level convenience functions  
**Language**: Python ≥3.9 | **Build**: hatchling | **Test**: pytest + pytest-asyncio + pytest-httpx

## Quick Start

Essential commands (extract from `Makefile`):

```bash
make install              # Install with dev extras
make test                 # Run all unit tests (pytest)
make lint-fix             # Auto-fix linting (ruff + mypy)
make docs                 # Build Quarto docs
make security             # Run security checks (bandit/safety)
```

Common pytest patterns:
```bash
pytest tests/test_query.py -v                # Single test file
pytest -m "not integration" -v               # Skip network-dependent tests
pytest tests/test_query.py::test_name -v    # Single test
```

Setup: See `CONTRIBUTING.md` for detailed environment setup.

## Find Things

### Code Structure

```
src/soildb/
├── __init__.py              # Public API re-exports
├── client.py                # SDAClient (async HTTP to NRCS web service)
├── query.py                 # Query builder (fluent interface for SQL)
├── response.py              # SDAResponse (DataFrame/dict/GeoDataFrame export)
├── spatial.py               # Spatial filtering (point/bbox queries)
├── fetch.py                 # Bulk key-based queries with pagination
├── convenience.py           # Single/simple queries
├── high_level.py            # Complex workflows returning nested dataclasses
├── type_conversion.py       # Type mapping (SQL → Python)
├── exceptions.py            # SoilDBError hierarchy
├── ldm/                     # Lab Data Model (KSSL pedon data)
├── awdb/                    # AWDB/SCAN/SNOTEL monitoring data
├── henry/                   # Henry climate database
├── backends/                # Multi-database backends (SDA, LDM, SQLite, PostGIS stubs)
└── schemas/                 # Table schemas with type metadata
```

Key files by task:

| Task | File(s) |
|------|---------|
| Add public API | `__init__.py` (`__all__` list) |
| Fix query bugs | `query.py`, `query_templates.py` |
| Spatial queries | `spatial.py` |
| Bulk fetch logic | `fetch.py` |
| Type conversion | `type_conversion.py` |
| Response export | `response.py` |
| Exceptions | `exceptions.py` |
| Async client | `client.py`, `base_client.py` |
| LDM workflows | `ldm/*.py` |
| AWDB workflows | `awdb/*.py` |
| Henry workflows | `henry/*.py` |

### Tests

```
tests/
├── test_query.py            # Query builder tests
├── test_fetch.py            # Bulk fetch tests
├── test_spatial.py          # Spatial query tests
├── test_response.py         # Response export tests
├── test_public_api.py       # Public API exports
├── test_ldm*.py             # LDM subsystem tests
├── test_awdb*.py            # AWDB subsystem tests
└── test_backends*.py        # Multi-backend infrastructure tests
```

Run via `pytest tests/<file>.py -v` or `pytest -m "not integration" -v` (skip network tests).

### Documentation

- `README.md` — API overview and quick examples
- `CONTRIBUTING.md` — Setup, PR guidelines, code conventions
- `docs/examples/` — Runnable code samples (client lifecycle, spatial, bulk fetch, etc.)
- `docs/` — Quarto source (build with `make docs`)
- `pyproject.toml` — Dependencies, build config, test config

## Workflows

### Data Hierarchy

USDA soil data is hierarchical:

- **SSURGO**: Survey area (legend) → Map unit (mukey) → Component (cokey) → Horizon (chorizonkey)
- **Lab Data (KSSL)**: Pedon (pedon_key) → Horizon → Physical/Chemical properties
- **Monitoring (AWDB/Henry)**: Station (site_id) → Sensor (variable) → Time series by depth

### API Tiers (use higher tier for simplicity)

1. **High-level** (`high_level.py`): Nested dataclasses with pre-fetched relationships
   - Examples: `fetch_ssurgo_mapunit_by_point()`, `fetch_labpedon_by_bbox()`

2. **Mid-level** (`fetch.py`, `convenience.py`): `SDAResponse` (exports to DataFrame/dict/GeoDataFrame)
   - Examples: `fetch_by_keys()`, `get_mapunit_by_areasymbol()`, `get_sacatalog()`

3. **Low-level** (`query.py`, `spatial.py`): Manual SQL + fluent Query builder
   - Use when mid-level functions don't fit

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

**SDA maintenance window**: ~12:45–1:00 AM US Central Time. Use `ClientConfig.reliable()` (120s timeout, 5 retries) for transient timeouts.

### Code Style

- **Type hints**: Full PEP 484 (target Python ≥3.9, run mypy)
- **Docstrings**: NumPy-style with Examples section
- **Line length**: 88 characters (ruff)
- **Linting**: `ruff check` + `mypy` (run via `make lint-fix`)
- **Imports**: Organize per black/isort standards

### Dependencies

**Core runtime**: httpx (async HTTP), aiosqlite (async SQLite)

**Dev**: pytest, pytest-asyncio, pytest-httpx, ruff, mypy, bandit, safety

**Optional extras**: pandas/polars, geopandas, soilprofilecollection, jupyter

See `pyproject.toml` for full list and version specs.

## References

- [[Code Conventions & Patterns]](CONTRIBUTING.md) — Setup, PR guidelines, test patterns
- [[User Guide & Examples]](README.md) — API overview and quick-start code
- [[Runnable Examples]](docs/examples/) — Client lifecycle, spatial, bulk fetch, LDM, AWDB
- [[Full Documentation]](docs/) — Build with `make docs`, serve with `make docs-serve`
