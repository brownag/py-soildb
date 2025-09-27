# AI Coding Agent Instructions for py-soildb

## Project Overview
py-soildb is an async Python client for the USDA-NRCS Soil Data Access (SDA) web service with integrated AWDB soil monitoring data. It provides SQL query building, spatial queries, bulk data fetching, and multi-format export (pandas/polars/SoilProfileCollection).

## Architecture

### Core Components
- **SDAClient**: Async HTTP client (httpx-based) with retry logic and error handling
- **Query**: Fluent SQL builder with parameterization and sanitization
- **SDAResponse**: Unified response handling with multi-format export (DataFrame, dict, WKT)
- **Spatial Queries**: Point, bbox, and polygon queries with flexible geometry input
- **Bulk Operations**: Paginated fetching with automatic chunking for large datasets
- **AWDB Integration**: Soil monitoring station data (SCAN, SNOTEL networks)
- **Type Conversion**: Automatic type inference and data validation

### Data Hierarchy
SSURGO: Legend (survey area) -> Mapunit -> Component -> Horizon (chorizon)
Lab Data: Pedon Site -> Pedon Horizon

### Module Organization
- **client.py**: HTTP client with connectivity and retry strategies
- **query.py**: SQL builder with ColumnSets for standard column groups
- **response.py**: Response parsing, validation, and multi-format export
- **spatial.py**: Spatial query builder for point/bbox/polygon queries
- **fetch.py**: Bulk data retrieval with automatic pagination and chunking
- **convenience.py**: High-level helpers (get_mapunit_*, get_sacatalog, etc.)
- **awdb_integration.py**: Soil water availability workflows combining SDA + AWDB
- **awdb/**: AWDB client and convenience functions for station data
- **exceptions.py**: Structured error hierarchy (SDANetworkError, SDAQueryError, etc.)
- **metadata.py**: Survey metadata parsing from fgdcmetadata column
- **spc_presets.py**: Configuration for SoilProfileCollection conversion
- **sync.py**: Synchronous wrappers for async functions

## Developer Workflows

### Setup & Installation
```bash
# Setup development environment
make install           # Install with all dev/test/docs dependencies
make install-prod      # Production install only
pip install -e ".[dev]"  # Manual editable install
```

### Development Commands
```bash
make test             # Run pytest with asyncio
make test-cov         # Coverage report
make lint             # Run ruff + mypy checks
make lint-fix         # Auto-fix linting issues
make format           # Format code with ruff
make docs             # Build Quarto documentation
make build            # Build distribution packages
make security         # Run security checks (bandit, safety)
make pre-commit-run   # Run pre-commit hooks on all files
```

### Key Tools
- **Testing**: pytest with asyncio support (pytest-asyncio, pytest-httpx for mocking)
- **Linting**: ruff (fast Python linter) + mypy (type checking)
- **Documentation**: Quarto + quartodoc (API docs from docstrings)
- **Build**: hatchling (modern Python build backend)
- **Package**: httpx (async HTTP client), pydantic (data validation)

## API Conventions

### Async-First Design
- All public APIs are async (async def / await)
- Use asyncio.run() or nest_asyncio in Jupyter notebooks
- Context manager pattern: `async with SDAClient() as client:`
- No blocking operations in async functions

### Error Handling
```python
from soildb import SoilDBError, SDANetworkError, SDAQueryError

try:
    result = await client.execute(query)
except SDANetworkError:  # Connection/timeout/maintenance
    pass
except SDAQueryError:    # Invalid query/response format
    pass
except SoilDBError:      # Any soildb error
    pass
```

### Query Building
```python
# Fluent API with method chaining
query = (Query()
    .select("mukey", "muname", "musym")
    .from_("mapunit")
    .where("areasymbol = 'IA109'")
    .order_by("mukey")
    .limit(100))

result = await client.execute(query)
```

### Column Sets
Use ColumnSets for standard column groups:
- `ColumnSets.MAPUNIT_BASIC` / `MAPUNIT_DETAILED` / `MAPUNIT_SPATIAL`
- `ColumnSets.COMPONENT_BASIC` / `COMPONENT_DETAILED`
- `ColumnSets.CHORIZON_BASIC` / `CHORIZON_TEXTURE` / `CHORIZON_CHEMICAL` / `CHORIZON_PHYSICAL`
- `ColumnSets.LEGEND_BASIC` / `LEGEND_DETAILED`

### Spatial Queries
```python
# Primary function: spatial_query()
from soildb import spatial_query

# Point query
response = await spatial_query("POINT(-93.6 42.0)", table="mupolygon")

# Bounding box (dict or WKT)
response = await spatial_query(
    {"xmin": -94.0, "ymin": 41.0, "xmax": -93.0, "ymax": 42.0},
    table="mupolygon"
)

# Polygon (WKT string)
wkt_polygon = "POLYGON((-94 41, -93 41, -93 42, -94 42, -94 41))"
response = await spatial_query(wkt_polygon, table="sapolygon", return_type="spatial")
```

### Bulk Operations
```python
from soildb import fetch_by_keys, get_mukey_by_areasymbol

# Discover keys first (small result set)
mukeys = await get_mukey_by_areasymbol("IA109")

# Bulk fetch with automatic chunking (handles pagination)
components = await fetch_by_keys(
    mukeys,
    "component",
    key_column="mukey",
    chunk_size=100,  # Default: 100
    client=client
)
```

### Data Export
```python
# SDAResponse supports multiple formats
response = await client.execute(query)

df = response.to_pandas()        # DataFrame
df = response.to_polars()        # Polars DataFrame
data = response.to_dict()        # List of dicts
spc = response.to_soilprofilecollection()  # SoilProfileCollection
gdf = response.to_geodataframe() # GeoDataFrame (with WKT geometries)
```

## Common Patterns

### Basic Point Query
```python
from soildb import SDAClient, spatial_query

async def query_point(lon, lat):
    async with SDAClient() as client:
        response = await spatial_query(
            f"POINT({lon} {lat})",
            table="mupolygon",
            client=client
        )
        return response.to_pandas()
```

### Survey Area Query
```python
from soildb import SDAClient, get_mapunit_by_areasymbol

async def get_survey_data(areasymbol):
    async with SDAClient() as client:
        response = await get_mapunit_by_areasymbol(areasymbol, client=client)
        return response.to_pandas()
```

### Multi-Level Data Fetch
```python
from soildb import get_mukey_by_areasymbol, fetch_by_keys

async def fetch_components(areasymbol):
    # Step 1: Get mukeys for survey area
    mukeys = await get_mukey_by_areasymbol(areasymbol)
    
    # Step 2: Fetch all components for those mukeys
    components = await fetch_by_keys(mukeys, "component", key_column="mukey")
    return components.to_pandas()
```

### Soil Water Availability
```python
from soildb import get_component_water_properties
from soildb.awdb import find_stations_by_criteria

async def water_analysis(areasymbol):
    # SDA soil properties
    soil = await get_component_water_properties(areasymbol)
    soil_df = soil.to_pandas()
    
    # AWDB monitoring stations
    stations = await find_stations_by_criteria(
        network_codes=["SCAN"],
        state_codes=["IA"],
        elements=["SMS:*"]  # Soil moisture sensors
    )
```

## Important Notes

### Synchronous Wrappers
Async functions have auto-generated sync wrappers (add "_sync" suffix):
```python
# Instead of: asyncio.run(get_mapunit_by_areasymbol("IA109"))
response = get_mapunit_by_areasymbol_sync("IA109")  # Automatic wrapper
```

### Client Management
```python
# Always use context manager for proper cleanup
async with SDAClient() as client:
    result = await client.execute(query)

# Or manual management
client = SDAClient()
try:
    result = await client.execute(query)
finally:
    await client.close()
```

### Type Annotations
Use type hints throughout:
```python
from typing import Optional, List
from soildb import SDAClient, SDAResponse

async def fetch_data(
    keys: List[int],
    client: Optional[SDAClient] = None
) -> SDAResponse:
    ...
```

### Testing
```bash
# Run specific test file
pytest tests/test_query.py -v

# Run with coverage
pytest --cov=soildb tests/

# Skip integration tests (require network)
pytest -m "not integration"

# Run with timeout
pytest --timeout=30
```

### Documentation Standards
- Docstrings: NumPy format with Examples section
- Type hints: PEP 484 (optional for Python < 3.10)
- Module docstrings: Brief description + architecture notes
- Error handling: Document expected exceptions

### Common Import Patterns
```python
from soildb import (
    SDAClient,
    Query,
    SDAResponse,
    spatial_query,
    fetch_by_keys,
    get_mapunit_by_areasymbol,
    get_mapunit_by_point,
    get_mapunit_by_bbox,
    SoilDBError,
    SDANetworkError,
    SDAQueryError,
)

from soildb.awdb import (
    AWDBClient,
    find_stations_by_criteria,
    get_monitoring_station_data,
)

from soildb.metadata import SurveyMetadata
```

## Performance Tips

### Query Optimization
- Use specific columns (avoid SELECT *)
- Filter in WHERE clause (not in Python)
- Use spatial queries for geographic filtering (faster than large WHERE clauses)
- Break complex joins into separate queries

### Bulk Operations
- Use fetch_by_keys() for automatic pagination
- Set chunk_size based on API limits (default 100)
- Use asyncio.gather() for concurrent requests
- Cache survey metadata when doing repeated lookups

### Memory Management
- Process large results in chunks
- Use generators for streaming data
- Delete large objects after use
- Monitor memory with large spatial queries

## Troubleshooting

### Connection Issues
Check SDA maintenance window (12:45 AM - 1:00 AM Central Time)
```python
except SDAMaintenanceError:
    # Service under maintenance, retry later
    pass
```

### Timeout Issues
Increase timeout for complex queries:
```python
from soildb import ClientConfig
config = ClientConfig(timeout=120.0)
async with SDAClient(config=config) as client:
    ...
```

### No Results
Verify:
- Correct areasymbol (use get_sacatalog() to list valid areas)
- Coordinate system (SDA uses WGS84 lon/lat)
- WHERE conditions are not too restrictive
- Data actually exists for query criteria

### Type Conversion Issues
Check column types in schema_system.py TypeMap
Override with custom type_map parameter if needed
</content>
<parameter name="filePath">/home/andrew/workspace/soilmcp/upstream/py-soildb/.github/copilot-instructions.md