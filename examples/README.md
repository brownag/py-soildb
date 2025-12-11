# py-soildb Examples

Practical examples for querying USDA-NRCS Soil Data Access (SDA) and AWDB soil monitoring data.

## Examples

| File | Purpose |
|------|---------|
| **01_basic.py** | Core SDA functionality - connection, queries, DataFrame export |
| **02_spatial.py** | Geographic queries (point, bbox, polygon) with GeoPandas integration |
| **03_metadata.py** | Survey metadata parsing from SDA responses |
| **04_schema.py** | Automatic T-SQL type inference and schema generation |
| **05_awdb.py** | AWDB station data retrieval (SCAN, SNOTEL networks) |
| **06_awdb_availability.py** | Data availability assessment across monitoring stations |
| **07_querybuilder.py** | SQL query builder for complex custom queries |
| **08_fetch.py** | Bulk data retrieval with automatic pagination and chunking |

Also included: [SoilProfileCollection](soilprofilecollection/) examples and [Jupyter notebooks](notebooks/) for exploratory analysis.

## Quick Start

```bash
# Install dependencies (from root py-soildb directory)
pip install -e ".[dev]"

# Run an example
python 01_basic.py

# Run all examples
python 0*.py
```

## API Patterns

### Synchronous (Recommended for Scripts)

```python
from soildb import get_mapunit_by_areasymbol

response = get_mapunit_by_areasymbol("IA109")  # Auto-created client
df = response.to_pandas()
```

### Asynchronous

```python
import asyncio
from soildb import get_mapunit_by_areasymbol

async def main():
    response = await get_mapunit_by_areasymbol("IA109")
    return response.to_pandas()

df = asyncio.run(main())
```

### Explicit Client Management

```python
from soildb import SDAClient, Query

async with SDAClient() as client:
    query = Query().select("*").from_("mapunit").where("areasymbol = 'IA109'")
    result = await client.execute(query)
```

## Export Formats

```python
df = response.to_pandas()              # pandas DataFrame
df = response.to_polars()              # Polars DataFrame
data = response.to_dict()              # List of dicts
spc = response.to_soilprofilecollection()  # SoilProfileCollection
gdf = response.to_geodataframe()       # GeoDataFrame (with WKT)
```

## Common Tasks

**Query by location**: `01_basic.py`, `02_spatial.py`

**Bulk data**: `08_fetch.py`

**Custom SQL**: `07_querybuilder.py`

**AWDB monitoring**: `05_awdb.py`, `06_awdb_availability.py`

## Requirements

- Python 3.9+
- py-soildb (installed with examples)
- Optional: `jupyter` for notebooks, `geopandas` for spatial examples

## See Also

- [API Documentation](../docs/api.qmd)
- [Async Usage](../docs/async.qmd)
- [AWDB Integration](../docs/awdb.qmd)
- [Troubleshooting](../docs/troubleshooting.qmd)
