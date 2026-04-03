# py-soildb Examples

Practical, runnable examples for querying USDA-NRCS Soil Data Access (SDA) and AWDB soil monitoring data.

**Location in docs**: These examples are referenced throughout the [workflows](../workflows.qmd), [async](../async.qmd), [AWDB](../awdb.qmd), and [API](../api.qmd) documentation. See the corresponding docs file for context.

---

## Core Examples (Primary Patterns)

These examples demonstrate the recommended approaches. Start here.

| File | Purpose | Use Case |
|------|---------|----------|
| **01_basic.py** | ✅ Core SDA functionality: client setup, queries, DataFrame export | How to connect and run basic queries |
| **02_spatial.py** | ✅ Geographic queries (point, bbox, polygon) with optional GeoPandas | Queries by location |
| **05_awdb.py** | ✅ AWDB station data retrieval (SCAN, SNOTEL networks) | Finding and fetching monitoring station data |
| **08_fetch.py** | ✅ Bulk data retrieval with automatic pagination using `fetch_by_keys()` | Fetching data for multiple keys |
| **07_querybuilder.py** | Query template patterns using `query_templates` module | Pre-built SQL queries for common patterns |

---

## Feature & Domain Examples

Demonstrate specific features or domain workflows.

| File | Purpose | Feature |
|------|---------|---------|
| **03_metadata.py** | Survey metadata parsing from SDA responses | Metadata extraction |
| **04_schema.py** | Schema inspection and working with column type information | Type system & schema discovery |
| **06_awdb_availability.py** | Data availability assessment across monitoring stations | AWDB analysis workflows |
| **09_wss_download.py** | Web Soil Survey (WSS) file downloads with extraction | WSS integration |

---

## Specialized Examples

### SoilProfileCollection

Converting horizon/layer data to `SoilProfileCollection` format for specialized analysis.

- **01_basic_conversion.py** — Basic horizon to SoilProfileCollection with default preset
- **02_with_site_metadata.py** — Include component metadata in site slot
- **03_lab_pedon_workflow.py** — Different preset configurations for lab pedon data
- **04_custom_columns.py** — Custom column mapping for non-standard data

See [SoilProfileCollection docs](../api.qmd) for preset options.

### Jupyter Notebooks

- **notebooks/01_metadata_discovery.ipynb** — Interactive survey area discovery, filtering by keywords/bbox

See [Using Jupyter Notebooks](#using-jupyter-notebooks) below.

---

## Running Examples

### Prerequisites

```bash
# Install dependencies (from root py-soildb directory)
pip install -e ".[dev]"
```

### Run Individual Examples

```bash
cd /path/to/docs/examples
python 01_basic.py
python 02_spatial.py
python 08_fetch.py
```

### Run All Examples

```bash
python 0*.py
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_metadata_discovery.ipynb
```

---

## API Patterns Reference

See the [workflows](../workflows.qmd) doc for detailed explanations. This is a quick reference.

### Pattern 1: Simple Synchronous (Scripts & Jupyter)

Use sync wrappers (`.sync()` suffix) for simplicity:

```python
from soildb import get_mapunit_by_areasymbol

response = get_mapunit_by_areasymbol("IA109")  # Auto-created sync wrapper
df = response.to_pandas()
```

### Pattern 2: Async with Context Manager

For production applications:

```python
import asyncio
from soildb import SDAClient, query_templates

async def main():
    async with SDAClient() as client:
        query = query_templates.query_mapunits_by_legend("IA109")
        response = await client.execute(query)
        return response.to_pandas()

df = asyncio.run(main())
```

### Pattern 3: Using Query Templates

Pre-built queries for common patterns:

```python
from soildb import query_templates, fetch_by_keys

# Get map units by survey area
query = query_templates.query_mapunits_by_legend("IA109")

# Get components for those map units
response = await fetch_by_keys(mukeys, "component", key_column="mukey")
```

### Pattern 4: Custom SQL

For complex queries, build your own using `Query()`:

```python
from soildb import Query

query = (Query()
    .select("mukey", "muname", "musym")
    .from_("mapunit")
    .where("areasymbol = 'IA109'")
    .order_by("mukey")
    .limit(100))
```

---

## Data Export Formats

All examples return `SDAResponse` which supports multiple export formats:

```python
df = response.to_pandas()                    # pandas DataFrame
df = response.to_polars()                    # Polars DataFrame  
data = response.to_dict()                    # List of dicts
spc = response.to_soilprofilecollection()   # SoilProfileCollection
gdf = response.to_geodataframe()            # GeoDataFrame (with WKT)
```

---

## Common Tasks

**Finding data by location**: Start with `01_basic.py`, then `02_spatial.py`

**Fetching bulk data**: See `08_fetch.py` and `07_querybuilder.py`

**AWDB monitoring stations**: See `05_awdb.py` and `06_awdb_availability.py`

**Converting to SoilProfileCollection**: See `soilprofilecollection/` examples

**Working with metadata**: See `03_metadata.py`

---

## Reference

- [Workflows Documentation](../workflows.qmd) — Common tasks with explanations
- [Async Guide](../async.qmd) — Advanced async patterns
- [AWDB Integration](../awdb.qmd) — Soil monitoring data
- [API Reference](../api.qmd) — Full public API
- [Error Handling](../error-handling.qmd) — Exception types and handling
- [Troubleshooting](../troubleshooting.qmd) — Common issues
