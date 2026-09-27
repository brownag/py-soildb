# py-soildb Examples

Runnable examples for querying USDA Soil Data Access (SDA) and AWDB soil monitoring stations.

## Core Examples

Start here for common patterns.

| File | Purpose | Use Case |
|------|---------|----------|
| **01_basic.py** | Core SDA queries and DataFrame export | Connect and run queries |
| **02_spatial.py** | Location queries (point, bbox, polygon) | Spatial filtering |
| **05_awdb.py** | AWDB station retrieval (SCAN, SNOTEL) | Monitoring station data |
| **08_fetch.py** | Bulk data fetching with `fetch_by_keys()` | Paginated bulk queries |
| **07_query_templates.py** | Pre-built SQL queries in `query_templates` | Reusable SQL templates |

---

## Feature & Domain Examples

| File | Purpose | Feature |
|------|---------|---------|
| **03_metadata.py** | Survey metadata parsing | Metadata extraction |
| **04_schema.py** | Schema inspection and column types | Schema discovery |
| **06_awdb_availability.py** | Station data availability checks | Availability audits |
| **09_wss_download.py** | Web Soil Survey file downloads and extraction | WSS downloads |

---

## Specialized Examples

### SoilProfileCollection

Convert horizon and layer data to `SoilProfileCollection` objects (requires `soildb[soil]`):

- **01_basic_conversion.py**: Convert horizon data with standard SSURGO columns (`cokey`, `chkey`, `hzdept_r`, `hzdepb_r`).
- **02_with_site_metadata.py**: Merge component site attributes with horizon data.
- **03_lab_pedon_workflow.py**: Convert lab pedon data using direct column parameter mappings (`site_id_col`, `hz_id_col`, `hz_top_col`, `hz_bot_col`).
- **04_custom_columns.py**: Custom column mappings for non-standard or aliased column names.

See [SoilProfileCollection](soilprofilecollection/) for runnable scripts.

### Jupyter Notebooks

- **notebooks/01_metadata_discovery.ipynb**: Interactive survey area discovery, filtering by keywords and bounding boxes.

---

## Running Examples

Install development dependencies from the repository root:

```bash
pip install -e ".[dev]"
```

Run an individual script:

```bash
python docs/examples/01_basic.py
```

Run all standalone examples:

```bash
python docs/examples/0*.py
```

---

## API Patterns Reference

### Pattern 1: Synchronous Script

Use `.sync()` for scripts and notebooks:

```python
from soildb import get_mapunit_by_areasymbol

response = get_mapunit_by_areasymbol.sync("IA109")
df = response.to_pandas()
```

### Pattern 2: Async Client Context

Use `async with SDAClient()` for services and concurrent requests:

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

### Pattern 3: Query Templates

Pre-built SQL templates for standard queries:

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

### Pattern 4: Custom SQL

Build custom queries with `Query`:

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

`SDAResponse` exports to several formats:

```python
df = response.to_pandas()                    # pandas DataFrame
df = response.to_polars()                    # Polars DataFrame
data = response.to_dict()                    # List of dicts
spc = response.to_soilprofilecollection()   # SoilProfileCollection
gdf = response.to_geopandas()               # GeoDataFrame (with WKT)
```

---

## Common Tasks

- **Location queries**: See `01_basic.py` and `02_spatial.py`.
- **Bulk data fetching**: See `08_fetch.py` and `07_query_templates.py`.
- **AWDB monitoring stations**: See `05_awdb.py` and `06_awdb_availability.py`.
- **SoilProfileCollection conversion**: See `soilprofilecollection/` examples.
- **Metadata parsing**: See `03_metadata.py`.

---

## Reference

- [Workflows](../workflows.qmd): Task-based walkthroughs
- [Async Guide](../async.qmd): Concurrency patterns
- [AWDB Integration](../awdb.qmd): Soil monitoring stations
- [API Reference](../api.qmd): Public API documentation
- [Error Handling](../error-handling.qmd): Exception types
- [Troubleshooting](../troubleshooting.qmd): Common errors
