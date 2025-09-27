# soildb


[![PyPI
version](https://badge.fury.io/py/soildb.svg)](https://pypi.org/project/soildb/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Python client for the USDA-NRCS Soil Data Access (SDA) web service, NRCS
monitoring networks (SCAN, SNOTEL), and other National Cooperative Soil
Survey data sources.

## Overview

`soildb` provides Python access to:
- **Soil Data**: USDA Soil Data Access (SDA) web service for soil survey data
- **Weather Data**: NRCS Air and Water Database (AWDB) for soil and weather monitoring
- **Integration**: Tools for combining soil and weather data for comprehensive analysis

Query soil survey data, environmental monitoring data, export to pandas/polars
DataFrames, and handle spatial queries.

**Note**: AWDB module provides complementary environmental data (soil moisture,
temperature, precipitation). See the documentation in `docs/awdb.qmd` for guidance
on how to use AWDB with soil data.

## Installation

``` bash
pip install soildb
```

For spatial functionality:

``` bash
pip install soildb[spatial]
```

For all optional features support:

``` bash
pip install soildb[all]
```

## Features

**Soil Data (SDA)**
- Query soil survey data from NRCS Soil Data Access
- Export to pandas and polars DataFrames
- Build custom SQL queries with fluent interface
- Spatial queries with points, bounding boxes, and polygons
- Bulk data fetching with automatic pagination
- Full pedon laboratory characterization data

**Environmental Data (AWDB)**
- Access soil moisture and temperature monitoring from SCAN stations
- Retrieve precipitation, temperature, and weather data from SNOTEL and NWCC networks
- Find nearest monitoring stations by location
- Query historical weather patterns for climate analysis

**Integration Features**
- Combine soil properties with weather patterns for suitability analysis
- Correlate soil characteristics with environmental responses
- Validate soil survey data against field observations
- Async I/O for high performance and concurrency

## Quick Start

### Query Builder

Build and execute custom SQL queries with the fluent interface:

``` python
from soildb import Query

query = (Query()
        .select("mukey", "muname", "musym")
        .from_("mapunit")
        .inner_join("legend", "mapunit.lkey = legend.lkey")
        .where("areasymbol = 'IA109'")
        .limit(5))

# Inspect the generated SQL
print(query.to_sql())

# Execute and get results
from soildb import SDAClient
result = SDAClient().execute.sync(query)
df = result.to_pandas()
print(df.head())
```

    SELECT TOP 5 mukey, muname, musym FROM mapunit INNER JOIN legend ON mapunit.lkey = legend.lkey WHERE areasymbol = 'IA109'
        mukey                                             muname  musym
    0  408337  Colo silty clay loam, channeled, 0 to 2 percen...   1133
    1  408339        Colo silty clay loam, 0 to 2 percent slopes    133
    2  408340        Colo silty clay loam, 2 to 4 percent slopes   133B
    3  408345  Clarion loam, 9 to 14 percent slopes, moderate...  138D2
    4  408348          Harpster silt loam, 0 to 2 percent slopes   1595

## Async vs Synchronous Usage

All soildb functions have both async and synchronous versions. For most use cases, the synchronous `.sync()` version is simpler and easier to use.

### Synchronous Usage

For simple scripts and interactive use, soildb provides synchronous versions of all async functions:

``` python
from soildb import get_mapunit_by_areasymbol

# Synchronous usage - no async/await needed!
mapunits = get_mapunit_by_areasymbol.sync("IA109")
df = mapunits.to_pandas()
print(f"Found {len(df)} map units")
df.head()
```

    Found 80 map units

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        vertical-align: right;
    }
</style>

|  | mukey | musym | muname | mukind | muacres | areasymbol | areaname |
|----|----|----|----|----|----|----|----|
| 0 | 408333 | 1032 | Spicer silty clay loam, 0 to 2 percent slopes | Consociation | 1834 | IA109 | Kossuth County, Iowa |
| 1 | 408334 | 107 | Webster clay loam, 0 to 2 percent slopes | Consociation | 46882 | IA109 | Kossuth County, Iowa |
| 2 | 408335 | 108 | Wadena loam, 0 to 2 percent slopes | Consociation | 807 | IA109 | Kossuth County, Iowa |
| 3 | 408336 | 108B | Wadena loam, 2 to 6 percent slopes | Consociation | 1103 | IA109 | Kossuth County, Iowa |
| 4 | 408337 | 1133 | Colo silty clay loam, channeled, 0 to 2 percen... | Consociation | 1403 | IA109 | Kossuth County, Iowa |

</div>

The `.sync` methods automatically manage SDA client connections for you. For multiple calls, consider reusing a client:

``` python
from soildb import SDAClient, get_mapunit_by_areasymbol

client = SDAClient()
mapunits1 = get_mapunit_by_areasymbol.sync("IA109", client=client)
mapunits2 = get_mapunit_by_areasymbol.sync("IA113", client=client)
client.close()
```

### Convenience Functions

soildb provides high-level functions for common tasks:

``` python
from soildb import get_mapunit_by_areasymbol

mapunits = get_mapunit_by_areasymbol.sync("IA109")
df = mapunits.to_pandas()
print(f"Found {len(df)} map units")
df.head()
```

    Found 80 map units

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | mukey | musym | muname | mukind | muacres | areasymbol | areaname |
|----|----|----|----|----|----|----|----|
| 0 | 408333 | 1032 | Spicer silty clay loam, 0 to 2 percent slopes | Consociation | 1834 | IA109 | Kossuth County, Iowa |
| 1 | 408334 | 107 | Webster clay loam, 0 to 2 percent slopes | Consociation | 46882 | IA109 | Kossuth County, Iowa |
| 2 | 408335 | 108 | Wadena loam, 0 to 2 percent slopes | Consociation | 807 | IA109 | Kossuth County, Iowa |
| 3 | 408336 | 108B | Wadena loam, 2 to 6 percent slopes | Consociation | 1103 | IA109 | Kossuth County, Iowa |
| 4 | 408337 | 1133 | Colo silty clay loam, channeled, 0 to 2 percen... | Consociation | 1403 | IA109 | Kossuth County, Iowa |

</div>

If you have suggestions for new convenience functions please file a
[feature request on
GitHub](https://github.com/brownag/py-soildb/issues/new).

### Spatial Queries

Query soil data by location with points, bounding boxes, or polygons:

``` python
from soildb import spatial_query

# Point query
response = spatial_query.sync(
    geometry="POINT(-93.6 42.0)",
    table="mupolygon"
)
df = response.to_pandas()
print(f"Point query found {len(df)} results")
```

    Point query found 1 results

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | mukey | areasymbol | musym | nationalmusym | muname | mukind |
|----|----|----|----|----|----|----|
| 0 | 411278 | IA169 | 1314 | fsz1 | Hanlon-Spillville complex, channeled, 0 to 2 p... | Complex |

</div>

### Bulk Data Fetching

Retrieve large datasets efficiently with automatic pagination and chunking:

``` python
from soildb import fetch_by_keys, get_mukey_by_areasymbol

# Get mukeys for survey areas
areas = ["IA109", "IA113", "IA117"]
all_mukeys = get_mukey_by_areasymbol.sync(areas)

print(f"Found {len(all_mukeys)} mukeys across {len(areas)} areas")

# Fetch components in chunks automatically
response = fetch_by_keys.sync(
    all_mukeys, 
    "component", 
    key_column="mukey", 
    chunk_size=100,
    columns=["mukey", "cokey", "compname", "localphase", "comppct_r"]
)
df = response.to_pandas()
print(f"Fetched {len(df)} component records")
```

    Found 410 mukeys across 3 areas
    Fetching 410 keys in 5 chunks of 100
    Fetched 1067 component records

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | mukey  | cokey    | compname | localphase | comppct_r |
|-----|--------|----------|----------|------------|-----------|
| 0   | 408333 | 25562547 | Kingston | \<NA\>     | 2         |
| 1   | 408333 | 25562548 | Okoboji  | \<NA\>     | 5         |
| 2   | 408333 | 25562549 | Spicer   | \<NA\>     | 90        |
| 3   | 408333 | 25562550 | Madelia  | \<NA\>     | 3         |
| 4   | 408334 | 25562837 | Okoboji  | \<NA\>     | 5         |
| 5   | 408334 | 25562838 | Glencoe  | \<NA\>     | 3         |
| 6   | 408334 | 25562839 | Canisteo | \<NA\>     | 2         |
| 7   | 408334 | 25562840 | Webster  | \<NA\>     | 85        |
| 8   | 408334 | 25562841 | Nicollet | \<NA\>     | 5         |
| 9   | 408335 | 25562135 | Biscay   | \<NA\>     | 1         |

</div>

The `component` table has a hierarchical relationship:

- mukey (map unit key) is the parent
- cokey (component key) is the child

So when fetching components, you typically want to filter by mukey to
get all components for specific map units.

Use the `fetch_by_keys()` function with the `"mukey"` as the
`key_column` to achieve this with automatic pagination over chunks with
`100` rows each (or specify your own `chunk_size`).

## Async Usage

For performance-critical applications, use async functions directly with concurrent requests:

``` python
import asyncio
from soildb import fetch_by_keys, get_mukey_by_areasymbol

async def concurrent_example():
    # Get mukeys for multiple areas concurrently
    areas = ["IA109", "IA113", "IA117"]
    all_mukeys = await get_mukey_by_areasymbol(areas)
    
    # Fetch components concurrently with automatic pagination
    response = await fetch_by_keys(
        all_mukeys,
        "component",
        key_column="mukey",
        chunk_size=100,
        columns=["mukey", "cokey", "compname", "comppct_r"]
    )
    return response.to_pandas()

# Run async function
df = asyncio.run(concurrent_example())
```

For more async patterns, see the [Async Programming Guide](docs/async.qmd).

# Examples

See the [`examples/` directory](examples/) and [documentation](docs/)
for detailed usage patterns.

## License

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.
