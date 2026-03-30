"""
LDM (Lab Data Mart) module for accessing Kellogg Soil Survey Laboratory data.

This module provides access to laboratory soil characterization data from the
NCSS Soil Characterization Database (Kellogg Soil Survey Laboratory - KSSL)
via two backends:

1. **Soil Data Access (SDA) web service** - Real-time data via HTTP API
2. **Local SQLite snapshots** - Offline analysis using downloadable databases

Basic Usage
-----------

Query via Soil Data Access (web service)::

    from soildb import fetch_ldm

    # Fetch by laboratory pedon ID
    response = await fetch_ldm(
        x=['85P0234', '40A3306'],
        what='pedlabsampnum'
    )
    df = response.to_pandas()

Query via local SQLite snapshot::

    response = await fetch_ldm(
        x=['85P0234'],
        what='pedlabsampnum',
        dsn='path/to/LDM_FY2025.sqlite'
    )
    df = response.to_pandas()

Using the client directly::

    from soildb.ldm import LDMClient

    async with LDMClient() as client:
        response = await client.query(x=['85P0234'], what='pedlabsampnum')

Synchronous API::

    # All async functions have .sync() method for synchronous execution
    response = fetch_ldm.sync(x=['85P0234'], what='pedlabsampnum')

References
----------
- R fetchLDM documentation: https://ncss-tech.github.io/soilDB/reference/fetchLDM.html
- LDM Data Model: https://jneme910.github.io/Lab_Data_Mart_Documentation/Documents/SDA_KSSL_Data_model.html
- SQLite Downloads: https://ncsslabdatamart.sc.egov.usda.gov/database_download.aspx
"""

from .client import LDMClient
from .exceptions import (
    LDMBackendError,
    LDMBackendSelectionError,
    LDMError,
    LDMParameterError,
    LDMQueryError,
    LDMResponseError,
    LDMSDAError,
    LDMSQLiteError,
    LDMTableError,
)

__all__ = [
    "LDMClient",
    # Exceptions
    "LDMError",
    "LDMBackendError",
    "LDMSQLiteError",
    "LDMSDAError",
    "LDMBackendSelectionError",
    "LDMQueryError",
    "LDMParameterError",
    "LDMTableError",
    "LDMResponseError",
]
