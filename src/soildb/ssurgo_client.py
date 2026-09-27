"""Generic SSURGO client for querying from any backend.

SSURGO (Soil Survey Geographic) is the USDA's detailed soil survey database.
This client provides convenient methods to query SSURGO tables from any backend:
- Soil Data Access (HTTP)
- Local SQLite snapshots
- GeoPackage files
- PostgreSQL (future)

All backends return SDAResponse for consistency.
"""

import logging
from typing import Optional, Union

from soildb.backends import BaseBackend
from soildb.query import Query
from soildb.response import SDAResponse
from soildb.ssurgo_tables import filter_fields

logger = logging.getLogger(__name__)


class SSURGOClient:
    """Client for querying SSURGO data from any backend.

    Provides high-level methods to query SSURGO tables, supporting
    all backends (SDA, SQLite, GeoPackage, PostgreSQL).

    Example:
        >>> from soildb.backends import SDABackend
        >>> from soildb.ssurgo_client import SSURGOClient
        >>> backend = SDABackend()
        >>> client = SSURGOClient(backend)
        >>> response = await client.fetch_mapunit(['IA001', 'IA002'])
        >>> df = response.to_pandas()
    """

    # SSURGO core tables
    MAPUNIT_TABLE = "mapunit"
    COMPONENT_TABLE = "component"
    CHORIZON_TABLE = "chorizon"
    LEGEND_TABLE = "legend"
    CHTEXTUREGRP_TABLE = "chtexturegrp"
    CHTEXTURE_TABLE = "chtexture"

    def __init__(self, backend: BaseBackend):
        """Initialize SSURGO client with a backend.

        Args:
            backend: Any BaseBackend instance (SDA, SQLite, GeoPackage, etc.)
        """
        self.backend = backend

    async def fetch_mapunit(
        self,
        mukey: Optional[Union[list[int], int]] = None,
        musym: Optional[Union[list[str], str]] = None,
        muname: Optional[Union[list[str], str]] = None,
        WHERE: Optional[str] = None,
    ) -> SDAResponse:
        """Query SSURGO mapunit table.

        Args:
            mukey: Mapunit key(s) to query
            musym: Mapunit symbol(s) to query
            muname: Mapunit name(s) to query
            WHERE: Custom SQL WHERE clause for advanced queries

        Returns:
            SDAResponse with mapunit records

        Example:
            >>> response = await client.fetch_mapunit(['101', '102'])
            >>> response = await client.fetch_mapunit(musym=['IA001A', 'IA001B'])
            >>> response = await client.fetch_mapunit(WHERE="muname LIKE 'Miami%'")
        """
        sql = self._build_query("mapunit", mukey, musym, muname, WHERE)
        return await self.backend.execute(sql)

    async def fetch_component(
        self,
        cokey: Optional[Union[list[int], int]] = None,
        mukey: Optional[Union[list[int], int]] = None,
        compname: Optional[Union[list[str], str]] = None,
        WHERE: Optional[str] = None,
    ) -> SDAResponse:
        """Query SSURGO component table.

        Args:
            cokey: Component key(s) to query
            mukey: Mapunit key(s) to query
            compname: Component name(s) to query
            WHERE: Custom SQL WHERE clause

        Returns:
            SDAResponse with component records

        Example:
            >>> response = await client.fetch_component(mukey='101')
            >>> response = await client.fetch_component(compname=['Miami', 'Cary'])
        """
        sql = self._build_query(
            "component",
            cokey,
            mukey,
            compname,
            WHERE,
            alt_field="compname",
        )
        return await self.backend.execute(sql)

    async def fetch_chorizon(
        self,
        chkey: Optional[Union[list[int], int]] = None,
        cokey: Optional[Union[list[int], int]] = None,
        hzname: Optional[Union[list[str], str]] = None,
        WHERE: Optional[str] = None,
    ) -> SDAResponse:
        """Query SSURGO chorizon (component horizon) table.

        Args:
            chkey: Chorizon key(s) to query
            cokey: Component key(s) to query
            hzname: Horizon name(s) to query
            WHERE: Custom SQL WHERE clause

        Returns:
            SDAResponse with chorizon records
        """
        sql = self._build_query(
            "chorizon",
            chkey,
            cokey,
            hzname,
            WHERE,
            alt_field="hzname",
        )
        return await self.backend.execute(sql)

    async def fetch_legend(
        self,
        lkey: Optional[Union[list[int], int]] = None,
        areasymbol: Optional[Union[list[str], str]] = None,
        WHERE: Optional[str] = None,
    ) -> SDAResponse:
        """Query SSURGO legend (soil survey area) table.

        Args:
            lkey: Legend key(s) to query
            areasymbol: Area symbol(s) to query (e.g., 'IA001', 'IA025')
            WHERE: Custom SQL WHERE clause

        Returns:
            SDAResponse with legend records

        Example:
            >>> response = await client.fetch_legend(areasymbol=['IA001', 'IA025'])
        """
        sql = self._build_query(
            "legend",
            lkey,
            areasymbol,
            None,
            WHERE,
            alt_field="areasymbol",
        )
        return await self.backend.execute(sql)

    def _build_query(
        self,
        table: str,
        primary_key: Optional[Union[list[int], list[str], int, str]] = None,
        secondary_key: Optional[Union[list[int], list[str], int, str]] = None,
        tertiary_key: Optional[Union[list[str], str]] = None,
        where_clause: Optional[str] = None,
        primary_field: Optional[str] = None,
        secondary_field: Optional[str] = None,
        alt_field: Optional[str] = None,
    ) -> str:
        """Build SQL query for SSURGO table.

        Args:
            table: Table name
            primary_key: Values for primary key field
            secondary_key: Values for secondary key field
            tertiary_key: Values for tertiary key field
            where_clause: Custom WHERE clause
            primary_field: Primary key field name
            secondary_field: Secondary key field name
            alt_field: Alternative field name

        Returns:
            SQL query string
        """
        if where_clause:
            # User provided WHERE clause
            return f"SELECT * FROM {table} WHERE {where_clause}"

        # Build query using Query builder
        query = Query().from_(table)

        # Get filter field names (primary, secondary, alternate) from metadata
        fields = filter_fields(table) or (None, None, None)

        # Resolve field names (prefer explicitly passed, then defaults)
        p_field = primary_field or fields[0]
        s_field = secondary_field or fields[1]
        t_field = alt_field or fields[2] if len(fields) > 2 else None

        # Add conditions for non-None values using Query.where_in/where_eq
        if primary_key is not None and p_field is not None:
            if isinstance(primary_key, (list, tuple)):
                if primary_key:  # Only add if non-empty
                    query.where_in(p_field, primary_key)
            else:
                query.where_eq(p_field, primary_key)

        if secondary_key is not None and s_field is not None:
            if isinstance(secondary_key, (list, tuple)):
                if secondary_key:  # Only add if non-empty
                    query.where_in(s_field, secondary_key)
            else:
                query.where_eq(s_field, secondary_key)

        if tertiary_key is not None and t_field is not None:
            if isinstance(tertiary_key, (list, tuple)):
                if tertiary_key:  # Only add if non-empty
                    query.where_in(t_field, tertiary_key)
            else:
                query.where_eq(t_field, tertiary_key)

        return query.to_sql()

    async def get_available_tables(self) -> list[str]:
        """Get list of available SSURGO tables from backend.

        Returns:
            List of table names available in the backend
        """
        try:
            return await self.backend.get_tables()
        except Exception as e:
            logger.warning(f"Failed to get available tables: {e}")
            return []

    async def get_table_schema(self, table: str) -> dict[str, str]:
        """Get schema for a SSURGO table.

        Args:
            table: Table name

        Returns:
            Dict mapping column names to their database types
        """
        try:
            return await self.backend.get_columns(table)
        except Exception as e:
            logger.warning(f"Failed to get schema for {table}: {e}")
            return {}


__all__ = ["SSURGOClient"]
