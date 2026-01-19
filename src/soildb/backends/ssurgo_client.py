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
from typing import Dict, List, Optional, Union

from soildb.response import SDAResponse

from .base import BaseBackend

logger = logging.getLogger(__name__)


class SSURGOClient:
    """Client for querying SSURGO data from any backend.

    Provides high-level methods to query SSURGO tables, supporting
    all backends (SDA, SQLite, GeoPackage, PostgreSQL).

    Example:
        >>> from soildb.backends import SDABackend, SSURGOClient
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
        mukey: Optional[Union[List[int], int]] = None,
        musym: Optional[Union[List[str], str]] = None,
        muname: Optional[Union[List[str], str]] = None,
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
        cokey: Optional[Union[List[int], int]] = None,
        mukey: Optional[Union[List[int], int]] = None,
        compname: Optional[Union[List[str], str]] = None,
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
        chkey: Optional[Union[List[int], int]] = None,
        cokey: Optional[Union[List[int], int]] = None,
        hzname: Optional[Union[List[str], str]] = None,
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
        lkey: Optional[Union[List[int], int]] = None,
        areasymbol: Optional[Union[List[str], str]] = None,
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
        primary_key: Optional[Union[List[int], List[str], int, str]] = None,
        secondary_key: Optional[Union[List[int], List[str], int, str]] = None,
        tertiary_key: Optional[Union[List[str], str]] = None,
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

        # Build WHERE from provided keys
        conditions = []

        # Default field names for common tables
        field_defaults = {
            "mapunit": ("mukey", "musym", "muname"),
            "component": ("cokey", "mukey", "compname"),
            "chorizon": ("chkey", "cokey", "hzname"),
            "legend": ("lkey", "areasymbol", None),
        }

        fields = field_defaults.get(table, (None, None, None))

        # Resolve field names (prefer explicitly passed, then defaults)
        p_field = primary_field or fields[0]
        s_field = secondary_field or fields[1]
        t_field = alt_field or fields[2] if len(fields) > 2 else None

        # Add conditions for non-None values
        if primary_key is not None and p_field is not None:
            conditions.append(self._build_in_condition(p_field, primary_key))
        if secondary_key is not None and s_field is not None:
            conditions.append(self._build_in_condition(s_field, secondary_key))
        if tertiary_key is not None and t_field is not None:
            conditions.append(self._build_in_condition(t_field, tertiary_key))

        if not conditions:
            # No filter provided, return all rows
            return f"SELECT * FROM {table}"

        where_part = " AND ".join(conditions)
        return f"SELECT * FROM {table} WHERE {where_part}"

    @staticmethod
    def _build_in_condition(field: str, values: Union[List, int, str]) -> str:
        """Build SQL IN condition.

        Args:
            field: Field name
            values: Single value or list of values

        Returns:
            SQL condition string
        """
        if isinstance(values, (list, tuple)):
            # Multiple values - use IN
            if not values:
                return "1=0"  # Empty list

            # Check if values are numeric or strings
            if all(isinstance(v, (int, float)) for v in values):
                values_str = ",".join(str(v) for v in values)
                return f"{field} IN ({values_str})"
            else:
                # String values - quote them
                values_str = ",".join(f"'{v}'" for v in values)
                return f"{field} IN ({values_str})"
        else:
            # Single value
            if isinstance(values, (int, float)):
                return f"{field} = {values}"
            else:
                return f"{field} = '{values}'"

    async def get_available_tables(self) -> List[str]:
        """Get list of available SSURGO tables from backend.

        Returns:
            List of table names available in the backend
        """
        try:
            return await self.backend.get_tables()
        except Exception as e:
            logger.warning(f"Failed to get available tables: {e}")
            return []

    async def get_table_schema(self, table: str) -> Dict[str, str]:
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
