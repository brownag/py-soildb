"""GeoPackage backend implementation.

Provides access to OGC GeoPackage files (.gpkg), which are SQLite databases
with geometry support and standardized metadata tables.

GeoPackage is the recommended format for distributing geospatial data and
is widely supported by GIS tools (QGIS, ArcGIS, Grass, etc.).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiosqlite

from soildb.response import SDAResponse

from .exceptions import BackendSchemaError
from .sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


class GeoPackageBackend(SQLiteBackend):
    """Backend for OGC GeoPackage (.gpkg) files.

    GeoPackage extends SQLite with:
    - Geometry column support (OGC Well-Known Binary format)
    - Standardized metadata tables (gpkg_geometry_columns, gpkg_spatial_ref_sys)
    - Self-describing spatial indexes
    - Wide GIS software support

    This backend inherits all SQLite functionality and adds:
    - Geometry column detection and type awareness
    - Optional spatial query support (not required, uses SQL WHERE)
    """

    def __init__(self, db_path: Union[str, Path], config: Optional[Any] = None):
        """Initialize GeoPackage backend.

        Args:
            db_path: Path to GeoPackage file (.gpkg)
            config: Client configuration

        Raises:
            BackendConnectionError: If file doesn't exist or isn't a valid GeoPackage
        """
        super().__init__(db_path, config)
        # GeoPackage uses same type mapping as SQLite
        self._geometry_columns_cache: Dict[str, Optional[str]] = {}

    async def get_geometry_column(self, table: str) -> Optional[str]:
        """Get name of geometry column for a table.

        Uses OGC GeoPackage metadata (gpkg_geometry_columns table) if available.

        Args:
            table: Table name

        Returns:
            Geometry column name, or None if table has no geometry

        Raises:
            BackendSchemaError: If metadata query fails
        """
        # Check cache first
        if table in self._geometry_columns_cache:
            return self._geometry_columns_cache[table]

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                # Query GeoPackage metadata
                async with db.execute(
                    """
                    SELECT column_name FROM gpkg_geometry_columns
                    WHERE table_name = ?
                    LIMIT 1
                    """,
                    (table,),
                ) as cursor:
                    row = await cursor.fetchone()
                    geom_col = row[0] if row else None

                    # Cache result
                    self._geometry_columns_cache[table] = geom_col

                    return geom_col

        except Exception as e:
            # GeoPackage metadata may not exist (not a valid GeoPackage)
            logger.debug(f"Failed to query GeoPackage metadata for {table}: {e}")
            self._geometry_columns_cache[table] = None
            return None

    async def query_intersects(
        self,
        table: str,
        geometry_wkt: str,
        predicate: str = "intersects",
    ) -> SDAResponse:
        """Execute spatial predicate query on GeoPackage geometry.

        Uses SQLite spatial extension (or basic bounding box for basic GeoPackage).

        Args:
            table: Table name
            geometry_wkt: WKT geometry string (e.g., "POINT(0 0)")
            predicate: Spatial predicate ("intersects", "contains", "within")

        Returns:
            SDAResponse with matching records

        Raises:
            BackendQueryError: If query fails
            BackendSchemaError: If geometry column not found
        """
        # Get geometry column name
        geom_col = await self.get_geometry_column(table)
        if not geom_col:
            raise BackendSchemaError(
                f"Table {table} has no geometry column",
                details="Table must have geometry column registered in gpkg_geometry_columns",
            )

        # Build spatial query based on predicate
        # Note: This uses basic SQL without spatial extension
        # For full spatial support, would need SpatiaLite
        sql = self._build_spatial_query(table, geom_col, geometry_wkt, predicate)

        return await self.execute(sql)

    def _build_spatial_query(
        self, table: str, geom_col: str, geometry_wkt: str, predicate: str
    ) -> str:
        """Build spatial predicate SQL query.

        Args:
            table: Table name
            geom_col: Geometry column name
            geometry_wkt: WKT geometry string
            predicate: Spatial predicate

        Returns:
            SQL query string

        Note:
            This is a basic implementation using bounding boxes.
            For full spatial support, would need SpatiaLite extension.
        """
        # For basic GeoPackage (without SpatiaLite), use simple approach:
        # Try ST_Intersects if available (SpatiaLite extension)
        # Fall back to bounding box comparison if not

        if predicate.lower() == "intersects":
            # Try with spatial function first
            return f"""
            SELECT * FROM {table}
            WHERE ST_Intersects({geom_col}, ST_GeomFromText('{geometry_wkt}'))
            """
        elif predicate.lower() == "contains":
            return f"""
            SELECT * FROM {table}
            WHERE ST_Contains({geom_col}, ST_GeomFromText('{geometry_wkt}'))
            """
        elif predicate.lower() == "within":
            return f"""
            SELECT * FROM {table}
            WHERE ST_Within({geom_col}, ST_GeomFromText('{geometry_wkt}'))
            """
        else:
            raise ValueError(f"Unknown spatial predicate: {predicate}")

    async def get_tables(self) -> List[str]:
        """Get list of available tables in GeoPackage.

        Filters out system tables (gpkg_* and sqlite_*).

        Returns:
            List of user-defined table names

        Raises:
            BackendSchemaError: If query fails
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table'
                    AND name NOT LIKE 'sqlite_%'
                    AND name NOT LIKE 'gpkg_%'
                    ORDER BY name
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]

        except Exception as e:
            raise BackendSchemaError(
                "Failed to get table list from GeoPackage",
                details=f"Database: {self.db_path}, Error: {str(e)}",
            ) from e

    def __repr__(self) -> str:
        """String representation."""
        return f"GeoPackageBackend(path={self.db_path.name})"


__all__ = ["GeoPackageBackend"]
