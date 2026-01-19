"""
Tests for GeoPackageBackend implementation.

Tests verify GeoPackage-specific functionality:
- Geometry column detection
- Spatial metadata awareness
- Filtering of system tables
"""

import pytest
import aiosqlite
from pathlib import Path

from soildb.backends import GeoPackageBackend
from soildb.backends.exceptions import BackendSchemaError


class TestGeoPackageBackend:
    """Tests for GeoPackageBackend."""

    @pytest.mark.asyncio
    async def test_geopackage_backend_initialization(self, tmp_path):
        """GeoPackageBackend should initialize like SQLiteBackend."""
        db_file = tmp_path / "test.gpkg"
        db_file.touch()

        backend = GeoPackageBackend(db_file)
        assert backend.db_path == db_file

    @pytest.mark.asyncio
    async def test_geopackage_backend_connect(self, tmp_path):
        """GeoPackageBackend should connect like SQLiteBackend."""
        db_file = tmp_path / "test.gpkg"
        db_file.touch()

        backend = GeoPackageBackend(db_file)
        result = await backend.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_geopackage_filters_system_tables(self, tmp_path):
        """GeoPackageBackend should filter out system tables."""
        db_file = tmp_path / "test.gpkg"

        # Create GeoPackage with system tables
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            await db.execute("CREATE TABLE gpkg_geometry_columns (id INTEGER)")
            await db.execute("CREATE TABLE gpkg_spatial_ref_sys (id INTEGER)")
            await db.commit()

        backend = GeoPackageBackend(db_file)
        tables = await backend.get_tables()

        # Should only include user table
        assert "users" in tables
        assert "gpkg_geometry_columns" not in tables
        assert "gpkg_spatial_ref_sys" not in tables

    @pytest.mark.asyncio
    async def test_geopackage_get_geometry_column_not_found(self, tmp_path):
        """GeoPackageBackend should return None for non-spatial table."""
        db_file = tmp_path / "test.gpkg"

        # Create simple table without geometry
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            await db.commit()

        backend = GeoPackageBackend(db_file)
        geom_col = await backend.get_geometry_column("users")

        assert geom_col is None

    @pytest.mark.asyncio
    async def test_geopackage_context_manager(self, tmp_path):
        """GeoPackageBackend should work as async context manager."""
        db_file = tmp_path / "test.gpkg"
        db_file.touch()

        backend = GeoPackageBackend(db_file)

        async with backend:
            assert backend._connected is True

        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_geopackage_query_with_data(self, tmp_path):
        """GeoPackageBackend should execute queries like SQLiteBackend."""
        db_file = tmp_path / "test.gpkg"

        # Create GeoPackage with test data
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute("CREATE TABLE features (id INTEGER, name TEXT)")
            await db.execute("INSERT INTO features VALUES (1, 'Feature1')")
            await db.execute("INSERT INTO features VALUES (2, 'Feature2')")
            await db.commit()

        backend = GeoPackageBackend(db_file)
        async with backend:
            response = await backend.execute("SELECT * FROM features ORDER BY id")
            df = response.to_pandas()

            assert len(df) == 2
            assert "id" in df.columns
            assert "name" in df.columns

    @pytest.mark.asyncio
    async def test_geopackage_repr(self, tmp_path):
        """GeoPackageBackend should have useful string representation."""
        db_file = tmp_path / "test.gpkg"
        db_file.touch()

        backend = GeoPackageBackend(db_file)
        repr_str = repr(backend)

        assert "GeoPackageBackend" in repr_str
        assert "test.gpkg" in repr_str


class TestGeoPackageBackendIntegration:
    """Integration tests for GeoPackageBackend."""

    @pytest.mark.asyncio
    async def test_geopackage_as_spatial_backend(self, tmp_path):
        """GeoPackageBackend should support spatial operations."""
        db_file = tmp_path / "test.gpkg"

        # Create a GeoPackage-like structure
        async with aiosqlite.connect(str(db_file)) as db:
            # Create geometry columns metadata table (simplified)
            await db.execute(
                """
                CREATE TABLE gpkg_geometry_columns (
                    table_name TEXT,
                    column_name TEXT,
                    geometry_type_name TEXT
                )
                """
            )

            # Create a spatial table
            await db.execute("CREATE TABLE points (id INTEGER, geom BLOB)")

            # Register geometry column
            await db.execute(
                """
                INSERT INTO gpkg_geometry_columns
                VALUES ('points', 'geom', 'POINT')
                """
            )

            await db.commit()

        backend = GeoPackageBackend(db_file)
        geom_col = await backend.get_geometry_column("points")

        assert geom_col == "geom"

    @pytest.mark.asyncio
    async def test_geopackage_caches_geometry_columns(self, tmp_path):
        """GeoPackageBackend should cache geometry column lookups."""
        db_file = tmp_path / "test.gpkg"

        # Create GeoPackage
        async with aiosqlite.connect(str(db_file)) as db:
            await db.execute(
                """
                CREATE TABLE gpkg_geometry_columns (
                    table_name TEXT,
                    column_name TEXT
                )
                """
            )
            await db.execute("CREATE TABLE points (id INTEGER)")
            await db.execute(
                "INSERT INTO gpkg_geometry_columns VALUES ('points', 'geom')"
            )
            await db.commit()

        backend = GeoPackageBackend(db_file)

        # First call queries metadata
        result1 = await backend.get_geometry_column("points")
        assert result1 == "geom"

        # Check cache was populated
        assert "points" in backend._geometry_columns_cache

        # Second call uses cache
        result2 = await backend.get_geometry_column("points")
        assert result2 == "geom"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
