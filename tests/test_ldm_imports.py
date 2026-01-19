"""Test LDM module imports and basic structure."""

import pytest


class TestLDMImports:
    """Test that LDM module can be imported and has expected structure."""

    def test_ldm_module_importable(self):
        """Test that ldm module can be imported."""
        from soildb import ldm
        assert ldm is not None

    def test_ldm_client_importable(self):
        """Test that LDMClient can be imported."""
        from soildb import LDMClient
        assert LDMClient is not None

    def test_fetch_ldm_importable(self):
        """Test that fetch_ldm can be imported."""
        from soildb import fetch_ldm
        assert fetch_ldm is not None

    def test_ldm_exceptions_importable(self):
        """Test that LDM exceptions can be imported."""
        from soildb import (
            LDMError,
            LDMBackendError,
            LDMSQLiteError,
            LDMSDAError,
            LDMParameterError,
            LDMQueryError,
        )
        assert LDMError is not None
        assert LDMBackendError is not None
        assert LDMSQLiteError is not None
        assert LDMSDAError is not None
        assert LDMParameterError is not None
        assert LDMQueryError is not None

    def test_ldm_module_exports(self):
        """Test that ldm module has expected exports."""
        from soildb import ldm

        assert hasattr(ldm, "LDMClient")
        assert hasattr(ldm, "LDMError")
        assert hasattr(ldm, "LDMBackendError")

    def test_ldm_client_has_async_methods(self):
        """Test that LDMClient has expected async methods."""
        from soildb import LDMClient

        assert hasattr(LDMClient, "query")
        assert hasattr(LDMClient, "connect")
        assert hasattr(LDMClient, "close")
        assert hasattr(LDMClient, "get_available_tables")
        assert hasattr(LDMClient, "get_table_schema")

    def test_fetch_ldm_has_sync_version(self):
        """Test that fetch_ldm has .sync() method."""
        from soildb import fetch_ldm

        assert hasattr(fetch_ldm, "sync")

    def test_ldm_tables_module_importable(self):
        """Test that ldm.tables module can be imported."""
        from soildb.ldm import tables

        assert tables is not None
        assert hasattr(tables, "DEFAULT_TABLES")
        assert hasattr(tables, "ALL_TABLES")

    def test_ldm_exceptions_module_importable(self):
        """Test that ldm.exceptions module can be imported."""
        from soildb.ldm import exceptions

        assert exceptions is not None
        assert hasattr(exceptions, "LDMError")

    def test_ldm_query_builder_importable(self):
        """Test that query builder can be imported."""
        from soildb.ldm.query_builder import LDMQueryBuilder, build_ldm_query

        assert LDMQueryBuilder is not None
        assert build_ldm_query is not None

    def test_ldm_backends_importable(self):
        """Test that backends can be imported."""
        from soildb.ldm.backends import SDABackend, SQLiteBackend, LDMBackend

        assert SDABackend is not None
        assert SQLiteBackend is not None
        assert LDMBackend is not None

    def test_ldm_in_main_init_all(self):
        """Test that LDMClient and fetch_ldm are in soildb.__all__."""
        import soildb

        assert "LDMClient" in soildb.__all__
        assert "fetch_ldm" in soildb.__all__
        assert "ldm" in soildb.__all__
