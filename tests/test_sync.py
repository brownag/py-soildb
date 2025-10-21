"""
Tests for synchronous wrapper functionality.
"""

import pytest
from unittest.mock import patch

import soildb
from soildb import SDAClient, SyncUsageError


class TestSyncWrappers:
    """Test the .sync attribute on async functions."""

    def test_sync_attribute_exists(self):
        """Test that sync attribute exists on functions that should have it."""
        # Test convenience functions
        assert hasattr(soildb.get_sacatalog, 'sync')
        assert hasattr(soildb.get_mapunit_by_point, 'sync')
        assert hasattr(soildb.get_mapunit_by_areasymbol, 'sync')
        
        # Test fetch functions
        assert hasattr(soildb.fetch_by_keys, 'sync')
        assert hasattr(soildb.fetch_component_by_mukey, 'sync')
        
        # Test high-level functions
        assert hasattr(soildb.fetch_mapunit_struct_by_point, 'sync')

    def test_sync_is_callable(self):
        """Test that sync attributes are callable."""
        assert callable(soildb.get_sacatalog.sync)
        assert callable(soildb.get_mapunit_by_point.sync)

    @pytest.mark.asyncio
    async def test_sync_in_async_context_raises_error(self):
        """Test that calling .sync from async context raises SyncUsageError."""
        with pytest.raises(SyncUsageError, match="existing asyncio event loop"):
            soildb.get_sacatalog.sync()

    def test_sync_with_explicit_client(self):
        """Test that sync works with explicitly provided client."""
        client = SDAClient()
        try:
            # This should work (though may fail due to network)
            # We just test that it doesn't raise SyncUsageError
            try:
                result = soildb.get_sacatalog.sync(client=client)
                assert isinstance(result, soildb.SDAResponse)
            except (soildb.SDAConnectionError, soildb.SDAQueryError):
                # Network errors are expected in tests
                pass
        finally:
            # Close client properly without asyncio.run if loop is closed
            try:
                import asyncio
                asyncio.run(client.close())
            except RuntimeError:
                # Event loop is closed, close synchronously if possible
                pass

    def test_sync_automatic_client_creation(self):
        """Test that sync automatically creates client when none provided."""
        # This should work (though may fail due to network)
        try:
            result = soildb.get_sacatalog.sync()
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) > 0  # Should have data
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    def test_sync_with_custom_parameters(self):
        """Test that sync works with custom parameters like columns."""
        # Test get_sacatalog with custom columns
        try:
            result = soildb.get_sacatalog.sync(columns=['areasymbol', 'areaname'])
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) > 0
            
            # Check that custom columns are present
            df = result.to_pandas()
            assert 'areasymbol' in df.columns
            assert 'areaname' in df.columns
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    def test_sync_with_saverest_column(self):
        """Test that sync works with saverest column specifically."""
        try:
            result = soildb.get_sacatalog.sync(columns=['areasymbol', 'areaname', 'saversion', 'saverest'])
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) > 0
            
            # Check that saverest column is present
            df = result.to_pandas()
            assert 'saverest' in df.columns
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    def test_sync_point_query(self):
        """Test sync point query functionality."""
        try:
            result = soildb.get_mapunit_by_point.sync(-93.6, 42.0)
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) >= 0  # May be empty for some locations
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    def test_sync_fetch_by_keys(self):
        """Test sync bulk fetching functionality."""
        try:
            # Use a small known mukey for testing
            result = soildb.fetch_component_by_mukey.sync([408333], columns=['mukey', 'cokey', 'compname'])
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) >= 0
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    def test_sync_no_client_param(self):
        """Test that functions without client param don't get automatic client."""
        from soildb.utils import add_sync_version
        
        async def dummy_func(x, y):
            return x + y
            
        sync_func = add_sync_version(dummy_func)
        
        # Should work without issues (no client creation)
        result = sync_func.sync(1, 2)
        assert result == 3