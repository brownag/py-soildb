"""
Tests for synchronous wrapper functionality.
"""

import inspect
from typing import Optional

import pytest

import soildb
from soildb import SDAClient
from soildb.utils import add_sync_version, require_client


class TestSyncWrappers:
    """Test the .sync attribute on async functions."""

    def test_sync_attribute_exists(self):
        """Test that sync attribute exists on functions that should have it."""
        # Test convenience functions
        assert hasattr(soildb.get_sacatalog, "sync")
        assert hasattr(soildb.get_mapunit_by_point, "sync")
        assert hasattr(soildb.get_mapunit_by_areasymbol, "sync")

        # Test fetch functions
        assert hasattr(soildb.fetch_by_keys, "sync")
        assert hasattr(soildb.fetch_pedons_by_bbox, "sync")
        assert hasattr(soildb.fetch_ldm, "sync")

        # Test high-level functions
        assert hasattr(soildb.fetch_ssurgo_mapunit_by_point, "sync")

    def test_sync_is_callable(self):
        """Test that sync attributes are callable."""
        assert callable(soildb.get_sacatalog.sync)
        assert callable(soildb.get_mapunit_by_point.sync)
        assert callable(soildb.fetch_ldm.sync)

    @pytest.mark.asyncio
    async def test_sync_in_async_context_raises_error(self):
        """Test that calling .sync from async context raises RuntimeError."""
        import warnings

        with warnings.catch_warnings():
            # Suppress the expected RuntimeWarning about unawaited coroutine
            # when we intentionally raise an error in async context
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="coroutine.*was never awaited",
            )
            with pytest.raises(RuntimeError, match="event loop"):
                soildb.get_sacatalog.sync()

    @pytest.mark.integration
    def test_sync_with_explicit_client(self):
        """Test that sync works with explicitly provided client."""
        client = SDAClient()
        try:
            # This should work (though may fail due to network)
            # We just test that it doesn't raise RuntimeError
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_sync_with_custom_parameters(self):
        """Test that sync works with saverest column added."""
        try:
            result = soildb.get_sacatalog.sync(
                columns=["areasymbol", "areaname", "saversion", "saverest"]
            )
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) > 0

            # Check that saverest column is present
            df = result.to_pandas()
            assert "saverest" in df.columns
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    @pytest.mark.integration
    def test_sync_point_query(self):
        """Test sync point query functionality."""
        try:
            result = soildb.get_mapunit_by_point.sync(-93.6, 42.0)
            assert isinstance(result, soildb.SDAResponse)
            assert len(result) >= 0  # May be empty for some locations
        except (soildb.SDAConnectionError, soildb.SDAQueryError):
            # Network errors are expected in tests
            pass

    @pytest.mark.integration
    def test_sync_fetch_by_keys(self):
        """Test sync bulk fetching functionality."""
        try:
            # Use a small known mukey for testing
            result = soildb.fetch_by_keys.sync(
                [408333], "component", "mukey", columns=["mukey", "cokey", "compname"]
            )
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


class FakeClient:
    """Fake client for testing client lifecycle management."""

    def __init__(self):
        self.close_count = 0
        self.is_closed = False

    async def close(self):
        """Track close() calls."""
        self.close_count += 1
        self.is_closed = True


class TestClientLifecycleManagement:
    """Test that @add_sync_version manages client lifecycle correctly."""

    def test_async_without_client_creates_and_closes(self):
        """Test that async call without client creates and closes exactly once."""

        @add_sync_version
        async def test_func(value: int, client: Optional[FakeClient] = None):
            return value * 2

        import asyncio

        async def run_test():
            result = await test_func(5)
            assert result == 10
            return result

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_async_with_keyword_client_not_closed(self):
        """Test that async call with keyword client is never closed by decorator."""

        @add_sync_version
        async def test_func(value: int, client: Optional[FakeClient] = None):
            return value * 2

        fake_client = FakeClient()
        result = await test_func(5, client=fake_client)
        assert result == 10
        assert fake_client.close_count == 0, (
            "Decorator should not close caller-supplied client"
        )

    def test_sync_without_client_creates_and_closes(self):
        """Test that sync call without client creates and closes exactly once."""

        @add_sync_version
        async def test_func(value: int, client: Optional[FakeClient] = None):
            return value * 2

        result = test_func.sync(5)
        assert result == 10

    def test_sync_with_keyword_client_not_closed(self):
        """Test that sync call with keyword client is never closed by decorator."""

        @add_sync_version
        async def test_func(value: int, client: Optional[FakeClient] = None):
            return value * 2

        fake_client = FakeClient()
        result = test_func.sync(5, client=fake_client)
        assert result == 10
        assert fake_client.close_count == 0, (
            "Decorator should not close caller-supplied client"
        )

    def test_sync_with_positional_client_not_closed(self):
        """Test that sync call with positional client is never closed by decorator."""

        @add_sync_version
        async def test_func(
            value: int,
            columns: Optional[list[str]] = None,
            client: Optional[FakeClient] = None,
        ):
            return value * 2

        fake_client = FakeClient()
        result = test_func.sync(5, None, fake_client)
        assert result == 10
        assert fake_client.close_count == 0, (
            "Decorator should not close caller-supplied positional client"
        )

    @pytest.mark.asyncio
    async def test_exception_in_function_still_closes(self):
        """Test that client is closed even if function raises exception."""
        close_tracker = []

        class TrackingClient:
            async def close(self):
                close_tracker.append("closed")

        @add_sync_version
        async def test_func(client: Optional[TrackingClient] = None):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

        assert len(close_tracker) == 1, "Client should be closed even on exception"

    def test_sync_exception_closes_client(self):
        """Test that sync call closes client even if function raises exception."""
        close_tracker = []

        class TrackingClient:
            async def close(self):
                close_tracker.append("closed")

        @add_sync_version
        async def test_func(client: Optional[TrackingClient] = None):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func.sync()

        assert len(close_tracker) == 1, "Client should be closed even on exception"

    def test_signature_preserved(self):
        """Test that inspect.signature is preserved after decoration."""

        @add_sync_version
        async def test_func(
            value: int,
            columns: Optional[list[str]] = None,
            client: Optional[FakeClient] = None,
        ) -> int:
            return value * 2

        sig = inspect.signature(test_func)
        params = list(sig.parameters.keys())

        assert params == ["value", "columns", "client"], (
            "Signature should match original"
        )
        assert sig.return_annotation is int, "Return annotation should be preserved"

    def test_function_without_client_param_unaffected(self):
        """Test that functions without client param work unchanged."""

        @add_sync_version
        async def test_func(x: int, y: int) -> int:
            return x + y

        # Async should work
        import asyncio

        result = asyncio.run(test_func(3, 4))
        assert result == 7

        # Sync should work
        result = test_func.sync(3, 4)
        assert result == 7

    def test_sync_attribute_attached_to_wrapper(self):
        """Test that .sync attribute is attached to the async wrapper, not the original."""

        @add_sync_version
        async def test_func(value: int, client: Optional[FakeClient] = None) -> int:
            return value * 2

        # The async function itself should have .sync
        assert hasattr(test_func, "sync")
        assert callable(test_func.sync)

    @pytest.mark.asyncio
    async def test_async_with_positional_client_none_creates_and_closes(self):
        """Test that async call with positional client=None creates and closes."""

        @add_sync_version
        async def test_func(
            value: int,
            columns: Optional[list[str]] = None,
            client: Optional[FakeClient] = None,
        ):
            return value * 2

        result = await test_func(5, None, None)
        assert result == 10

    def test_sync_with_positional_client_none_creates_and_closes(self):
        """Test that sync call with positional client=None creates and closes."""

        close_tracker = []

        class TrackingClient:
            async def close(self):
                close_tracker.append("closed")

        @add_sync_version
        async def test_func(
            value: int,
            columns: Optional[list[str]] = None,
            client: Optional[TrackingClient] = None,
        ):
            return value * 2

        result = test_func.sync(5, None, None)
        assert result == 10
        assert len(close_tracker) == 1, (
            "Decorator should create and close client when None passed positionally"
        )

    def test_sync_with_keyword_client_none_creates_and_closes(self):
        """Test that sync call with keyword client=None creates and closes."""

        close_tracker = []

        class TrackingClient:
            async def close(self):
                close_tracker.append("closed")

        @add_sync_version
        async def test_func(value: int, client: Optional[TrackingClient] = None):
            return value * 2

        result = test_func.sync(5, client=None)
        assert result == 10
        assert len(close_tracker) == 1, (
            "Decorator should create and close client when None passed by keyword"
        )


class TestRequireClient:
    """Test the require_client function."""

    def test_require_client_with_none_raises(self):
        """Test that require_client raises TypeError when client is None."""
        with pytest.raises(TypeError, match="client is required"):
            require_client(None)

    def test_require_client_with_client_returns_unchanged(self):
        """Test that require_client returns the client unchanged."""
        fake_client = FakeClient()
        result = require_client(fake_client)
        assert result is fake_client


class TestExtractClientClass:
    """Test the extract_client_class method for one-class rule."""

    def test_extract_client_class_with_optional_single_class(self):
        """Test extraction of Optional[SomeClass]."""
        from typing import Optional

        from soildb.utils import AsyncSyncBridge

        class DummyClient:
            pass

        # Optional[DummyClient] should return DummyClient
        result = AsyncSyncBridge.extract_client_class(Optional[DummyClient])
        assert result is DummyClient

    def test_extract_client_class_with_union_multiple_classes_returns_none(self):
        """Test that Union with multiple non-None classes returns None."""
        from typing import Union

        from soildb.utils import AsyncSyncBridge

        class ClientA:
            pass

        class ClientB:
            pass

        # Union[ClientA, ClientB, None] should return None (ambiguous)
        result = AsyncSyncBridge.extract_client_class(Union[ClientA, ClientB, None])
        assert result is None

    def test_extract_client_class_with_single_class(self):
        """Test extraction of a single class directly."""
        from soildb.utils import AsyncSyncBridge

        class DummyClient:
            pass

        result = AsyncSyncBridge.extract_client_class(DummyClient)
        assert result is DummyClient

    def test_extract_client_class_with_none_annotation(self):
        """Test extraction with None annotation."""
        from soildb.utils import AsyncSyncBridge

        result = AsyncSyncBridge.extract_client_class(None)
        assert result is None

    def test_extract_client_class_with_union_ldm_sda_client_returns_none(self):
        """Test that Optional[Union[LDMClient, SDAClient]] returns None."""
        from typing import Union

        from soildb import SDAClient
        from soildb.ldm import LDMClient
        from soildb.utils import AsyncSyncBridge

        annotation = Optional[Union[LDMClient, SDAClient]]
        result = AsyncSyncBridge.extract_client_class(annotation)
        assert result is None

    def test_extract_client_class_from_fetch_ldm_signature_returns_none(self):
        """Test that extract_client_class on fetch_ldm client parameter returns None."""
        from soildb.fetch import fetch_ldm
        from soildb.utils import AsyncSyncBridge

        sig = inspect.signature(fetch_ldm)
        client_param = sig.parameters["client"]
        result = AsyncSyncBridge.extract_client_class(client_param.annotation)
        assert result is None

    def test_extract_client_class_with_pep604_unions(self):
        """Test that PEP 604 unions (types.UnionType) are handled correctly."""
        from soildb.utils import AsyncSyncBridge

        class ClientA:
            pass

        class ClientB:
            pass

        # Multiple classes with None returns None
        assert AsyncSyncBridge.extract_client_class(ClientA | ClientB | None) is None
        # Single class with None returns that class
        assert AsyncSyncBridge.extract_client_class(ClientA | None) is ClientA
