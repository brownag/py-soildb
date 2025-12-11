"""
Tests for the fetch module (key-based bulk data retrieval).
"""

from unittest.mock import AsyncMock, patch

import pytest

from soildb.client import SDAClient
from soildb.fetch import (
    TABLE_KEY_MAPPING,
    FetchError,
    _format_key_for_sql,
    _get_geometry_column_for_table,
    fetch_by_keys,
    fetch_pedons_by_bbox,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
)
from soildb.response import SDAResponse


class TestKeyFormatting:
    """Test key formatting for SQL."""

    def test_format_string_key(self):
        """Test formatting string keys."""
        assert _format_key_for_sql("CA630") == "'CA630'"
        assert _format_key_for_sql("test'quote") == "'test''quote'"

    def test_format_numeric_key(self):
        """Test formatting numeric keys."""
        assert _format_key_for_sql(123456) == "123456"
        assert _format_key_for_sql(123456.0) == "123456.0"


class TestGeometryColumns:
    """Test geometry column mapping."""

    def test_known_tables(self):
        """Test geometry column detection for known tables."""
        assert _get_geometry_column_for_table("mupolygon") == "mupolygongeo"
        assert _get_geometry_column_for_table("sapolygon") == "sapolygongeo"

    def test_unknown_table(self):
        """Test geometry column detection for unknown tables."""
        assert _get_geometry_column_for_table("unknown") is None


class TestTableKeyMapping:
    """Test the table-key mapping."""

    def test_core_tables(self):
        """Test key mapping for core tables."""
        assert TABLE_KEY_MAPPING["mapunit"] == "mukey"
        assert TABLE_KEY_MAPPING["component"] == "cokey"
        assert TABLE_KEY_MAPPING["chorizon"] == "chkey"

    def test_spatial_tables(self):
        """Test key mapping for spatial tables."""
        assert TABLE_KEY_MAPPING["mupolygon"] == "mukey"
        assert TABLE_KEY_MAPPING["sapolygon"] == "areasymbol"


@pytest.mark.asyncio
class TestFetchByKeys:
    """Test the main fetch_by_keys function."""

    async def test_empty_keys_error(self):
        """Test that empty keys list raises error."""
        mock_client = AsyncMock(spec=SDAClient)
        with pytest.raises(
            FetchError, match="The 'keys' parameter cannot be an empty list."
        ):
            await fetch_by_keys([], "mapunit", client=mock_client)

    async def test_unknown_table_error(self):
        """Test that unknown table without key_column raises error."""
        mock_client = AsyncMock(spec=SDAClient)
        with pytest.raises(FetchError, match="Unknown table"):
            await fetch_by_keys([1, 2, 3], "unknown_table", client=mock_client)

    async def test_single_chunk(self):
        """Test fetch with keys that fit in single chunk."""
        # Mock client and response
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)
        mock_response.data = [{"mukey": 123456, "muname": "Test Unit"}]

        mock_client.execute.return_value = mock_response

        result = await fetch_by_keys([123456], "mapunit", client=mock_client)

        assert result == mock_response
        mock_client.execute.assert_called_once()

    async def test_multiple_chunks(self):
        """Test fetch with keys requiring multiple chunks."""
        # Mock client and responses
        mock_client = AsyncMock(spec=SDAClient)
        mock_response1 = AsyncMock(spec=SDAResponse)
        mock_response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        mock_response1.columns = ["mukey", "muname"]
        mock_response1.metadata = ["Int", "NVarChar"]
        mock_response1.is_empty.return_value = False
        mock_response1.validation_result = None

        mock_response2 = AsyncMock(spec=SDAResponse)
        mock_response2.data = [{"mukey": 2, "muname": "Unit 2"}]
        mock_response2.columns = ["mukey", "muname"]  # Same schema!
        mock_response2.metadata = ["Int", "NVarChar"]
        mock_response2.is_empty.return_value = False
        mock_response2.validation_result = None

        mock_client.execute.side_effect = [mock_response1, mock_response2]

        #  use chunk_size=1 to force multiple chunks
        result = await fetch_by_keys(
            [1, 2], "mapunit", chunk_size=1, client=mock_client
        )

        assert len(result.data) == 2
        assert result.data[0]["mukey"] == 1
        assert result.data[1]["mukey"] == 2

    async def test_custom_columns(self):
        """Test fetch with custom column selection."""
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)
        mock_client.execute.return_value = mock_response

        await fetch_by_keys(
            [123456], "mapunit", columns=["mukey", "muname"], client=mock_client
        )

        # Check that query was built with correct columns
        # The Query object should have the specified columns
        # (This is a simplified check - in real implementation we'd check the SQL)
        assert mock_client.execute.called

    async def test_include_geometry(self):
        """Test fetch with geometry inclusion."""
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)
        mock_client.execute.return_value = mock_response

        await fetch_by_keys(
            [123456], "mupolygon", include_geometry=True, client=mock_client
        )

        assert mock_client.execute.called


@pytest.mark.asyncio
class TestSpecializedFunctions:
    """Test the specialized fetch functions."""


@pytest.mark.asyncio
class TestKeyExtractionHelpers:
    """Test helper functions for extracting keys."""

    async def test_get_mukey_by_areasymbol(self):
        """Test getting mukeys from area symbols."""
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)

        # Mock pandas DataFrame
        mock_df = AsyncMock()
        mock_df.empty = False
        mock_df.__getitem__.return_value.tolist.return_value = [123456, 123457]
        mock_response.to_pandas.return_value = mock_df

        mock_client.execute.return_value = mock_response

        result = await get_mukey_by_areasymbol(["CA630", "CA632"], client=mock_client)

        assert result == [123456, 123457]
        mock_client.execute.assert_called_once()

    @patch("soildb.fetch.fetch_by_keys")
    async def test_get_cokey_by_mukey(self, mock_fetch):
        """Test getting cokeys from mukeys."""
        mock_response = AsyncMock(spec=SDAResponse)

        # Mock pandas DataFrame
        mock_df = AsyncMock()
        mock_df.empty = False
        mock_df.__getitem__.return_value.tolist.return_value = ["123456:1", "123456:2"]
        mock_response.to_pandas.return_value = mock_df

        mock_fetch.return_value = mock_response

        result = await get_cokey_by_mukey([123456])

        assert result == ["123456:1", "123456:2"]
        mock_fetch.assert_called_once_with(
            [123456], "component", "mukey", "cokey", client=None
        )


@pytest.mark.asyncio
class TestFetchPedonsByBbox:
    """Test the fetch_pedons_by_bbox function."""

    async def test_fetch_pedons_chunking_bug_regression(self):
        """Test that chunking doesn't cause UnboundLocalError for horizons_response.

        This test reproduces the bug where pedon_keys > chunk_size would cause
        an UnboundLocalError when trying to access horizons_response.columns
        and horizons_response.metadata in the reconstruction code.
        """
        # Mock client
        mock_client = AsyncMock(spec=SDAClient)

        # Mock site response with many pedon keys
        site_response = AsyncMock(spec=SDAResponse)
        site_response.is_empty.return_value = False
        # Create mock DataFrame with 5 pedon keys
        mock_site_df = AsyncMock()
        mock_site_df.__getitem__.return_value.unique.return_value.tolist.return_value = [
            "1001",
            "1002",
            "1003",
            "1004",
            "1005",
        ]
        site_response.to_pandas.return_value = mock_site_df
        mock_client.execute.side_effect = [
            site_response
        ]  # First call returns site data

        # Mock horizon responses for chunks
        # First chunk: empty
        empty_chunk_response = AsyncMock(spec=SDAResponse)
        empty_chunk_response.is_empty.return_value = True

        # Second chunk: has data
        data_chunk_response = AsyncMock(spec=SDAResponse)
        data_chunk_response.is_empty.return_value = False
        data_chunk_response.data = [
            {"layer_key": 1, "hzn_top": 0, "hzn_bot": 10, "pedon_key": "1003"},
            {"layer_key": 2, "hzn_top": 10, "hzn_bot": 20, "pedon_key": "1003"},
        ]
        data_chunk_response.columns = ["layer_key", "hzn_top", "hzn_bot", "pedon_key"]
        data_chunk_response.metadata = ["meta1", "meta2"]

        # Third chunk: has data
        data_chunk_response2 = AsyncMock(spec=SDAResponse)
        data_chunk_response2.is_empty.return_value = False
        data_chunk_response2.data = [
            {"layer_key": 3, "hzn_top": 0, "hzn_bot": 15, "pedon_key": "1004"},
        ]
        data_chunk_response2.columns = ["layer_key", "hzn_top", "hzn_bot", "pedon_key"]
        data_chunk_response2.metadata = ["meta1", "meta2"]

        # Set up the side effects: site query, then horizon chunks
        mock_client.execute.side_effect = [
            site_response,  # Site query
            empty_chunk_response,  # First horizon chunk (empty)
            data_chunk_response,  # Second horizon chunk (has data)
            data_chunk_response2,  # Third horizon chunk (has data)
        ]

        # Call with small chunk_size to force chunking
        bbox = (-95.0, 40.0, -94.0, 41.0)
        result = await fetch_pedons_by_bbox(
            bbox, chunk_size=2, return_type="combined", client=mock_client
        )

        # Verify the result structure
        assert "site" in result
        assert "horizons" in result
        assert result["site"] == site_response

        # Verify horizons response was reconstructed correctly
        horizons_response = result["horizons"]
        assert not horizons_response.is_empty()
        assert len(horizons_response.data) == 3  # Combined data from chunks
        assert horizons_response.columns == [
            "layer_key",
            "hzn_top",
            "hzn_bot",
            "pedon_key",
        ]
        assert horizons_response.metadata == ["meta1", "meta2"]

    async def test_fetch_pedons_single_chunk(self):
        """Test fetch_pedons_by_bbox with single chunk (no chunking)."""
        # Mock client
        mock_client = AsyncMock(spec=SDAClient)

        # Mock site response
        site_response = AsyncMock(spec=SDAResponse)
        site_response.is_empty.return_value = False
        mock_site_df = AsyncMock()
        mock_site_df.__getitem__.return_value.unique.return_value.tolist.return_value = [
            "1001",
            "1002",
        ]
        site_response.to_pandas.return_value = mock_site_df

        # Mock horizons response
        horizons_response = AsyncMock(spec=SDAResponse)
        horizons_response.is_empty.return_value = False
        horizons_response.data = [
            {"layer_key": 1, "hzn_top": 0, "hzn_bot": 10, "pedon_key": "1001"},
        ]
        horizons_response.columns = ["layer_key", "hzn_top", "hzn_bot", "pedon_key"]
        horizons_response.metadata = ["meta1", "meta2"]

        mock_client.execute.side_effect = [site_response, horizons_response]

        # Call with large chunk_size to avoid chunking
        bbox = (-95.0, 40.0, -94.0, 41.0)
        result = await fetch_pedons_by_bbox(
            bbox, chunk_size=100, return_type="combined", client=mock_client
        )

        assert "site" in result
        assert "horizons" in result
        assert result["site"] == site_response

        # In single chunk case, it still reconstructs the response
        reconstructed_horizons = result["horizons"]
        assert not reconstructed_horizons.is_empty()
        assert len(reconstructed_horizons.data) == 1
        assert reconstructed_horizons.columns == [
            "layer_key",
            "hzn_top",
            "hzn_bot",
            "pedon_key",
        ]
        assert reconstructed_horizons.metadata == ["meta1", "meta2"]


class TestResponseCombining:
    """Test the _combine_responses function and helper functions."""

    def test_combine_empty_list_error(self):
        """Test that empty responses list raises error."""
        from soildb.fetch import _combine_responses

        with pytest.raises(FetchError):
            _combine_responses([])

    def test_combine_single_response(self):
        """Test that single response is returned as-is."""
        from soildb.fetch import _combine_responses

        mock_response = AsyncMock(spec=SDAResponse)
        result = _combine_responses([mock_response])

        assert result == mock_response

    def test_combine_two_responses(self):
        """Test combining two responses with different data."""
        from soildb.fetch import _combine_responses

        # Create mock responses with different data
        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        response1.is_empty.return_value = False

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]
        response2.data = [{"mukey": 2, "muname": "Unit 2"}]
        response2.is_empty.return_value = False

        # Don't set validation_result to avoid complications
        for r in [response1, response2]:
            if not hasattr(r, "validation_result"):
                r.validation_result = None

        combined = _combine_responses([response1, response2])

        assert combined is not None
        assert len(combined.data) == 2
        assert combined.data[0] == {"mukey": 1, "muname": "Unit 1"}
        assert combined.data[1] == {"mukey": 2, "muname": "Unit 2"}

    def test_combine_many_responses(self):
        """Test combining many responses."""
        from soildb.fetch import _combine_responses

        # Create 5 mock responses
        responses = []
        for i in range(5):
            response = AsyncMock(spec=SDAResponse)
            response.columns = ["mukey", "muname"]
            response.metadata = ["Int", "NVarChar"]
            response.data = [{"mukey": i + 1, "muname": f"Unit {i+1}"}]
            response.is_empty.return_value = False
            response.validation_result = None
            responses.append(response)

        combined = _combine_responses(responses)

        assert combined is not None
        assert len(combined.data) == 5

    def test_combine_with_empty_responses_skip(self):
        """Test that empty responses in the list are skipped."""
        from soildb.fetch import _combine_responses

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        response1.is_empty.return_value = False
        response1.validation_result = None

        # Empty response
        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]
        response2.data = []
        response2.is_empty.return_value = True
        response2.validation_result = None

        response3 = AsyncMock(spec=SDAResponse)
        response3.columns = ["mukey", "muname"]
        response3.metadata = ["Int", "NVarChar"]
        response3.data = [{"mukey": 3, "muname": "Unit 3"}]
        response3.is_empty.return_value = False
        response3.validation_result = None

        combined = _combine_responses([response1, response2, response3])

        assert combined is not None
        assert len(combined.data) == 2
        assert combined.data[0] == {"mukey": 1, "muname": "Unit 1"}
        assert combined.data[1] == {"mukey": 3, "muname": "Unit 3"}

    def test_combine_schema_mismatch_columns(self):
        """Test that schema mismatch on columns raises error."""
        from soildb.fetch import _combine_responses

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        response1.is_empty.return_value = False

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname", "clay"]  # Different columns!
        response2.metadata = ["Int", "NVarChar", "Float"]
        response2.data = [{"mukey": 2, "muname": "Unit 2", "clay": 25.5}]
        response2.is_empty.return_value = False

        with pytest.raises(FetchError, match="Schema mismatch"):
            _combine_responses([response1, response2])

    def test_combine_schema_mismatch_metadata(self):
        """Test that schema mismatch on metadata logs warning."""
        from soildb.fetch import _combine_responses

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        response1.is_empty.return_value = False
        response1.validation_result = None

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "Varchar"]  # Different metadata!
        response2.data = [{"mukey": 2, "muname": "Unit 2"}]
        response2.is_empty.return_value = False
        response2.validation_result = None

        # Should not raise, but log warning
        combined = _combine_responses([response1, response2])

        assert combined is not None

    def test_combine_with_deduplication(self):
        """Test combining responses with deduplication."""
        from soildb.fetch import _combine_responses

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [
            {"mukey": 1, "muname": "Unit 1"},
            {"mukey": 2, "muname": "Unit 2"},
        ]
        response1.is_empty.return_value = False
        response1.validation_result = None

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]
        response2.data = [
            {"mukey": 2, "muname": "Unit 2"},  # Duplicate!
            {"mukey": 3, "muname": "Unit 3"},
        ]
        response2.is_empty.return_value = False
        response2.validation_result = None

        # Combine without deduplication
        combined = _combine_responses([response1, response2], deduplicate=False)
        assert len(combined.data) == 4  # All rows kept

        # Combine with deduplication
        combined = _combine_responses([response1, response2], deduplicate=True)
        assert len(combined.data) == 3  # Duplicate removed

        # Check that first occurrence is kept
        mukeys = [row["mukey"] for row in combined.data]
        assert mukeys.count(2) == 1  # Only one instance of key 2

    def test_combine_preserves_column_order(self):
        """Test that combining preserves column order."""
        from soildb.fetch import _combine_responses

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname", "muacres"]
        response1.metadata = ["Int", "NVarChar", "Float"]
        response1.data = [
            {"mukey": 1, "muname": "Unit 1", "muacres": 1000.0},
        ]
        response1.is_empty.return_value = False
        response1.validation_result = None

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname", "muacres"]
        response2.metadata = ["Int", "NVarChar", "Float"]
        response2.data = [
            {"mukey": 2, "muname": "Unit 2", "muacres": 2000.0},
        ]
        response2.is_empty.return_value = False
        response2.validation_result = None

        combined = _combine_responses([response1, response2])

        # Check structure
        assert combined.columns == ["mukey", "muname", "muacres"]
        assert combined.metadata == ["Int", "NVarChar", "Float"]

    def test_combine_large_dataset(self):
        """Test combining responses with large datasets."""
        from soildb.fetch import _combine_responses

        # Create responses with many rows (simulating chunked fetches)
        responses = []
        for chunk_idx in range(3):
            response = AsyncMock(spec=SDAResponse)
            response.columns = ["mukey", "muname"]
            response.metadata = ["Int", "NVarChar"]

            # Each chunk has 1000 rows
            response.data = [
                {"mukey": chunk_idx * 1000 + i, "muname": f"Unit {chunk_idx * 1000 + i}"}
                for i in range(1000)
            ]
            response.is_empty.return_value = False
            response.validation_result = None
            responses.append(response)

        combined = _combine_responses(responses)

        assert len(combined.data) == 3000

    def test_validate_schema_consistency_pass(self):
        """Test that schema validation passes for consistent schemas."""
        from soildb.fetch import _validate_schema_consistency

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]

        # Should not raise
        _validate_schema_consistency([response1, response2])

    def test_validate_row_integrity_pass(self):
        """Test that row integrity validation passes for valid rows."""
        from soildb.fetch import _validate_row_integrity

        rows = [
            {"mukey": 1, "muname": "Unit 1"},
            {"mukey": 2, "muname": "Unit 2"},
        ]
        expected_columns = ["mukey", "muname"]

        # Should not raise
        _validate_row_integrity(rows, expected_columns)

    def test_validate_row_integrity_fail_column_count(self):
        """Test that row integrity validation fails for wrong column count."""
        from soildb.fetch import _validate_row_integrity

        rows = [
            {"mukey": 1, "muname": "Unit 1"},
            {"mukey": 2},  # Missing muname!
        ]
        expected_columns = ["mukey", "muname"]

        with pytest.raises(FetchError, match="Row .* has .* columns"):
            _validate_row_integrity(rows, expected_columns)

    def test_combine_logging(self, caplog):
        """Test that combining produces appropriate log messages."""
        import logging

        from soildb.fetch import _combine_responses

        logging.getLogger("soildb.fetch").setLevel(logging.DEBUG)

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [{"mukey": 1, "muname": "Unit 1"}]
        response1.is_empty.return_value = False
        response1.validation_result = None

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]
        response2.data = [{"mukey": 2, "muname": "Unit 2"}]
        response2.is_empty.return_value = False
        response2.validation_result = None

        _combine_responses([response1, response2])

        # Check that appropriate log messages were generated
        assert any("Combining" in record.message for record in caplog.records)

    def test_combine_deduplication_logging(self, caplog):
        """Test that deduplication is logged appropriately."""
        import logging

        from soildb.fetch import _combine_responses

        logging.getLogger("soildb.fetch").setLevel(logging.WARNING)

        response1 = AsyncMock(spec=SDAResponse)
        response1.columns = ["mukey", "muname"]
        response1.metadata = ["Int", "NVarChar"]
        response1.data = [
            {"mukey": 1, "muname": "Unit 1"},
            {"mukey": 2, "muname": "Unit 2"},
        ]
        response1.is_empty.return_value = False
        response1.validation_result = None

        response2 = AsyncMock(spec=SDAResponse)
        response2.columns = ["mukey", "muname"]
        response2.metadata = ["Int", "NVarChar"]
        response2.data = [
            {"mukey": 2, "muname": "Unit 2"},
            {"mukey": 3, "muname": "Unit 3"},
        ]
        response2.is_empty.return_value = False
        response2.validation_result = None

        _combine_responses([response1, response2], deduplicate=True)

        # Check that deduplication warning was logged
        assert any("Deduplication" in record.message for record in caplog.records)


# Integration tests (require network access)
@pytest.mark.integration
@pytest.mark.asyncio
class TestFetchIntegration:
    """Integration tests for fetch functions (require network access)."""

    @pytest.mark.timeout(20)
    async def test_fetch_real_mapunit_data(self):
        """Test fetching real map unit data."""
        # Use known good mukeys from California
        mukeys = [461994, 461995]  # CA630 mukeys

        async with SDAClient() as client:
            response = await fetch_by_keys(mukeys, "mapunit", client=client)
            df = response.to_pandas()

            assert not df.empty
            assert len(df) <= len(mukeys)  # Some keys might not exist
            assert "mukey" in df.columns
            assert "muname" in df.columns

    @pytest.mark.timeout(20)
    async def test_fetch_real_component_data(self):
        """Test fetching real component data."""
        # Use explicit client to avoid cleanup issues
        async with SDAClient() as client:
            # Get mukeys first, then components
            mukeys = await get_mukey_by_areasymbol(["CA630"], client)
            assert len(mukeys) > 0

            # Take first few mukeys to avoid large queries
            test_mukeys = mukeys[:5]

            response = await fetch_by_keys(test_mukeys, "component", "mukey", client=client)
            df = response.to_pandas()

            assert not df.empty
            assert "mukey" in df.columns
            assert "cokey" in df.columns
            assert "compname" in df.columns

    @pytest.mark.timeout(20)
    async def test_fetch_with_chunking(self):
        """Test that chunking works with real data."""
        async with SDAClient() as client:
            # Get enough mukeys to require chunking
            mukeys = await get_mukey_by_areasymbol(["CA630", "CA632"], client)

            if len(mukeys) > 5:
                # Use small chunk size to force chunking
                response = await fetch_by_keys(
                    mukeys[:10], "mapunit", chunk_size=3, client=client
                )
                df = response.to_pandas()

                assert not df.empty
                assert len(df) <= 10

    @pytest.mark.timeout(20)
    async def test_fetch_with_geometry(self):
        """Test fetching spatial data with geometry."""
        async with SDAClient() as client:
            mukeys = await get_mukey_by_areasymbol(["CA630"], client)
            test_mukeys = mukeys[:3]  # Small sample

            response = await fetch_by_keys(test_mukeys, "mupolygon", include_geometry=True, client=client)
            df = response.to_pandas()

            assert not df.empty
            assert "geometry" in df.columns
            # Check that geometry column contains WKT strings
            if len(df) > 0:
                geom_sample = df["geometry"].iloc[0]
                assert isinstance(geom_sample, str)
                assert any(
                    geom_type in geom_sample.upper()
                    for geom_type in ["POLYGON", "MULTIPOLYGON"]
                )


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
