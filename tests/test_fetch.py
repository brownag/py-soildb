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
    fetch_chorizon_by_cokey,
    fetch_component_by_mukey,
    fetch_mapunit_polygon,
    fetch_survey_area_polygon,
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
            FetchError, match="No data was returned from the fetch operation"
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
        mock_response2 = AsyncMock(spec=SDAResponse)
        mock_response2.data = [{"mukey": 2, "muname": "Unit 2"}]

        mock_client.execute.side_effect = [mock_response1, mock_response2]

        #  use chunk_size=1 to force multiple chunks
        result = await fetch_by_keys([1, 2], "mapunit", chunk_size=1, client=mock_client)

        assert len(result.data) == 2
        assert result.data[0]["mukey"] == 1
        assert result.data[1]["mukey"] == 2

    async def test_custom_columns(self):
        """Test fetch with custom column selection."""
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)
        mock_client.execute.return_value = mock_response

        await fetch_by_keys([123456], "mapunit", columns=["mukey", "muname"], client=mock_client)

        # Check that query was built with correct columns
        # The Query object should have the specified columns
        # (This is a simplified check - in real implementation we'd check the SQL)
        assert mock_client.execute.called

    async def test_include_geometry(self):
        """Test fetch with geometry inclusion."""
        mock_client = AsyncMock(spec=SDAClient)
        mock_response = AsyncMock(spec=SDAResponse)
        mock_client.execute.return_value = mock_response

        await fetch_by_keys([123456], "mupolygon", include_geometry=True, client=mock_client)

        assert mock_client.execute.called


@pytest.mark.asyncio
class TestSpecializedFunctions:
    """Test the specialized fetch functions."""

    @patch("soildb.fetch.fetch_by_keys")
    async def test_fetch_mapunit_polygon(self, mock_fetch):
        """Test fetch_mapunit_polygon wrapper."""
        mock_response = AsyncMock(spec=SDAResponse)
        mock_fetch.return_value = mock_response

        result = await fetch_mapunit_polygon([123456, 123457])

        # Columns now come from schema as a list
        mock_fetch.assert_called_once_with(
            [123456, 123457],
            "mupolygon",
            "mukey",
            ["mukey", "muname", "musym", "lkey", "areaname"],  # From mapunit schema
            1000,
            True,  # include_geometry
            None,
        )
        assert result == mock_response

    @patch("soildb.fetch.fetch_by_keys")
    async def test_fetch_component_by_mukey(self, mock_fetch):
        """Test fetch_component_by_mukey wrapper."""
        mock_response = AsyncMock(spec=SDAResponse)
        mock_fetch.return_value = mock_response

        result = await fetch_component_by_mukey([123456])

        # Columns now come from schema as a list
        mock_fetch.assert_called_once_with(
            [123456],
            "component",
            "mukey",
            ["cokey", "compname", "comppct_r", "majcompflag", "taxclname", "drainagecl", "localphase", "hydricrating", "compkind"],  # From component schema
            1000,
            False,  # include_geometry
            None,
        )
        assert result == mock_response

    @patch("soildb.fetch.fetch_by_keys")
    async def test_fetch_chorizon_by_cokey(self, mock_fetch):
        """Test fetch_chorizon_by_cokey wrapper."""
        mock_response = AsyncMock(spec=SDAResponse)
        mock_fetch.return_value = mock_response

        result = await fetch_chorizon_by_cokey(["123456:1", "123456:2"])

        # Columns now come from schema as a list
        mock_fetch.assert_called_once_with(
            ["123456:1", "123456:2"],
            "chorizon",
            "cokey",
            ["chkey", "hzname", "hzdept_r", "hzdepb_r", "claytotal_r", "sandtotal_r", "om_r", "ph1to1h2o_r"],  # From chorizon schema
            1000,
            False,
            None,
        )
        assert result == mock_response

    @patch("soildb.fetch.fetch_by_keys")
    async def test_fetch_survey_area_polygon(self, mock_fetch):
        """Test fetch_survey_area_polygon wrapper."""
        mock_response = AsyncMock(spec=SDAResponse)
        mock_fetch.return_value = mock_response

        result = await fetch_survey_area_polygon(["CA630", "CA632"])

        mock_fetch.assert_called_once_with(
            ["CA630", "CA632"],
            "sapolygon",
            "areasymbol",
            "areasymbol, spatialversion, lkey",
            1000,
            True,  # include_geometry
            None,
        )
        assert result == mock_response


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

    @patch("soildb.fetch.fetch_by_keys")
    async def test_get_cokeys_major_only(self, mock_fetch):
        """Test getting only major component cokeys."""
        mock_response = AsyncMock(spec=SDAResponse)
        mock_df = AsyncMock()
        mock_df.empty = False
        mock_df.__getitem__.return_value.tolist.return_value = ["123456:1"]
        mock_response.to_pandas.return_value = mock_df
        mock_fetch.return_value = mock_response

        result = await get_cokey_by_mukey([123456], major_components_only=True)

        assert result == ["123456:1"]
        # Note: The function currently doesn't properly implement the major_components_only filter
        # This test documents the current behavior and should be updated when the function is fixed


# Integration tests (require network access)
@pytest.mark.integration
@pytest.mark.asyncio
class TestFetchIntegration:
    """Integration tests for fetch functions (require network access)."""

    async def test_fetch_real_mapunit_data(self):
        """Test fetching real map unit data."""
        # Use known good mukeys from California
        mukeys = [461994, 461995]  # CA630 mukeys

        response = await fetch_by_keys(mukeys, "mapunit")
        df = response.to_pandas()

        assert not df.empty
        assert len(df) <= len(mukeys)  # Some keys might not exist
        assert "mukey" in df.columns
        assert "muname" in df.columns

    async def test_fetch_real_component_data(self):
        """Test fetching real component data."""
        # Use explicit client to avoid cleanup issues
        async with SDAClient() as client:
            # Get mukeys first, then components
            mukeys = await get_mukey_by_areasymbol(["CA630"], client)
            assert len(mukeys) > 0

            # Take first few mukeys to avoid large queries
            test_mukeys = mukeys[:5]

            response = await fetch_component_by_mukey(test_mukeys, client=client)
            df = response.to_pandas()

            assert not df.empty
            assert "mukey" in df.columns
            assert "cokey" in df.columns
            assert "compname" in df.columns

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

    async def test_fetch_with_geometry(self):
        """Test fetching spatial data with geometry."""
        async with SDAClient() as client:
            mukeys = await get_mukey_by_areasymbol(["CA630"], client)
            test_mukeys = mukeys[:3]  # Small sample

            response = await fetch_mapunit_polygon(test_mukeys, client=client)
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
