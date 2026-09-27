"""
Tests for the fetch module (key-based bulk data retrieval).
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

import pytest
import pytest_asyncio

from soildb.client import SDAClient
from soildb.fetch import (
    FetchError,
    fetch_by_keys,
    fetch_ldm,
    fetch_pedons_by_bbox,
    get_cokey_by_mukey,
    get_mukey_by_areasymbol,
)
from soildb.ldm import LDMClient
from soildb.response import SDAResponse


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
        mock_response = SDAResponse.from_rows(
            [[123456, "Test Unit"]],
            ["mukey", "muname"],
            ["int", "varchar"],
        )

        mock_client.execute.return_value = mock_response

        result = await fetch_by_keys([123456], "mapunit", client=mock_client)

        assert len(result.data) == 1
        assert result.data[0][0] == 123456
        mock_client.execute.assert_called_once()

    async def test_multiple_chunks(self):
        """Test fetch with keys requiring multiple chunks."""
        # Mock client and responses
        mock_client = AsyncMock(spec=SDAClient)
        mock_response1 = SDAResponse.from_rows(
            [[1, "Unit 1"]],
            ["mukey", "muname"],
            ["int", "varchar"],
        )
        mock_response2 = SDAResponse.from_rows(
            [[2, "Unit 2"]],
            ["mukey", "muname"],
            ["int", "varchar"],
        )

        mock_client.execute.side_effect = [mock_response1, mock_response2]

        #  use chunk_size=1 to force multiple chunks
        result = await fetch_by_keys(
            [1, 2], "mapunit", chunk_size=1, client=mock_client
        )

        assert len(result.data) == 2
        assert result.data[0][0] == 1
        assert result.data[1][0] == 2

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

    async def test_chunked_fetch_2500_keys(self):
        """Test that 2500 keys with chunk_size=1000 makes exactly 3 queries.

        This verifies the chunked fetching behavior: 2500 keys with chunk_size=1000
        should result in 3 chunks: [1000, 1000, 500], making 3 calls to execute.
        """
        # Create mock responses for each chunk
        mock_client = AsyncMock(spec=SDAClient)

        # Create 3 mock responses (one per chunk, each with 1 row)
        mock_responses = []
        for i in range(3):
            mock_resp = SDAResponse.from_rows(
                [[i, f"Unit {i}"]],
                ["mukey", "muname"],
                ["int", "varchar"],
            )
            mock_responses.append(mock_resp)

        # Set the side_effect to return one response per call
        mock_client.execute.side_effect = mock_responses

        # Create 2500 keys to ensure 3 chunks with chunk_size=1000
        keys = list(range(1, 2501))

        result = await fetch_by_keys(
            keys, "mapunit", chunk_size=1000, client=mock_client
        )

        # Verify that execute was called exactly 3 times
        assert mock_client.execute.call_count == 3
        # Verify that we got a result with 3 rows (one per chunk)
        assert len(result.data) == 3


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
            [123456], "component", "mukey", "cokey", client=ANY
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
        data_chunk_response.metadata = [
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=varchar",
        ]

        # Third chunk: has data
        data_chunk_response2 = AsyncMock(spec=SDAResponse)
        data_chunk_response2.is_empty.return_value = False
        data_chunk_response2.data = [
            {"layer_key": 3, "hzn_top": 0, "hzn_bot": 15, "pedon_key": "1004"},
        ]
        data_chunk_response2.columns = ["layer_key", "hzn_top", "hzn_bot", "pedon_key"]
        data_chunk_response2.metadata = [
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=varchar",
        ]

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
        # Metadata should have proper SDA format from concat
        assert len(horizons_response.metadata) == 4

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
        horizons_response.metadata = [
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=varchar",
        ]

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
        # Metadata should have proper SDA format from concat
        assert len(reconstructed_horizons.metadata) == 4


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

            response = await fetch_by_keys(
                test_mukeys, "component", "mukey", client=client
            )
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

            response = await fetch_by_keys(
                test_mukeys, "mupolygon", include_geometry=True, client=client
            )
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


@pytest_asyncio.fixture
async def temp_ldm_db(tmp_path) -> AsyncGenerator[Path, None]:
    """Create a minimal temp SQLite LDM database with core tables.

    Creates:
    - lab_combine_nasis_ncss: pedon_key (PK), pedlabsampnum, corr_name
    - lab_pedon: pedon_key (PK), upedonid, corr_name
    - lab_layer: lab_layer_key (PK), pedon_key (FK), layer_type
    - Property tables for filtering

    Yields:
        Path to the temporary SQLite database file
    """
    db_path = tmp_path / "test_ldm.db"

    # Create database and schema synchronously
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Create lab_combine_nasis_ncss table
        cursor.execute("""
            CREATE TABLE lab_combine_nasis_ncss (
                pedon_key INTEGER PRIMARY KEY,
                site_key INTEGER,
                pedlabsampnum TEXT,
                corr_name TEXT
            )
        """)

        # Create lab_pedon table
        cursor.execute("""
            CREATE TABLE lab_pedon (
                pedon_key INTEGER PRIMARY KEY,
                site_key INTEGER,
                upedonid TEXT,
                corr_name TEXT
            )
        """)

        # Create lab_layer table
        cursor.execute("""
            CREATE TABLE lab_layer (
                lab_layer_key INTEGER PRIMARY KEY,
                layer_key INTEGER,
                pedon_key INTEGER,
                labsampnum TEXT,
                layer_type TEXT
            )
        """)

        # Create default property tables (required for full query path)
        cursor.execute("""
            CREATE TABLE lab_physical_properties (
                labsampnum TEXT,
                pedon_key INTEGER,
                prep_code TEXT,
                analyzed_size_fraction TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE lab_chemical_properties (
                labsampnum TEXT,
                pedon_key INTEGER,
                prep_code TEXT,
                analyzed_size_fraction TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE lab_calculations_including_estimates_and_default_values (
                labsampnum TEXT,
                pedon_key INTEGER,
                prep_code TEXT,
                analyzed_size_fraction TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE lab_rosetta_Key (
                layer_key INTEGER,
                pedon_key INTEGER,
                prep_code TEXT,
                analyzed_size_fraction TEXT
            )
        """)

        # Insert test data into lab_combine_nasis_ncss
        test_data = [
            (1, 101, "S001", "Miami"),
            (2, 102, "S002", "Clarion"),
            (3, 103, "S003", "Mollisol"),
            (4, 104, "S004", "Vertisol"),
            (5, 105, "S005", "Alfisol"),
        ]

        cursor.executemany(
            """
            INSERT INTO lab_combine_nasis_ncss (pedon_key, site_key, pedlabsampnum, corr_name)
            VALUES (?, ?, ?, ?)
            """,
            test_data,
        )

        # Also insert into lab_pedon (without pedlabsampnum)
        pedon_data = [
            (1, 101, "P001", "Miami"),
            (2, 102, "1'0'2", "Clarion"),
            (3, 103, "P003", "Mollisol"),
            (4, 104, "P004", "Vertisol"),
            (5, 105, "P005", "Alfisol"),
        ]
        cursor.executemany(
            """
            INSERT INTO lab_pedon (pedon_key, site_key, upedonid, corr_name)
            VALUES (?, ?, ?, ?)
            """,
            pedon_data,
        )

        # Insert corresponding lab_layer records
        for pedon_key in range(1, 6):
            lab_layer_key = pedon_key * 10
            cursor.execute(
                """
                INSERT INTO lab_layer (lab_layer_key, layer_key, pedon_key, labsampnum, layer_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (lab_layer_key, pedon_key, pedon_key, f"S{pedon_key:03d}", "horizon"),
            )

        # Insert into property tables
        for pedon_key in range(1, 6):
            labsampnum = f"S{pedon_key:03d}"
            # lab_physical_properties
            cursor.execute(
                """
                INSERT INTO lab_physical_properties (labsampnum, pedon_key, prep_code, analyzed_size_fraction)
                VALUES (?, ?, ?, ?)
                """,
                (labsampnum, pedon_key, "S", "<2 mm"),
            )
            # lab_chemical_properties
            cursor.execute(
                """
                INSERT INTO lab_chemical_properties (labsampnum, pedon_key, prep_code, analyzed_size_fraction)
                VALUES (?, ?, ?, ?)
                """,
                (labsampnum, pedon_key, "S", "<2 mm"),
            )
            # lab_calculations_including_estimates_and_default_values
            cursor.execute(
                """
                INSERT INTO lab_calculations_including_estimates_and_default_values (labsampnum, pedon_key, prep_code, analyzed_size_fraction)
                VALUES (?, ?, ?, ?)
                """,
                (labsampnum, pedon_key, "S", "<2 mm"),
            )

        conn.commit()
        yield db_path
    finally:
        conn.close()


@pytest.mark.asyncio
class TestFetchLDM:
    """Test the fetch_ldm function with SQLite database backend."""

    async def test_fetch_ldm_with_dsn_creates_no_sdaclient(self, temp_ldm_db):
        """Test that decorator creates no SDAClient when dsn is provided."""
        with patch.object(SDAClient, "__init__", return_value=None) as mock_init:
            # Call fetch_ldm with dsn parameter
            response = await fetch_ldm(x=1, what="pedlabsampnum", dsn=temp_ldm_db)

            # Verify SDAClient.__init__ was never called
            # (decorator creates nothing for Optional[Union[...]] annotation)
            mock_init.assert_not_called()

            # Verify response is returned (even if empty, since test data is minimal)
            assert response is not None

    async def test_fetch_ldm_with_dsn_returns_response(self, temp_ldm_db):
        """Test that fetch_ldm with dsn returns a response object."""
        response = await fetch_ldm(x=1, what="pedlabsampnum", dsn=temp_ldm_db)

        # Verify response is returned
        assert response is not None
        assert isinstance(response, SDAResponse)

    async def test_fetch_ldm_with_explicit_ldm_client_does_not_close(self, temp_ldm_db):
        """Test that passing an explicit LDMClient uses it directly without closing."""
        client = LDMClient(dsn=temp_ldm_db)
        with patch.object(client, "close", wraps=client.close) as mock_close:
            response = await fetch_ldm(
                x=1, what="pedlabsampnum", dsn=temp_ldm_db, client=client
            )
            assert isinstance(response, SDAResponse)
            mock_close.assert_not_called()
        await client.close()

    async def test_fetch_ldm_with_sda_client_wraps_without_closing_caller_client(
        self, temp_ldm_db
    ):
        """Test that passing an SDAClient wraps it and does not close caller's client."""
        sda_client = AsyncMock(spec=SDAClient)
        mock_response = SDAResponse.from_rows([], ["col"], ["varchar"])
        sda_client.execute.return_value = mock_response

        orig_init = LDMClient.__init__
        init_calls = []

        def tracking_init(self_obj, *args, **kwargs):
            init_calls.append((args, kwargs))
            return orig_init(self_obj, *args, **kwargs)

        with patch.object(LDMClient, "__init__", tracking_init):
            response = await fetch_ldm(
                x=1, what="pedlabsampnum", dsn=temp_ldm_db, client=sda_client
            )
            assert isinstance(response, SDAResponse)
            assert len(init_calls) == 1
            assert init_calls[0][1].get("sda_client") is sda_client
            sda_client.close.assert_not_called()

    async def test_fetch_ldm_client_none_uses_context_manager_and_closes(
        self, temp_ldm_db
    ):
        """Test that fetch_ldm with client=None uses async context manager and closes."""
        close_called = []
        original_aexit = LDMClient.__aexit__

        async def tracking_aexit(self_obj, exc_type, exc_val, exc_tb):
            close_called.append("closed")
            return await original_aexit(self_obj, exc_type, exc_val, exc_tb)

        with patch.object(LDMClient, "__aexit__", tracking_aexit):
            response = await fetch_ldm(x=1, what="pedlabsampnum", dsn=temp_ldm_db)
            assert isinstance(response, SDAResponse)
            assert len(close_called) == 1


class TestFetchLDMSync:
    """Test synchronous execution of fetch_ldm via .sync."""

    def test_fetch_ldm_sync_execution(self, temp_ldm_db):
        """Test that fetch_ldm.sync(...) executes synchronously and returns SDAResponse."""
        response = fetch_ldm.sync(x=1, what="pedlabsampnum", dsn=temp_ldm_db)
        assert response is not None
        assert isinstance(response, SDAResponse)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
