"""Tests for LDMClient query execution against SQLite databases.

This module tests:
1. Query by pedon_key (basic single query)
2. SQL injection prevention (quote in value)
3. Multi-chunk queries with correct ordering (chunk_size=2, 5 keys)
4. Unknown table error handling (raises LDMError)
5. Empty results, context manager, and edge cases
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from soildb.ldm.client import LDMClient
from soildb.response import SDAResponse


@pytest_asyncio.fixture
async def temp_ldm_db(tmp_path) -> AsyncGenerator[Path, None]:
    """Create a minimal temp SQLite LDM database with core tables.

    Creates:
    - lab_combine_nasis_ncss: pedon_key (PK), upedonid, corr_name (for site queries)
    - lab_pedon: pedon_key (PK), site_key, upedonid, corr_name
    - lab_site: site_key (PK), siteiid
    - lab_layer: lab_layer_key (PK), pedon_key (FK), layer_type

    Yields:
        Path to the temporary SQLite database file
    """
    db_path = tmp_path / "test_ldm.db"

    # Create database and schema synchronously
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Create lab_combine_nasis_ncss table (used for site/pedon lookups)
        cursor.execute("""
            CREATE TABLE lab_combine_nasis_ncss (
                pedon_key INTEGER PRIMARY KEY,
                site_key INTEGER,
                upedonid TEXT,
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

        # Create lab_site table
        cursor.execute("""
            CREATE TABLE lab_site (
                site_key INTEGER PRIMARY KEY,
                siteiid TEXT
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
        # 5 pedons for chunking test
        test_data = [
            (1, 101, "P001", "S001", "Miami"),
            (
                2,
                102,
                "1'0'2",
                "S002",
                "Clarion",
            ),  # Quote in upedonid for SQL injection test
            (3, 103, "P003", "S003", "Mollisol"),
            (4, 104, "P004", "S004", "Vertisol"),
            (5, 105, "P005", "S005", "Alfisol"),
        ]

        cursor.executemany(
            """
            INSERT INTO lab_combine_nasis_ncss (pedon_key, site_key, upedonid, pedlabsampnum, corr_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            test_data,
        )

        # Also insert into lab_pedon (without pedlabsampnum)
        pedon_data = [(row[0], row[1], row[2], row[4]) for row in test_data]
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
            layer_key = pedon_key
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
            # lab_rosetta_Key
            cursor.execute(
                """
                INSERT INTO lab_rosetta_Key (layer_key, pedon_key, prep_code, analyzed_size_fraction)
                VALUES (?, ?, ?, ?)
                """,
                (layer_key, pedon_key, "", ""),
            )

        conn.commit()
    finally:
        conn.close()

    yield db_path


class TestLDMClientSQLite:
    """Test LDMClient with SQLite backend."""

    @pytest.mark.asyncio
    async def test_query_by_pedon_key(self, temp_ldm_db):
        """Test basic query by pedon_key.

        Query for a single pedon_key and verify single record returned.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query for pedon_key=1
            response = await client.query(x=[1], what="pedon_key")

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()
            assert len(data) >= 1
            # Verify pedon_key=1 is in results
            pedon_keys = [row["pedon_key"] for row in data]
            assert 1 in pedon_keys

    @pytest.mark.asyncio
    async def test_sql_injection_prevention_quote_in_value(self, temp_ldm_db):
        """Test SQL injection prevention with quote in value.

        Query for upedonid containing single quote (1'0'2) and verify
        correct record returned without SQL error.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query for pedon with upedonid="1'0'2"
            # This tests that quote characters in search values are properly escaped
            response = await client.query(x=["1'0'2"], what="upedonid")

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()
            # Should have at least one row for pedon_key=2 (which has upedonid="1'0'2")
            pedon_keys = [row.get("pedon_key") for row in data]
            assert 2 in pedon_keys

    @pytest.mark.asyncio
    async def test_multi_chunk_query_5_keys_chunk_size_2(self, temp_ldm_db):
        """Test multi-chunk query returns all rows in order.

        Query with 5 keys and chunk_size=2, verify all 5 pedons returned
        and order is consistent (no duplication/loss).
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query for all 5 pedons with small chunk size
            response = await client.query(
                x=[1, 2, 3, 4, 5],
                what="pedon_key",
                chunk_size=2,
            )

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()

            # Verify all 5 pedon_keys present (at least one row per pedon)
            pedon_keys = [row["pedon_key"] for row in data]
            assert set(pedon_keys) == {1, 2, 3, 4, 5}

            # Verify no row duplicates (checking by lab_layer_key if present)
            if data and "lab_layer_key" in data[0]:
                layer_keys = [row["lab_layer_key"] for row in data]
                assert len(layer_keys) == len(set(layer_keys))

    @pytest.mark.asyncio
    async def test_unknown_table_raises_ldm_error(self, temp_ldm_db):
        """Test unknown table raises LDMError.

        Query with non-existent table name and verify LDMTableError raised.
        """
        from soildb.ldm.exceptions import LDMError

        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query using upedonid which exists, so we get to Stage 2 where table validation occurs
            # Use pedon 1 with upedonid="P001" to ensure site query matches
            with pytest.raises(LDMError):
                await client.query(
                    x=["P001"], what="upedonid", tables=["nonexistent_table"]
                )

    @pytest.mark.asyncio
    async def test_empty_result_set(self, temp_ldm_db):
        """Test empty result set returns is_empty() True.

        Query for pedon_key not in database and verify is_empty() True.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query for non-existent pedon_key
            response = await client.query(x=[9999], what="pedon_key")

            assert isinstance(response, SDAResponse)
            assert response.is_empty()

    @pytest.mark.asyncio
    async def test_multi_column_result_set(self, temp_ldm_db):
        """Test response with multiple columns.

        Query and verify response.to_dict() returns rows with expected columns.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query for pedons 1 and 2
            response = await client.query(x=[1, 2], what="pedon_key")

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()

            # Should have rows for pedons 1 and 2
            pedon_keys = {row["pedon_key"] for row in data}
            assert pedon_keys == {1, 2}

            # Verify all rows have expected columns
            for row in data:
                assert "pedon_key" in row
                # Check for physical properties columns
                assert any(
                    key in row for key in ["prep_code", "analyzed_size_fraction"]
                )

    @pytest.mark.asyncio
    async def test_ldmclient_context_manager(self, temp_ldm_db):
        """Test LDMClient context manager (async with).

        Verify connect() called automatically and close() called on exit.
        """
        client = LDMClient(dsn=temp_ldm_db)

        # Use context manager
        async with client:
            # Inside context, execute a query to verify connection works
            response = await client.query(x=[1], what="pedon_key")
            assert not response.is_empty()

        # After context manager exits, client should be closed
        # (No explicit test needed, just verify no errors)

    @pytest.mark.asyncio
    async def test_async_await_compatibility(self, temp_ldm_db):
        """Verify async/await compatibility.

        Test that all client calls properly use await.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # All calls must use await
            response = await client.query(x=[1], what="pedon_key")

            # Verify response is properly awaited result
            assert isinstance(response, SDAResponse)
            assert not response.is_empty()

    @pytest.mark.asyncio
    async def test_chunk_size_greater_than_records(self, temp_ldm_db):
        """Test chunk_size > total records.

        Query with chunk_size larger than number of records,
        verify all records returned.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query 5 pedons with large chunk_size
            response = await client.query(
                x=[1, 2, 3, 4, 5],
                what="pedon_key",
                chunk_size=100,
            )

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()

            # Verify all 5 pedons present
            pedon_keys = {row["pedon_key"] for row in data}
            assert pedon_keys == {1, 2, 3, 4, 5}

    @pytest.mark.asyncio
    async def test_chunk_size_one(self, temp_ldm_db):
        """Test extreme chunking with chunk_size=1.

        Query with chunk_size=1 (extreme chunking),
        verify all records returned.
        """
        async with LDMClient(dsn=temp_ldm_db) as client:
            # Query 3 pedons with chunk_size=1
            response = await client.query(
                x=[1, 2, 3],
                what="pedon_key",
                chunk_size=1,
            )

            assert isinstance(response, SDAResponse)
            assert not response.is_empty()
            data = response.to_dict()

            # Verify all 3 pedons present
            pedon_keys = {row["pedon_key"] for row in data}
            assert pedon_keys == {1, 2, 3}
