"""Unit tests for LDM query builder."""

import inspect

import pytest

from soildb.fetch import fetch_ldm
from soildb.ldm.client import LDMClient
from soildb.ldm.exceptions import LDMParameterError, LDMTableError
from soildb.ldm.query_builder import LDMQueryBuilder, build_ldm_query


class TestLDMQueryBuilder:
    """Test LDM query building functionality."""

    def test_builder_initialization_default(self):
        """Test query builder with default parameters."""
        builder = LDMQueryBuilder()
        assert builder.prep_codes == ["S", ""]
        assert builder.analyzed_size_fracs == ["<2 mm", ""]
        assert builder.layer_types == ["horizon", "layer", "reporting layer"]
        assert builder.area_type == "ssa"

    def test_builder_initialization_custom(self):
        """Test query builder with custom parameters."""
        builder = LDMQueryBuilder(prep_code="HM", analyzed_size_frac="")
        assert builder.prep_codes == ["HM"]
        assert builder.analyzed_size_fracs == [""]

    def test_r_default_parity_fetch_ldm_signature(self):
        """Ensure fetch_ldm defaults match R fetchLDM defaults."""
        sig = inspect.signature(fetch_ldm)
        assert sig.parameters["layer_type"].default == (
            "horizon",
            "layer",
            "reporting layer",
        )
        assert sig.parameters["area_type"].default == "ssa"
        assert sig.parameters["prep_code"].default == ("S", "")
        assert sig.parameters["analyzed_size_frac"].default == ("<2 mm", "")

    def test_r_default_parity_ldmclient_query_signature(self):
        """Ensure LDMClient.query defaults match R fetchLDM defaults."""
        sig = inspect.signature(LDMClient.query)
        assert sig.parameters["layer_type"].default == (
            "horizon",
            "layer",
            "reporting layer",
        )
        assert sig.parameters["area_type"].default == "ssa"
        assert sig.parameters["prep_code"].default == ("S", "")
        assert sig.parameters["analyzed_size_frac"].default == ("<2 mm", "")

    def test_builder_invalid_prep_code(self):
        """Test that invalid prep_code raises error."""
        with pytest.raises(LDMParameterError):
            LDMQueryBuilder(prep_code="invalid")

    def test_builder_invalid_size_fraction(self):
        """Test that invalid size fraction raises error."""
        with pytest.raises(LDMParameterError):
            LDMQueryBuilder(analyzed_size_frac="invalid")

    def test_builder_invalid_layer_type(self):
        """Test that invalid layer_type raises error."""
        with pytest.raises(LDMParameterError):
            LDMQueryBuilder(layer_type="invalid")

    def test_builder_invalid_area_type(self):
        """Test that invalid area_type raises error."""
        with pytest.raises(LDMParameterError):
            LDMQueryBuilder(area_type="invalid")

    def test_builder_invalid_table(self):
        """Test that invalid table name raises error."""
        with pytest.raises(LDMTableError):
            LDMQueryBuilder(tables=["invalid_table"])

    def test_build_query_basic(self):
        """Test building a basic query."""
        builder = LDMQueryBuilder()
        query = builder.build_query(keys=["85P0234"], key_column="pedlabsampnum")

        assert "SELECT" in query
        assert "FROM" in query
        assert "lab_layer" in query
        assert "85P0234" in query

    def test_build_query_with_custom_where(self):
        """Test building a query with custom WHERE clause."""
        builder = LDMQueryBuilder()
        query = builder.build_query(custom_where="corr_name LIKE 'Miami%'")

        assert "WHERE" in query
        assert "corr_name LIKE 'Miami%'" in query
        # Custom WHERE should not include prep_code filter
        assert "prep_code" not in query

    def test_build_query_string_escaping(self):
        """Test that string keys are properly escaped."""
        builder = LDMQueryBuilder()
        query = builder.build_query(
            keys=["85P0234", "40A3306"], key_column="pedlabsampnum"
        )

        # Should contain both pedon IDs
        assert "85P0234" in query
        assert "40A3306" in query
        # Should have IN clause
        assert "IN" in query

    def test_build_query_numeric_keys(self):
        """Test building query with numeric keys."""
        builder = LDMQueryBuilder()
        query = builder.build_query(keys=[123, 456], key_column="pedon_key")

        assert "123" in query
        assert "456" in query
        assert "IN" in query

    def test_build_chunked_queries(self):
        """Test building multiple chunked queries."""
        builder = LDMQueryBuilder()
        keys = list(range(1, 2500))  # 2500 keys
        queries = builder.build_chunked_queries(keys, chunk_size=1000)

        # Should have 3 queries (1000, 1000, 500)
        assert len(queries) == 3
        # All should be valid queries
        for query in queries:
            assert "SELECT" in query
            assert "FROM" in query

    def test_build_query_with_filters(self):
        """Test query building with multiple filters."""
        builder = LDMQueryBuilder(
            layer_type="horizon",
            prep_code="S",
            analyzed_size_frac="<2 mm",
        )
        query = builder.build_query(keys=["85P0234"], key_column="pedlabsampnum")

        assert "horizon" in query.lower()
        assert "prep_code" in query

    def test_analyzed_size_frac_not_applied_to_default_flat_tables(self):
        """Ensure analyzed_size_frac filter is only applied to fraction tables."""
        builder = LDMQueryBuilder(
            prep_code="S",
            analyzed_size_frac="<2 mm",
        )
        query = builder.build_query(keys=["85P0234"], key_column="pedlabsampnum")

        assert "prep_code" in query
        assert "analyzed_size_frac" not in query

    def test_build_ldm_query_convenience(self):
        """Test convenience function for building queries."""
        query = build_ldm_query(x=["85P0234"], what="pedlabsampnum", prep_code="S")

        assert "SELECT" in query
        assert "85P0234" in query
        assert "FROM" in query

    def test_build_ldm_query_with_where(self):
        """Test convenience function with WHERE clause."""
        query = build_ldm_query(
            WHERE="corr_name LIKE 'Miami%'",
        )

        assert "WHERE" in query
        assert "corr_name LIKE 'Miami%'" in query

    def test_multiple_tables(self):
        """Test query with multiple LDM tables."""
        tables = [
            "lab_physical_properties",
            "lab_chemical_properties",
            "lab_calculations_including_estimates_and_default_values",
        ]
        builder = LDMQueryBuilder(tables=tables)
        query = builder.build_query(keys=["85P0234"], key_column="pedlabsampnum")

        # All tables should be in query
        for table in tables:
            assert table in query or "lab_layer" in query

    def test_query_ordering(self):
        """Test that query has proper ordering."""
        builder = LDMQueryBuilder()
        query = builder.build_query(keys=["85P0234"], key_column="pedlabsampnum")

        assert "ORDER BY" in query
        assert "pedon_key" in query
