"""Unit tests for LDM table metadata and constants."""

import pytest

from soildb.ldm import tables


class TestLDMTables:
    """Test LDM table constants and metadata."""

    def test_default_tables_exist(self):
        """Test that default tables are defined."""
        assert tables.DEFAULT_TABLES is not None
        assert len(tables.DEFAULT_TABLES) > 0
        assert "lab_physical_properties" in tables.DEFAULT_TABLES

    def test_all_tables_include_default(self):
        """Test that ALL_TABLES includes DEFAULT_TABLES."""
        for table in tables.DEFAULT_TABLES:
            assert table in tables.ALL_TABLES

    def test_optional_tables_exist(self):
        """Test that optional tables are defined."""
        assert len(tables.OPTIONAL_TABLES) > 0

    def test_table_key_columns(self):
        """Test that key columns are mapped for all tables."""
        for table in tables.ALL_TABLES:
            if table in tables.TABLE_KEY_COLUMNS:
                assert tables.TABLE_KEY_COLUMNS[table] is not None

    def test_valid_prep_codes(self):
        """Test valid prep code values."""
        assert "S" in tables.PREP_CODES
        assert "D" in tables.PREP_CODES
        assert "" in tables.PREP_CODES

    def test_valid_size_fractions(self):
        """Test valid size fraction values."""
        assert "<2 mm" in tables.ANALYZED_SIZE_FRACTIONS
        assert ">2 mm" in tables.ANALYZED_SIZE_FRACTIONS

    def test_is_valid_table(self):
        """Test table validation."""
        assert tables.is_valid_table("lab_physical_properties") is True
        assert tables.is_valid_table("invalid_table") is False

    def test_is_valid_prep_code(self):
        """Test prep code validation."""
        assert tables.is_valid_prep_code("S") is True
        assert tables.is_valid_prep_code("invalid") is False

    def test_is_valid_size_fraction(self):
        """Test size fraction validation."""
        assert tables.is_valid_size_fraction("<2 mm") is True
        assert tables.is_valid_size_fraction("invalid") is False

    def test_is_valid_layer_type(self):
        """Test layer type validation."""
        assert tables.is_valid_layer_type("horizon") is True
        assert tables.is_valid_layer_type(None) is True
        assert tables.is_valid_layer_type("invalid") is False

    def test_is_valid_area_type(self):
        """Test area type validation."""
        assert tables.is_valid_area_type("state") is True
        assert tables.is_valid_area_type(None) is True
        assert tables.is_valid_area_type("invalid") is False

    def test_validate_tables(self):
        """Test table list validation."""
        valid_list = ["lab_physical_properties", "lab_chemical_properties"]
        assert tables.validate_tables(valid_list) is True

        invalid_list = ["invalid_table"]
        assert tables.validate_tables(invalid_list) is False

    def test_get_table_description(self):
        """Test table description retrieval."""
        desc = tables.get_table_description("lab_physical_properties")
        assert desc is not None
        assert len(desc) > 0

    def test_all_valid_prep_codes(self):
        """Test getting all valid prep codes."""
        codes = tables.all_valid_prep_codes()
        assert "S" in codes
        assert len(codes) > 0

    def test_all_valid_size_fractions(self):
        """Test getting all valid size fractions."""
        fracs = tables.all_valid_size_fractions()
        assert "<2 mm" in fracs
        assert len(fracs) > 0

    def test_all_valid_layer_types(self):
        """Test getting all valid layer types."""
        types = tables.all_valid_layer_types()
        assert "horizon" in types
        assert None in types

    def test_all_valid_area_types(self):
        """Test getting all valid area types."""
        types = tables.all_valid_area_types()
        assert "state" in types
        assert None in types
