"""
Tests for the ssurgo_tables metadata module.
"""

from soildb.ssurgo_tables import (
    DEFAULT_COLUMNS,
    FILTER_FIELDS,
    GEOMETRY_COLUMNS,
    KEY_COLUMNS,
    TABLE_ALIASES,
    default_columns,
    filter_fields,
    geometry_column,
    key_column,
)


class TestKeyColumns:
    """Test KEY_COLUMNS dictionary and key_column() function."""

    def test_key_columns_not_empty(self):
        """Test that KEY_COLUMNS is populated."""
        assert len(KEY_COLUMNS) > 0

    def test_core_table_keys(self):
        """Test key column for core SSURGO tables."""
        assert KEY_COLUMNS["legend"] == "lkey"
        assert KEY_COLUMNS["mapunit"] == "mukey"
        assert KEY_COLUMNS["component"] == "cokey"
        assert KEY_COLUMNS["chorizon"] == "chkey"

    def test_spatial_table_keys(self):
        """Test key column for spatial SSURGO tables."""
        assert KEY_COLUMNS["mupolygon"] == "mukey"
        assert KEY_COLUMNS["sapolygon"] == "areasymbol"
        assert KEY_COLUMNS["mupoint"] == "mukey"
        assert KEY_COLUMNS["muline"] == "mukey"

    def test_interpretation_table_keys(self):
        """Test key column for interpretation tables."""
        assert KEY_COLUMNS["cointerp"] == "cokey"
        assert KEY_COLUMNS["chinterp"] == "chkey"

    def test_administrative_table_keys(self):
        """Test key column for administrative tables."""
        assert KEY_COLUMNS["sacatalog"] == "areasymbol"
        assert KEY_COLUMNS["laoverlap"] == "lkey"
        assert KEY_COLUMNS["legendtext"] == "lkey"

    def test_key_column_known_table(self):
        """Test key_column() function with known tables."""
        assert key_column("mapunit") == "mukey"
        assert key_column("component") == "cokey"
        assert key_column("chorizon") == "chkey"

    def test_key_column_case_insensitive(self):
        """Test key_column() is case-insensitive."""
        assert key_column("MAPUNIT") == "mukey"
        assert key_column("Component") == "cokey"
        assert key_column("ChOrIzOn") == "chkey"

    def test_key_column_unknown_table(self):
        """Test key_column() returns None for unknown tables."""
        assert key_column("unknown_table") is None
        assert key_column("nonexistent") is None
        assert key_column("") is None


class TestGeometryColumns:
    """Test GEOMETRY_COLUMNS dictionary and geometry_column() function."""

    def test_geometry_columns_not_empty(self):
        """Test that GEOMETRY_COLUMNS is populated."""
        assert len(GEOMETRY_COLUMNS) > 0

    def test_polygon_tables(self):
        """Test geometry columns for polygon tables."""
        assert GEOMETRY_COLUMNS["mupolygon"] == "mupolygongeo"
        assert GEOMETRY_COLUMNS["sapolygon"] == "sapolygongeo"

    def test_point_tables(self):
        """Test geometry columns for point tables."""
        assert GEOMETRY_COLUMNS["mupoint"] == "mupointgeo"
        assert GEOMETRY_COLUMNS["featpoint"] == "featpointgeo"

    def test_line_tables(self):
        """Test geometry columns for line tables."""
        assert GEOMETRY_COLUMNS["muline"] == "mulinegeo"
        assert GEOMETRY_COLUMNS["featline"] == "featlinegeo"

    def test_geometry_column_known_table(self):
        """Test geometry_column() function with known spatial tables."""
        assert geometry_column("mupolygon") == "mupolygongeo"
        assert geometry_column("sapolygon") == "sapolygongeo"
        assert geometry_column("mupoint") == "mupointgeo"

    def test_geometry_column_case_insensitive(self):
        """Test geometry_column() is case-insensitive."""
        assert geometry_column("MUPOLYGON") == "mupolygongeo"
        assert geometry_column("Sapolygon") == "sapolygongeo"
        assert geometry_column("MuPoint") == "mupointgeo"

    def test_geometry_column_non_spatial_table(self):
        """Test geometry_column() returns None for non-spatial tables."""
        assert geometry_column("mapunit") is None
        assert geometry_column("component") is None
        assert geometry_column("chorizon") is None

    def test_geometry_column_unknown_table(self):
        """Test geometry_column() returns None for unknown tables."""
        assert geometry_column("unknown_table") is None
        assert geometry_column("nonexistent") is None
        assert geometry_column("") is None


class TestMetadataCompleteness:
    """Test that metadata covers expected tables."""

    def test_all_spatial_tables_have_keys(self):
        """Test that all spatial tables are in KEY_COLUMNS."""
        spatial_tables = GEOMETRY_COLUMNS.keys()
        for table in spatial_tables:
            assert table in KEY_COLUMNS, (
                f"Spatial table '{table}' missing from KEY_COLUMNS"
            )

    def test_all_geometry_tables_have_keys(self):
        """Test that all geometry tables are also in KEY_COLUMNS."""
        for table in GEOMETRY_COLUMNS:
            assert key_column(table) is not None, (
                f"Key missing for spatial table '{table}'"
            )

    def test_geometry_columns_count(self):
        """Test that GEOMETRY_COLUMNS has expected number of spatial tables."""
        # At least the documented spatial tables
        expected_spatial = {
            "mupolygon",
            "sapolygon",
            "mupoint",
            "muline",
            "featpoint",
            "featline",
        }
        for table in expected_spatial:
            assert table in GEOMETRY_COLUMNS


class TestTableAliases:
    """Test TABLE_ALIASES dictionary for spatial query construction."""

    def test_table_aliases_not_empty(self):
        """Test that TABLE_ALIASES is populated."""
        assert len(TABLE_ALIASES) > 0

    def test_spatial_table_aliases(self):
        """Test aliases for spatial tables."""
        assert TABLE_ALIASES["mupolygon"] == "p"
        assert TABLE_ALIASES["sapolygon"] == "s"
        assert TABLE_ALIASES["mupoint"] == "pt"
        assert TABLE_ALIASES["muline"] == "ln"
        assert TABLE_ALIASES["featpoint"] == "fp"
        assert TABLE_ALIASES["featline"] == "fl"

    def test_joined_table_aliases(self):
        """Test aliases for tables used in joins."""
        assert TABLE_ALIASES["mapunit"] == "m"
        assert TABLE_ALIASES["legend"] == "l"

    def test_aliases_are_strings(self):
        """Test that all aliases are non-empty strings."""
        for table, alias in TABLE_ALIASES.items():
            assert isinstance(alias, str) and len(alias) > 0, (
                f"Invalid alias for {table}: {alias}"
            )


class TestDefaultColumns:
    """Test DEFAULT_COLUMNS dictionary and default_columns() function."""

    def test_default_columns_not_empty(self):
        """Test that DEFAULT_COLUMNS is populated."""
        assert len(DEFAULT_COLUMNS) > 0

    def test_core_table_defaults(self):
        """Test default columns for core SSURGO tables."""
        assert DEFAULT_COLUMNS["mapunit"] == [
            "mukey",
            "musym",
            "muname",
            "mukind",
            "muacres",
        ]
        assert DEFAULT_COLUMNS["component"] == ["cokey", "mukey", "compname"]
        assert DEFAULT_COLUMNS["chorizon"] == ["chkey", "cokey", "hzname"]
        assert "lkey" in DEFAULT_COLUMNS["legend"]
        assert "areasymbol" in DEFAULT_COLUMNS["legend"]

    def test_spatial_table_defaults(self):
        """Test default columns for spatial SSURGO tables."""
        assert "mukey" in DEFAULT_COLUMNS["mupolygon"]
        assert "musym" in DEFAULT_COLUMNS["mupolygon"]
        assert "areasymbol" in DEFAULT_COLUMNS["mupolygon"]
        assert "areasymbol" in DEFAULT_COLUMNS["sapolygon"]
        assert "mukey" in DEFAULT_COLUMNS["mupoint"]
        assert "featkey" in DEFAULT_COLUMNS["featpoint"]

    def test_default_columns_are_lists(self):
        """Test that all default column sets are lists of strings."""
        for table, cols in DEFAULT_COLUMNS.items():
            assert isinstance(cols, list), f"DEFAULT_COLUMNS[{table}] should be a list"
            assert all(isinstance(col, str) for col in cols), (
                f"Invalid columns in {table}: {cols}"
            )
            assert len(cols) > 0, f"Empty column list for table {table}"

    def test_default_columns_function_known_table(self):
        """Test default_columns() function with known tables."""
        assert default_columns("mapunit") == [
            "mukey",
            "musym",
            "muname",
            "mukind",
            "muacres",
        ]
        assert default_columns("component") == ["cokey", "mukey", "compname"]
        assert default_columns("mupolygon") is not None
        assert "mukey" in default_columns("mupolygon")

    def test_default_columns_case_insensitive(self):
        """Test default_columns() is case-insensitive."""
        assert default_columns("MAPUNIT") == default_columns("mapunit")
        assert default_columns("Component") == default_columns("component")
        assert default_columns("MuPolygon") == default_columns("mupolygon")

    def test_default_columns_unknown_table(self):
        """Test default_columns() returns None for unknown tables."""
        assert default_columns("unknown_table") is None
        assert default_columns("nonexistent") is None
        assert default_columns("") is None

    def test_default_columns_returns_list(self):
        """Test that default_columns() returns a list (or None)."""
        result = default_columns("mapunit")
        assert isinstance(result, list)
        result = default_columns("unknown")
        assert result is None


class TestDefaultColumnsConsistency:
    """Test consistency between DEFAULT_COLUMNS and function behavior."""

    def test_all_tables_in_default_columns_have_entries(self):
        """Test that all tables in DEFAULT_COLUMNS are accessible via function."""
        for table in DEFAULT_COLUMNS.keys():
            result = default_columns(table)
            assert result is not None, (
                f"default_columns('{table}') returned None but table is in DEFAULT_COLUMNS"
            )
            assert result == DEFAULT_COLUMNS[table], f"Mismatch for table '{table}'"

    def test_default_columns_covers_documented_tables(self):
        """Test that DEFAULT_COLUMNS covers all documented SSURGO tables."""
        documented_tables = {
            "legend",
            "mapunit",
            "component",
            "chorizon",
            "mupolygon",
            "sapolygon",
            "mupoint",
            "muline",
            "featpoint",
            "featline",
        }
        for table in documented_tables:
            assert table in DEFAULT_COLUMNS, f"Table '{table}' not in DEFAULT_COLUMNS"
            assert default_columns(table) is not None, (
                f"default_columns('{table}') returned None"
            )


class TestFilterFields:
    """Test FILTER_FIELDS dictionary and filter_fields() function."""

    def test_filter_fields_not_empty(self):
        """Test that FILTER_FIELDS is populated."""
        assert len(FILTER_FIELDS) > 0

    def test_filter_fields_type(self):
        """Test that FILTER_FIELDS values are 3-tuples."""
        for table, fields in FILTER_FIELDS.items():
            assert isinstance(fields, tuple), (
                f"FILTER_FIELDS[{table}] should be a tuple, got {type(fields)}"
            )
            assert len(fields) == 3, (
                f"FILTER_FIELDS[{table}] should be a 3-tuple, got length {len(fields)}"
            )
            for i, field in enumerate(fields):
                assert field is None or isinstance(field, str), (
                    f"FILTER_FIELDS[{table}][{i}] should be str or None, got {type(field)}"
                )

    def test_filter_fields_match_default_columns_first_three(self):
        """Test that FILTER_FIELDS values match first 3 cols from DEFAULT_COLUMNS."""
        for table in DEFAULT_COLUMNS.keys():
            default_cols = DEFAULT_COLUMNS[table]
            # Build expected tuple: first 3 columns, padded with None to 3 elements
            expected = tuple(default_cols[:3]) + (None,) * (3 - len(default_cols[:3]))

            assert table in FILTER_FIELDS, (
                f"Table '{table}' in DEFAULT_COLUMNS but not in FILTER_FIELDS"
            )
            actual = FILTER_FIELDS[table]
            assert actual == expected, (
                f"FILTER_FIELDS['{table}'] = {actual}, expected {expected}"
            )

    def test_filter_fields_function_known_table(self):
        """Test filter_fields() function with known tables."""
        assert filter_fields("mapunit") == ("mukey", "musym", "muname")
        assert filter_fields("component") == ("cokey", "mukey", "compname")
        assert filter_fields("chorizon") == ("chkey", "cokey", "hzname")
        assert filter_fields("legend") == ("lkey", "areasymbol", "areaname")

    def test_filter_fields_function_with_padding(self):
        """Test filter_fields() function for tables with fewer than 3 columns."""
        # featpoint and featline have only 2 default columns
        result = filter_fields("featpoint")
        assert result == ("featkey", "featsym", None)
        result = filter_fields("featline")
        assert result == ("featkey", "featsym", None)

    def test_filter_fields_case_insensitive(self):
        """Test filter_fields() is case-insensitive."""
        assert filter_fields("MAPUNIT") == filter_fields("mapunit")
        assert filter_fields("Component") == filter_fields("component")
        assert filter_fields("FeatPoint") == filter_fields("featpoint")

    def test_filter_fields_unknown_table(self):
        """Test filter_fields() returns None for unknown tables."""
        assert filter_fields("unknown_table") is None
        assert filter_fields("nonexistent") is None
        assert filter_fields("") is None

    def test_all_default_columns_tables_have_filter_fields(self):
        """Test that every table in DEFAULT_COLUMNS has a FILTER_FIELDS entry."""
        for table in DEFAULT_COLUMNS.keys():
            result = filter_fields(table)
            assert result is not None, (
                f"filter_fields('{table}') returned None but table is in DEFAULT_COLUMNS"
            )
            assert isinstance(result, tuple) and len(result) == 3, (
                f"filter_fields('{table}') should return a 3-tuple"
            )
