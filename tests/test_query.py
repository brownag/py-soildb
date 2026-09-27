"""
Tests for query building functionality.
"""

import pytest

from soildb import ssurgo_tables
from soildb.query import ColumnSets, Query, eq_condition, in_condition


class TestQuery:
    """Test the Query builder class."""

    def test_basic_select(self):
        query = Query().select("mukey", "muname").from_("mapunit")
        sql = query.to_sql()
        assert "SELECT mukey, muname" in sql
        assert "FROM mapunit" in sql

    def test_where_condition(self):
        query = Query().select("mukey").from_("mapunit").where("areasymbol = 'IA109'")
        sql = query.to_sql()
        assert "WHERE areasymbol = 'IA109'" in sql

    def test_multiple_where_conditions(self):
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where("areasymbol = 'IA109'")
            .where("mukind = 'Consociation'")
        )
        sql = query.to_sql()
        assert "WHERE areasymbol = 'IA109' AND mukind = 'Consociation'" in sql

    def test_inner_join(self):
        query = (
            Query()
            .select("m.mukey", "c.compname")
            .from_("mapunit m")
            .inner_join("component c", "m.mukey = c.mukey")
        )
        sql = query.to_sql()
        assert "INNER JOIN component c ON m.mukey = c.mukey" in sql

    def test_limit(self):
        query = Query().select("mukey").from_("mapunit").limit(10)
        sql = query.to_sql()
        assert "SELECT TOP 10 mukey" in sql

    def test_order_by(self):
        query = Query().select("mukey").from_("mapunit").order_by("mukey", "DESC")
        sql = query.to_sql()
        assert "ORDER BY mukey DESC" in sql

    def test_raw_sql(self):
        raw = "SELECT COUNT(*) FROM mapunit"
        query = Query.from_sql(raw)
        assert query.to_sql() == raw


class TestInCondition:
    """Tests for in_condition() module-level function."""

    def test_in_condition_string_list(self):
        """Test basic string list in IN clause."""
        result = in_condition("musym", ["IA001", "IA002", "IA003"])
        assert result == "musym IN ('IA001', 'IA002', 'IA003')"

    def test_in_condition_string_with_single_quote(self):
        """Test string with single quote is escaped."""
        result = in_condition("musym", ["O'Brien", "Smith"])
        assert result == "musym IN ('O''Brien', 'Smith')"

    def test_in_condition_int_list(self):
        """Test integer list in IN clause."""
        result = in_condition("mukey", [100, 200, 300])
        assert result == "mukey IN (100, 200, 300)"

    def test_in_condition_float_list(self):
        """Test float list in IN clause."""
        result = in_condition("comppct_r", [25.5, 50.0, 75.25])
        assert result == "comppct_r IN (25.5, 50.0, 75.25)"

    def test_in_condition_mixed_types(self):
        """Test mixed int and string values."""
        result = in_condition("data_id", [1, "text_id", 3])
        assert result == "data_id IN (1, 'text_id', 3)"

    def test_in_condition_case_insensitive_string(self):
        """Test case_insensitive=True with string values."""
        result = in_condition("areasymbol", ["IA001", "IA002"], case_insensitive=True)
        assert result == "LOWER(areasymbol) IN (LOWER('IA001'), LOWER('IA002'))"

    def test_in_condition_case_insensitive_numeric(self):
        """Test case_insensitive=True with numeric values (not wrapped in LOWER)."""
        result = in_condition("mukey", [100, 200], case_insensitive=True)
        assert result == "LOWER(mukey) IN (100, 200)"

    def test_in_condition_case_insensitive_mixed(self):
        """Test case_insensitive with mixed types (numerics not wrapped)."""
        result = in_condition("data_id", [100, "text"], case_insensitive=True)
        assert result == "LOWER(data_id) IN (100, LOWER('text'))"

    def test_in_condition_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one value"):
            in_condition("mukey", [])

    def test_in_condition_bad_identifier_raises(self):
        """Test that invalid identifier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            in_condition("mukey; DROP TABLE", [1, 2, 3])

    def test_in_condition_invalid_value_type_raises(self):
        """Test that invalid value type raises ValueError."""
        with pytest.raises(ValueError, match="values must be str, int, or float"):
            in_condition("mukey", [1, None, 3])

    def test_in_condition_dotted_identifier(self):
        """Test that dotted identifier (alias.column) is accepted."""
        result = in_condition("m.mukey", [100, 200])
        assert result == "m.mukey IN (100, 200)"


class TestEqCondition:
    """Tests for eq_condition() module-level function."""

    def test_eq_condition_string(self):
        """Test basic string equality."""
        result = eq_condition("areasymbol", "IA109")
        assert result == "areasymbol = 'IA109'"

    def test_eq_condition_string_with_single_quote(self):
        """Test string with single quote is escaped."""
        result = eq_condition("musym", "O'Brien")
        assert result == "musym = 'O''Brien'"

    def test_eq_condition_integer(self):
        """Test integer equality."""
        result = eq_condition("mukey", 100)
        assert result == "mukey = 100"

    def test_eq_condition_float(self):
        """Test float equality."""
        result = eq_condition("comppct_r", 25.5)
        assert result == "comppct_r = 25.5"

    def test_eq_condition_case_insensitive_string(self):
        """Test case_insensitive=True with string value."""
        result = eq_condition("areasymbol", "IA001", case_insensitive=True)
        assert result == "LOWER(areasymbol) = LOWER('IA001')"

    def test_eq_condition_case_insensitive_numeric(self):
        """Test case_insensitive=True with numeric value (not wrapped in LOWER)."""
        result = eq_condition("mukey", 100, case_insensitive=True)
        assert result == "mukey = 100"

    def test_eq_condition_bad_identifier_raises(self):
        """Test that invalid identifier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            eq_condition("mukey; DROP TABLE", "value")

    def test_eq_condition_invalid_value_type_raises(self):
        """Test that invalid value type raises ValueError."""
        with pytest.raises(ValueError, match="value must be str, int, or float"):
            eq_condition("mukey", None)

    def test_eq_condition_dotted_identifier(self):
        """Test that dotted identifier (alias.column) is accepted."""
        result = eq_condition("m.mukey", 100)
        assert result == "m.mukey = 100"


class TestColumnSets:
    """Tests for ColumnSets metadata alignment."""

    def test_mapunit_basic_alignment_with_metadata(self):
        """Test that MAPUNIT_BASIC matches ssurgo_tables metadata."""
        assert ColumnSets.MAPUNIT_BASIC == ssurgo_tables.default_columns("mapunit")

    def test_legend_basic_pinned_value(self):
        """Test that LEGEND_BASIC retains its expected value."""
        assert ColumnSets.LEGEND_BASIC == [
            "lkey",
            "areasymbol",
            "areaname",
            "saversion",
        ]


class TestWhereIn:
    """Tests for Query.where_in() method."""

    def test_where_in_string_list(self):
        """Test basic string list in IN clause."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("musym", ["IA001", "IA002", "IA003"])
        )
        sql = query.to_sql()
        assert "WHERE musym IN ('IA001', 'IA002', 'IA003')" in sql

    def test_where_in_string_with_single_quote(self):
        """Test string with single quote is escaped."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("musym", ["O'Brien", "Smith"])
        )
        sql = query.to_sql()
        assert "musym IN ('O''Brien', 'Smith')" in sql

    def test_where_in_int_list(self):
        """Test integer list in IN clause."""
        query = (
            Query().select("mukey").from_("mapunit").where_in("mukey", [100, 200, 300])
        )
        sql = query.to_sql()
        assert "mukey IN (100, 200, 300)" in sql

    def test_where_in_float_list(self):
        """Test float list in IN clause."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("comppct_r", [25.5, 50.0, 75.25])
        )
        sql = query.to_sql()
        assert "comppct_r IN (25.5, 50.0, 75.25)" in sql

    def test_where_in_mixed_types(self):
        """Test mixed int and string values."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("data_id", [1, "text_id", 3])
        )
        sql = query.to_sql()
        assert "data_id IN (1, 'text_id', 3)" in sql

    def test_where_in_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one value"):
            Query().where_in("mukey", [])

    def test_where_in_bad_identifier_raises(self):
        """Test that invalid identifier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            Query().where_in("mukey; DROP TABLE", [1, 2, 3])

    def test_where_in_identifier_starts_with_number_raises(self):
        """Test that identifier starting with number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            Query().where_in("123invalid", [1, 2, 3])

    def test_where_in_dotted_identifier_accepted(self):
        """Test that dotted identifier (alias.column) is accepted."""
        query = (
            Query().select("mukey").from_("mapunit m").where_in("m.mukey", [100, 200])
        )
        sql = query.to_sql()
        assert "m.mukey IN (100, 200)" in sql

    def test_where_in_case_insensitive_string(self):
        """Test case_insensitive=True generates LOWER() wrapper."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("areasymbol", ["IA001", "IA002"], case_insensitive=True)
        )
        sql = query.to_sql()
        assert "LOWER(areasymbol) IN (LOWER('IA001'), LOWER('IA002'))" in sql

    def test_where_in_case_insensitive_mixed(self):
        """Test case_insensitive with mixed string and numeric values."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("mukey", [100, 200], case_insensitive=True)
        )
        sql = query.to_sql()
        assert "LOWER(mukey) IN (100, 200)" in sql

    def test_where_in_chaining_with_where(self):
        """Test that where_in chains with existing where() using AND."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where("mustatus = 'inactivated'")
            .where_in("mukey", [100, 200])
        )
        sql = query.to_sql()
        assert "WHERE mustatus = 'inactivated' AND mukey IN (100, 200)" in sql

    def test_where_in_multiple_chaining(self):
        """Test multiple where_in() calls chain correctly."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_in("mukey", [1, 2])
            .where_in("cokey", [10, 20])
        )
        sql = query.to_sql()
        assert "mukey IN (1, 2) AND cokey IN (10, 20)" in sql

    def test_where_in_single_value(self):
        """Test where_in with a single value."""
        query = Query().select("mukey").from_("mapunit").where_in("mukey", [42])
        sql = query.to_sql()
        assert "mukey IN (42)" in sql

    def test_where_in_invalid_value_type_raises(self):
        """Test that invalid value type raises ValueError."""
        with pytest.raises(ValueError, match="values must be str, int, or float"):
            Query().where_in("mukey", [1, None, 3])

    def test_where_in_empty_string_value(self):
        """Test that empty string value is handled correctly."""
        query = Query().select("mukey").from_("mapunit").where_in("musym", [""])
        sql = query.to_sql()
        assert "musym IN ('')" in sql


class TestWhereEq:
    """Tests for Query.where_eq() method."""

    def test_where_eq_string(self):
        """Test basic string equality."""
        query = Query().select("mukey").from_("mapunit").where_eq("areasymbol", "IA109")
        sql = query.to_sql()
        assert "WHERE areasymbol = 'IA109'" in sql

    def test_where_eq_string_with_single_quote(self):
        """Test string with single quote is escaped."""
        query = Query().select("mukey").from_("mapunit").where_eq("musym", "O'Brien")
        sql = query.to_sql()
        assert "musym = 'O''Brien'" in sql

    def test_where_eq_integer(self):
        """Test integer equality."""
        query = Query().select("mukey").from_("mapunit").where_eq("mukey", 100)
        sql = query.to_sql()
        assert "mukey = 100" in sql

    def test_where_eq_float(self):
        """Test float equality."""
        query = Query().select("mukey").from_("mapunit").where_eq("comppct_r", 25.5)
        sql = query.to_sql()
        assert "comppct_r = 25.5" in sql

    def test_where_eq_empty_string(self):
        """Test empty string value is handled correctly."""
        query = Query().select("mukey").from_("mapunit").where_eq("musym", "")
        sql = query.to_sql()
        assert "musym = ''" in sql

    def test_where_eq_bad_identifier_raises(self):
        """Test that invalid identifier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            Query().where_eq("mukey; DROP TABLE", "value")

    def test_where_eq_dotted_identifier_accepted(self):
        """Test that dotted identifier (alias.column) is accepted."""
        query = Query().select("mukey").from_("mapunit m").where_eq("m.mukey", 100)
        sql = query.to_sql()
        assert "m.mukey = 100" in sql

    def test_where_eq_case_insensitive_string(self):
        """Test case_insensitive=True with string value."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_eq("areasymbol", "IA001", case_insensitive=True)
        )
        sql = query.to_sql()
        assert "LOWER(areasymbol) = LOWER('IA001')" in sql

    def test_where_eq_case_insensitive_numeric(self):
        """Test case_insensitive=True with numeric value."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_eq("mukey", 100, case_insensitive=True)
        )
        sql = query.to_sql()
        assert "mukey = 100" in sql

    def test_where_eq_chaining_with_where(self):
        """Test that where_eq chains with existing where() using AND."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where("mustatus = 'inactivated'")
            .where_eq("mukey", 100)
        )
        sql = query.to_sql()
        assert "WHERE mustatus = 'inactivated' AND mukey = 100" in sql

    def test_where_eq_chaining_multiple(self):
        """Test multiple where_eq() calls chain correctly."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_eq("mukey", 100)
            .where_eq("cokey", 200)
        )
        sql = query.to_sql()
        assert "mukey = 100 AND cokey = 200" in sql

    def test_where_eq_invalid_value_type_raises(self):
        """Test that invalid value type raises ValueError."""
        with pytest.raises(ValueError, match="value must be str, int, or float"):
            Query().where_eq("mukey", None)

    def test_where_eq_mixed_with_where_in(self):
        """Test where_eq and where_in can be mixed."""
        query = (
            Query()
            .select("mukey")
            .from_("mapunit")
            .where_eq("mustatus", "inactivated")
            .where_in("mukey", [100, 200, 300])
        )
        sql = query.to_sql()
        assert "WHERE mustatus = 'inactivated' AND mukey IN (100, 200, 300)" in sql
