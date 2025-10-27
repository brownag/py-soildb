"""
Tests for query building functionality.
"""

from soildb.query import Query


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
