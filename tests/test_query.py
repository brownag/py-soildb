"""
Tests for query building functionality.
"""

from soildb.query import Query, QueryBuilder, SpatialQuery


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


class TestSpatialQuery:
    """Test the SpatialQuery builder class."""

    def test_bbox_intersection(self):
        query = (
            SpatialQuery()
            .select("mukey", "geometry")
            .from_("mupolygon")
            .intersects_bbox(-94.0, 42.0, -93.0, 43.0)
        )
        sql = query.to_sql()
        assert "STIntersects" in sql
        assert "POLYGON" in sql
        assert "-94.0 42.0" in sql

    def test_point_containment(self):
        query = (
            SpatialQuery()
            .select("mukey")
            .from_("mupolygon")
            .contains_point(-93.5, 42.5)
        )
        sql = query.to_sql()
        assert "STContains" in sql
        assert "POINT(-93.5 42.5)" in sql

    def test_spatial_with_other_conditions(self):
        query = (
            SpatialQuery()
            .select("mukey")
            .from_("mupolygon")
            .contains_point(-93.5, 42.5)
            .where("areasymbol = 'IA109'")
        )
        sql = query.to_sql()
        assert "STContains" in sql
        assert "areasymbol = 'IA109'" in sql


class TestQueryBuilder:
    """Test the QueryBuilder factory methods."""

    def test_mapunits_by_legend(self):
        query = QueryBuilder.mapunits_by_legend("IA109")
        sql = query.to_sql()
        assert "mukey" in sql
        assert "musym" in sql
        assert "muname" in sql
        assert "FROM mapunit m" in sql
        assert "areasymbol = 'IA109'" in sql

    def test_components_at_point(self):
        query = QueryBuilder.components_at_point(-93.5, 42.5)
        sql = query.to_sql()
        assert "STContains" in sql
        assert "POINT(-93.5 42.5)" in sql
        assert "majcompflag = 'Yes'" in sql

    def test_available_survey_areas(self):
        query = QueryBuilder.available_survey_areas()
        sql = query.to_sql()
        assert "FROM sacatalog" in sql
        assert "ORDER BY areasymbol" in sql
