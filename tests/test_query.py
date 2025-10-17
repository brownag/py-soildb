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

    def test_pedons_intersecting_bbox_basic(self):
        """Test basic pedons_intersecting_bbox functionality."""
        query = QueryBuilder.pedons_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
        sql = query.to_sql()
        assert "FROM lab_combine_nasis_ncss p" in sql
        assert (
            "p.latitude_decimal_degrees >= 42.0 AND p.latitude_decimal_degrees <= 43.0"
            in sql
        )
        assert (
            "p.longitude_decimal_degrees >= -94.0 AND p.longitude_decimal_degrees <= -93.0"
            in sql
        )
        assert "pedon_key" in sql
        assert "upedonid" in sql

    def test_pedons_intersecting_bbox_custom_columns(self):
        """Test pedons_intersecting_bbox with custom column names."""
        query = QueryBuilder.pedons_intersecting_bbox(
            -94.0, 42.0, -93.0, 43.0, lon_column="custom_lon", lat_column="custom_lat"
        )
        sql = query.to_sql()
        assert "p.custom_lat >= 42.0 AND p.custom_lat <= 43.0" in sql
        assert "p.custom_lon >= -94.0 AND p.custom_lon <= -93.0" in sql
        assert "p.custom_lat IS NOT NULL AND p.custom_lon IS NOT NULL" in sql

    def test_pedons_intersecting_bbox_custom_table(self):
        """Test pedons_intersecting_bbox with custom base table."""
        query = QueryBuilder.pedons_intersecting_bbox(
            -94.0, 42.0, -93.0, 43.0, base_table="custom_pedon_table"
        )
        sql = query.to_sql()
        assert "FROM custom_pedon_table p" in sql

    def test_pedons_intersecting_bbox_related_tables(self):
        """Test pedons_intersecting_bbox with related tables."""
        query = QueryBuilder.pedons_intersecting_bbox(
            -94.0,
            42.0,
            -93.0,
            43.0,
            related_tables=["lab_physical_properties", "lab_chemical_properties"],
        )
        sql = query.to_sql()
        assert (
            "LEFT JOIN lab_physical_properties t0 ON p.pedon_key = t0.pedon_key" in sql
        )
        assert (
            "LEFT JOIN lab_chemical_properties t1 ON p.pedon_key = t1.pedon_key" in sql
        )

    def test_pedons_intersecting_bbox_custom_columns_and_tables(self):
        """Test pedons_intersecting_bbox with all custom parameters."""
        query = QueryBuilder.pedons_intersecting_bbox(
            -94.0,
            42.0,
            -93.0,
            43.0,
            columns=["custom_col1", "custom_col2"],
            base_table="custom_table",
            related_tables=["related_table1"],
            lon_column="lon_col",
            lat_column="lat_col",
        )
        sql = query.to_sql()
        assert "SELECT custom_col1, custom_col2" in sql
        assert "FROM custom_table p" in sql
        assert "LEFT JOIN related_table1 t0 ON p.pedon_key = t0.pedon_key" in sql
        assert "p.lat_col >= 42.0 AND p.lat_col <= 43.0" in sql
        assert "p.lon_col >= -94.0 AND p.lon_col <= -93.0" in sql

    def test_pedon_by_pedon_key_basic(self):
        """Test basic pedon_by_pedon_key functionality."""
        query = QueryBuilder.pedon_by_pedon_key("12345")
        sql = query.to_sql()
        assert "FROM lab_combine_nasis_ncss p" in sql
        assert "p.pedon_key = '12345'" in sql
        assert "pedon_key" in sql
        assert "upedonid" in sql

    def test_pedon_by_pedon_key_custom_table(self):
        """Test pedon_by_pedon_key with custom base table."""
        query = QueryBuilder.pedon_by_pedon_key(
            "12345", base_table="custom_pedon_table"
        )
        sql = query.to_sql()
        assert "FROM custom_pedon_table p" in sql

    def test_pedon_by_pedon_key_related_tables(self):
        """Test pedon_by_pedon_key with related tables."""
        query = QueryBuilder.pedon_by_pedon_key(
            "12345",
            related_tables=["lab_physical_properties", "lab_chemical_properties"],
        )
        sql = query.to_sql()
        assert (
            "LEFT JOIN lab_physical_properties t0 ON p.pedon_key = t0.pedon_key" in sql
        )
        assert (
            "LEFT JOIN lab_chemical_properties t1 ON p.pedon_key = t1.pedon_key" in sql
        )

    def test_pedon_by_pedon_key_custom_columns_and_tables(self):
        """Test pedon_by_pedon_key with all custom parameters."""
        query = QueryBuilder.pedon_by_pedon_key(
            "12345",
            columns=["custom_col1", "custom_col2"],
            base_table="custom_table",
            related_tables=["related_table1", "related_table2"],
        )
        sql = query.to_sql()
        assert "SELECT custom_col1, custom_col2" in sql
        assert "FROM custom_table p" in sql
        assert "LEFT JOIN related_table1 t0 ON p.pedon_key = t0.pedon_key" in sql
        assert "LEFT JOIN related_table2 t1 ON p.pedon_key = t1.pedon_key" in sql

    def test_pedon_horizons_by_pedon_keys_basic(self):
        """Test basic pedon_horizons_by_pedon_keys functionality."""
        query = QueryBuilder.pedon_horizons_by_pedon_keys(["12345", "67890"])
        sql = query.to_sql()
        assert "FROM lab_layer l" in sql
        assert "l.pedon_key IN ('12345', '67890')" in sql
        assert "l.layer_type = 'horizon'" in sql
        assert "ORDER BY l.pedon_key, l.layer_sequence ASC" in sql

    def test_pedon_horizons_by_pedon_keys_custom_table(self):
        """Test pedon_horizons_by_pedon_keys with custom base table."""
        query = QueryBuilder.pedon_horizons_by_pedon_keys(
            ["12345"], base_table="custom_horizon_table"
        )
        sql = query.to_sql()
        assert "FROM custom_horizon_table l" in sql

    def test_pedon_horizons_by_pedon_keys_related_tables_lab(self):
        """Test pedon_horizons_by_pedon_keys with lab-related tables."""
        query = QueryBuilder.pedon_horizons_by_pedon_keys(
            ["12345"],
            related_tables=["lab_physical_properties", "lab_chemical_properties"],
        )
        sql = query.to_sql()
        # Lab tables should join on labsampnum
        assert (
            "LEFT JOIN lab_physical_properties t0 ON l.labsampnum = t0.labsampnum"
            in sql
        )
        assert (
            "LEFT JOIN lab_chemical_properties t1 ON l.labsampnum = t1.labsampnum"
            in sql
        )

    def test_pedon_horizons_by_pedon_keys_related_tables_non_lab(self):
        """Test pedon_horizons_by_pedon_keys with non-lab related tables."""
        query = QueryBuilder.pedon_horizons_by_pedon_keys(
            ["12345"], related_tables=["custom_table"]
        )
        sql = query.to_sql()
        # Non-lab tables should join on pedon_key
        assert "LEFT JOIN custom_table t0 ON l.pedon_key = t0.pedon_key" in sql

    def test_pedon_horizons_by_pedon_keys_custom_columns_and_tables(self):
        """Test pedon_horizons_by_pedon_keys with all custom parameters."""
        query = QueryBuilder.pedon_horizons_by_pedon_keys(
            ["12345"],
            columns=["custom_col1", "custom_col2"],
            base_table="custom_horizon_table",
            related_tables=["lab_physical_properties", "custom_table"],
        )
        sql = query.to_sql()
        assert "SELECT custom_col1, custom_col2" in sql
        assert "FROM custom_horizon_table l" in sql
        assert (
            "LEFT JOIN lab_physical_properties t0 ON l.labsampnum = t0.labsampnum"
            in sql
        )
        assert "LEFT JOIN custom_table t1 ON l.pedon_key = t1.pedon_key" in sql
