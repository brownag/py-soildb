"""
Tests for query_templates module.

Verifies each public function:
- Emits correct SQL table in FROM clause
- Includes proper key predicates (WHERE, spatial, IN)
- Escapes single quotes correctly (' → '')
"""

from soildb.query_templates import (
    query_available_survey_areas,
    query_component_horizons_by_legend,
    query_components_at_point,
    query_components_by_legend,
    query_from_sql,
    query_mapunits_by_legend,
    query_mapunits_intersecting_bbox,
    query_pedon_by_pedon_key,
    query_pedon_horizons_by_pedon_keys,
    query_pedons_intersecting_bbox,
    query_spatial_by_legend,
    query_survey_area_boundaries,
)


class TestQueryTemplates:
    """Test all query_templates public functions."""

    def test_query_mapunits_by_legend(self):
        """Test query_mapunits_by_legend: mapunit table, legend join, areasymbol predicate, quote escaping."""
        query = query_mapunits_by_legend("IA109")
        sql = query.to_sql()

        # Table: FROM mapunit
        assert "FROM mapunit" in sql
        # Join: INNER JOIN legend
        assert "INNER JOIN legend" in sql
        # Key predicate: areasymbol
        assert "l.areasymbol = 'IA109'" in sql

        # Quote escaping: single quote → doubled
        query_escaped = query_mapunits_by_legend("O'Brien")
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped

    def test_query_components_by_legend(self):
        """Test query_components_by_legend: component table, mapunit+legend joins, areasymbol predicate, quote escaping."""
        query = query_components_by_legend("IA109")
        sql = query.to_sql()

        # Table: FROM component
        assert "FROM component" in sql
        # Joins: INNER JOIN mapunit and legend
        assert "INNER JOIN mapunit" in sql
        assert "INNER JOIN legend" in sql
        # Key predicate: areasymbol
        assert "l.areasymbol = 'IA109'" in sql

        # Quote escaping
        query_escaped = query_components_by_legend("O'Brien")
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped

    def test_query_component_horizons_by_legend(self):
        """Test query_component_horizons_by_legend: mapunit table, chorizon join, areasymbol+majcompflag predicates, quote escaping."""
        query = query_component_horizons_by_legend("IA109")
        sql = query.to_sql()

        # Table: FROM mapunit
        assert "FROM mapunit" in sql
        # Joins
        assert "INNER JOIN legend" in sql
        assert "INNER JOIN component" in sql
        assert "INNER JOIN chorizon" in sql
        # Key predicates
        assert "l.areasymbol = 'IA109'" in sql
        assert "c.majcompflag = 'Yes'" in sql

        # Quote escaping
        query_escaped = query_component_horizons_by_legend("O'Brien")
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped

    def test_query_components_at_point(self):
        """Test query_components_at_point: mupolygon table, spatial point filter, major component predicate."""
        query = query_components_at_point(-93.5, 42.5)
        sql = query.to_sql()

        # Table: FROM mupolygon
        assert "FROM mupolygon" in sql
        # Joins
        assert "INNER JOIN mapunit" in sql
        assert "INNER JOIN component" in sql
        assert "INNER JOIN chorizon" in sql
        # Spatial filter: coordinates appear in SQL
        assert "-93.5" in sql
        assert "42.5" in sql
        # Major component filter
        assert "c.majcompflag = 'Yes'" in sql

    def test_query_mapunits_intersecting_bbox(self):
        """Test query_mapunits_intersecting_bbox: mupolygon table, spatial bbox filter with coordinates."""
        query = query_mapunits_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
        sql = query.to_sql()

        # Table: FROM mupolygon
        assert "FROM mupolygon" in sql
        # Join: mapunit
        assert "INNER JOIN mapunit" in sql
        # Spatial filter: bbox coordinates
        assert "-94.0" in sql
        assert "42.0" in sql
        assert "-93.0" in sql
        assert "43.0" in sql

    def test_query_spatial_by_legend(self):
        """Test query_spatial_by_legend: mupolygon table, areasymbol predicate, quote escaping."""
        query = query_spatial_by_legend("IA109")
        sql = query.to_sql()

        # Table: FROM mupolygon
        assert "FROM mupolygon" in sql
        # Key predicate: areasymbol
        assert "areasymbol = 'IA109'" in sql

        # Quote escaping
        query_escaped = query_spatial_by_legend("O'Brien")
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped

    def test_query_available_survey_areas_default(self):
        """Test query_available_survey_areas (default table): sacatalog table."""
        query = query_available_survey_areas()
        sql = query.to_sql()

        # Table: FROM sacatalog (default)
        assert "FROM sacatalog" in sql
        # Default columns
        assert "areasymbol" in sql
        assert "areaname" in sql

    def test_query_available_survey_areas_custom_table(self):
        """Test query_available_survey_areas (custom table): alternate table parameter."""
        query = query_available_survey_areas(table="legend")
        sql = query.to_sql()

        # Table: FROM legend (custom)
        assert "FROM legend" in sql

    def test_query_survey_area_boundaries_default(self):
        """Test query_survey_area_boundaries (default table): sapolygon table."""
        query = query_survey_area_boundaries()
        sql = query.to_sql()

        # Table: FROM sapolygon (default)
        assert "FROM sapolygon" in sql
        # Default columns
        assert "areasymbol" in sql

    def test_query_survey_area_boundaries_custom_table(self):
        """Test query_survey_area_boundaries (custom table): alternate table parameter."""
        query = query_survey_area_boundaries(table="sa_boundary")
        sql = query.to_sql()

        # Table: FROM sa_boundary (custom)
        assert "FROM sa_boundary" in sql

    def test_query_from_sql(self):
        """Test query_from_sql: raw SQL passthrough unchanged."""
        raw_sql = "SELECT TOP 10 mukey, muname FROM mapunit"
        query = query_from_sql(raw_sql)
        sql = query.to_sql()

        # Raw SQL returned unchanged
        assert sql == raw_sql

    def test_query_pedons_intersecting_bbox(self):
        """Test query_pedons_intersecting_bbox: lab_combine_nasis_ncss table, bbox predicates."""
        query = query_pedons_intersecting_bbox(-94.0, 42.0, -93.0, 43.0)
        sql = query.to_sql()

        # Table: FROM lab_combine_nasis_ncss
        assert "FROM lab_combine_nasis_ncss" in sql
        # Bbox predicates: coordinates in WHERE clause
        assert "-94.0" in sql
        assert "42.0" in sql
        assert "-93.0" in sql
        assert "43.0" in sql
        # Nullable check
        assert "IS NOT NULL" in sql

    def test_query_pedon_horizons_by_pedon_keys(self):
        """Test query_pedon_horizons_by_pedon_keys: lab_layer table, IN clause with pedon_keys, quote escaping."""
        query = query_pedon_horizons_by_pedon_keys(["12345", "67890"])
        sql = query.to_sql()

        # Table: FROM lab_layer
        assert "FROM lab_layer" in sql
        # IN clause with keys
        assert "IN ('12345', '67890')" in sql
        # Layer type filter
        assert "layer_type = 'horizon'" in sql

        # Quote escaping: test with apostrophe in pedon key
        query_escaped = query_pedon_horizons_by_pedon_keys(["O'Brien", "Smith"])
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped

    def test_query_pedon_by_pedon_key(self):
        """Test query_pedon_by_pedon_key: lab_combine_nasis_ncss table, pedon_key predicate, quote escaping."""
        query = query_pedon_by_pedon_key("12345")
        sql = query.to_sql()

        # Table: FROM lab_combine_nasis_ncss
        assert "FROM lab_combine_nasis_ncss" in sql
        # Key predicate: pedon_key
        assert "p.pedon_key = '12345'" in sql

        # Quote escaping
        query_escaped = query_pedon_by_pedon_key("O'Brien")
        sql_escaped = query_escaped.to_sql()
        assert "O''Brien" in sql_escaped
