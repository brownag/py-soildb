"""Tests for public API exports."""

import pytest


def test_ssurgo_tier1_functions_exported():
    """Tier 1 (primary) SSURGO functions should be exported."""
    import soildb

    assert hasattr(soildb, "fetch_by_keys"), "fetch_by_keys not exported"
    assert hasattr(soildb, "QueryPresets"), "QueryPresets not exported"


def test_ssurgo_tier2_deprecated_functions_exported():
    """Tier 2 (deprecated wrapper) functions should be exported."""
    import soildb

    assert hasattr(soildb, "fetch_mapunit_polygon"), \
        "fetch_mapunit_polygon not exported"
    assert hasattr(soildb, "fetch_component_by_mukey"), \
        "fetch_component_by_mukey not exported"
    assert hasattr(soildb, "fetch_chorizon_by_cokey"), \
        "fetch_chorizon_by_cokey not exported"
    assert hasattr(soildb, "fetch_survey_area_polygon"), \
        "fetch_survey_area_polygon not exported"


def test_ssurgo_tier3_functions_exported():
    """Tier 3 (complex workflows) functions should be exported."""
    import soildb

    assert hasattr(soildb, "fetch_ldm"), "fetch_ldm not exported"
    assert hasattr(soildb, "fetch_pedons_by_bbox"), "fetch_pedons_by_bbox not exported"
    assert hasattr(soildb, "fetch_pedon_horizons"), "fetch_pedon_horizons not exported"


def test_ssurgo_tier4_functions_exported():
    """Tier 4 (helper) functions should be exported."""
    import soildb

    assert hasattr(soildb, "get_mukey_by_areasymbol"), "get_mukey_by_areasymbol not exported"
    assert hasattr(soildb, "get_cokey_by_mukey"), "get_cokey_by_mukey not exported"


def test_spatial_functions_exported():
    """Spatial query functions should be exported."""
    import soildb

    assert hasattr(soildb, "spatial_query"), "spatial_query not exported"
    assert hasattr(soildb, "point_query"), "point_query not exported"
    assert hasattr(soildb, "bbox_query"), "bbox_query not exported"
    assert hasattr(soildb, "SpatialQueryBuilder"), "SpatialQueryBuilder not exported"


def test_core_classes_exported():
    """Core classes should be exported."""
    import soildb

    assert hasattr(soildb, "SDAClient"), "SDAClient not exported"
    assert hasattr(soildb, "SDAResponse"), "SDAResponse not exported"
    assert hasattr(soildb, "Query"), "Query not exported"


def test_all_list_matches_exports():
    """All exported symbols should be in __all__."""
    import soildb

    # Check that key functions are in __all__
    expected_in_all = [
        "fetch_by_keys",
        "fetch_mapunit_polygon",
        "fetch_component_by_mukey",
        "fetch_chorizon_by_cokey",
        "fetch_survey_area_polygon",
        "SDAClient",
        "SDAResponse",
        "spatial_query",
    ]

    for name in expected_in_all:
        assert name in soildb.__all__, f"{name} not in __all__"


def test_deprecated_functions_have_correct_signatures():
    """Deprecated functions should have compatible signatures."""
    from soildb.fetch import (
        fetch_mapunit_polygon,
        fetch_component_by_mukey,
        fetch_chorizon_by_cokey,
        fetch_survey_area_polygon,
    )
    import inspect

    # Check fetch_mapunit_polygon
    sig = inspect.signature(fetch_mapunit_polygon)
    assert "mukeys" in sig.parameters
    assert "client" in sig.parameters

    # Check fetch_component_by_mukey
    sig = inspect.signature(fetch_component_by_mukey)
    assert "mukeys" in sig.parameters
    assert "client" in sig.parameters

    # Check fetch_chorizon_by_cokey
    sig = inspect.signature(fetch_chorizon_by_cokey)
    assert "cokeys" in sig.parameters
    assert "client" in sig.parameters

    # Check fetch_survey_area_polygon
    sig = inspect.signature(fetch_survey_area_polygon)
    assert "areasymbols" in sig.parameters
    assert "client" in sig.parameters


def test_convenience_functions_exported():
    """Convenience functions should be exported."""
    import soildb

    assert hasattr(soildb, "get_mapunit_by_areasymbol")
    assert hasattr(soildb, "get_mapunit_by_point")
    assert hasattr(soildb, "get_mapunit_by_bbox")
    assert hasattr(soildb, "get_sacatalog")


def test_response_class_exported():
    """Response class should be exported."""
    import soildb

    assert hasattr(soildb, "SDAResponse")

    # Test that it's actually callable
    response_class = getattr(soildb, "SDAResponse")
    assert callable(response_class)


def test_exception_classes_exported():
    """Exception classes should be exported."""
    import soildb

    expected_exceptions = [
        "SoilDBError",
        "SDANetworkError",
        "SDAConnectionError",
        "SDAQueryError",
        "SDAResponseError",
    ]

    for exc_name in expected_exceptions:
        assert hasattr(soildb, exc_name), f"{exc_name} not exported"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
