"""
Unit tests for spatial.py interface functions (spatial_query, point_query, bbox_query).

Tests cover:
- spatial_query with various geometries and parameters via httpx_mock
- point_query and bbox_query convenience functions
- Client lifecycle management (auto-creation and closure)
- Invalid WKT validation
- .sync() methods for decorated functions
"""

import json
import re
from unittest.mock import patch

import pytest

from soildb.client import SDAClient
from soildb.response import SDAResponse
from soildb.spatial import (
    SpatialQueryBuilder,
    bbox_query,
    point_query,
    spatial_query,
)


@pytest.fixture
def mock_sda_response():
    """Create a valid SDA JSON response for mocking."""
    return {
        "Table": [
            ["mukey", "musym", "muname", "mukind", "areasymbol", "areaname"],
            [
                "ColumnOrdinal=0,DataTypeName=int",
                "ColumnOrdinal=1,DataTypeName=varchar",
                "ColumnOrdinal=2,DataTypeName=varchar",
                "ColumnOrdinal=3,DataTypeName=varchar",
                "ColumnOrdinal=4,DataTypeName=varchar",
                "ColumnOrdinal=5,DataTypeName=varchar",
            ],
            [
                "123456",
                "55B",
                "Clarion loam, 2 to 5 percent slopes",
                "Series",
                "IA109",
                "Story County, Iowa",
            ],
            [
                "123457",
                "138B",
                "Nicollet loam, 1 to 3 percent slopes",
                "Series",
                "IA109",
                "Story County, Iowa",
            ],
        ]
    }


@pytest.fixture
def mock_sda_response_spatial():
    """Create a SDA response with geometry column for spatial queries."""
    return {
        "Table": [
            ["mukey", "musym", "muname", "geometry"],
            [
                "ColumnOrdinal=0,DataTypeName=int",
                "ColumnOrdinal=1,DataTypeName=varchar",
                "ColumnOrdinal=2,DataTypeName=varchar",
                "ColumnOrdinal=3,DataTypeName=varchar",
            ],
            ["123456", "55B", "Clarion loam, 2 to 5 percent slopes", "POLYGON((...))"],
        ]
    }


# ============================================================================
# Tests for spatial_query with httpx_mock
# ============================================================================


@pytest.mark.asyncio
class TestSpatialQueryViaHttpx:
    """Test spatial_query through httpx_mock with no client provided."""

    async def test_spatial_query_point_drives_http_request(
        self, httpx_mock, mock_sda_response
    ):
        """Test spatial_query with point geometry drives HTTP request with no explicit client.

        The decorated function creates a client, executes the query, and closes the client.
        httpx_mock intercepts the HTTP request at the httpx level.
        Uses custom columns to test standard spatial join path (not UDF).
        """
        # Setup httpx_mock to return SDA response
        httpx_mock.add_response(json=mock_sda_response)

        # Call spatial_query without client (decorator creates one)
        # Use custom columns to avoid UDF path and test standard spatial join
        result = await spatial_query(
            "POINT(-94.68 42.03)",
            table="mupolygon",
            return_type="tabular",
            what="mukey,muname,musym",
        )

        # Verify response is parsed correctly
        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2  # Two rows of data
        assert "mukey" in result.columns

        # Verify request was made
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]
        request_body = request.content.decode("utf-8")

        # Verify SQL contains expected spatial query elements
        assert "POINT" in request_body
        assert "mupolygongeo" in request_body

    async def test_spatial_query_bbox_drives_http_request(
        self, httpx_mock, mock_sda_response
    ):
        """Test spatial_query with bbox geometry drives HTTP request.

        Uses custom columns to test standard spatial join path (not UDF).
        """
        httpx_mock.add_response(json=mock_sda_response)

        bbox = {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}
        result = await spatial_query(
            bbox,
            table="mupolygon",
            return_type="tabular",
            what="mukey,muname,musym",
        )

        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2

        # Verify SQL contains polygon WKT
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request_body = requests[0].content.decode("utf-8")
        assert "POLYGON" in request_body
        assert "mupolygongeo" in request_body


# ============================================================================
# Tests for point_query with httpx_mock
# ============================================================================


class TestPointQueryViaHttpx:
    """Test point_query through httpx_mock with no client provided."""

    @pytest.mark.asyncio
    async def test_point_query_drives_http_request(self, httpx_mock, mock_sda_response):
        """Test point_query without explicit client drives HTTP request.

        Verifies that latitude/longitude are converted to POINT WKT and sent to SDA.
        """
        httpx_mock.add_response(json=mock_sda_response)

        result = await point_query(latitude=42.0, longitude=-93.6)

        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2
        assert "mukey" in result.columns

        # Verify request contains POINT WKT with correct order (longitude, latitude)
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request_body = requests[0].content.decode("utf-8")
        assert "POINT" in request_body
        assert "-93.6" in request_body  # longitude
        assert "42.0" in request_body  # latitude

    def test_point_query_sync_method_works(self, httpx_mock, mock_sda_response):
        """Test that point_query.sync() method exists and works.

        The @add_sync_version decorator adds a .sync() method that runs the
        async function in a fresh event loop.
        """
        httpx_mock.add_response(json=mock_sda_response)

        # Call via .sync() from synchronous context
        result = point_query.sync(latitude=42.0, longitude=-93.6)

        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2
        assert "mukey" in result.columns

        # Verify HTTP request was made
        requests = httpx_mock.get_requests()
        assert len(requests) == 1


# ============================================================================
# Tests for bbox_query with httpx_mock
# ============================================================================


class TestBboxQueryViaHttpx:
    """Test bbox_query through httpx_mock with no client provided."""

    @pytest.mark.asyncio
    async def test_bbox_query_drives_http_request(self, httpx_mock, mock_sda_response):
        """Test bbox_query without explicit client drives HTTP request.

        Verifies that bbox coordinates are converted to POLYGON WKT and sent to SDA.
        """
        httpx_mock.add_response(json=mock_sda_response)

        result = await bbox_query(xmin=-93.8, ymin=41.8, xmax=-93.4, ymax=42.2)

        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2
        assert "mukey" in result.columns

        # Verify request contains POLYGON WKT with correct bbox bounds
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request_body = requests[0].content.decode("utf-8")
        assert "POLYGON" in request_body
        assert "-93.8" in request_body
        assert "41.8" in request_body
        assert "-93.4" in request_body
        assert "42.2" in request_body


# ============================================================================
# Tests for client lifecycle (auto-creation and closure)
# ============================================================================


@pytest.mark.asyncio
class TestClientLifecycle:
    """Test client lifecycle management with auto-creation and closure."""

    async def test_auto_client_created_and_closed_on_success(
        self, httpx_mock, mock_sda_response
    ):
        """Test that auto-created client is closed after successful query.

        When spatial_query is called without a client, the @add_sync_version
        decorator's async_wrapper creates one. Patching SDAClient.close
        verifies it was called exactly once.
        """
        httpx_mock.add_response(json=mock_sda_response)

        with patch.object(SDAClient, "close") as mock_close:
            result = await spatial_query(
                "POINT(-94.68 42.03)",
                table="mupolygon",
            )

        # Verify query executed and returned result
        assert isinstance(result, SDAResponse)

        # Verify close was called exactly once (once for the auto-created client)
        assert mock_close.call_count == 1

    async def test_auto_client_closed_even_on_http_error(self, httpx_mock):
        """Test that auto-created client is closed even when HTTP request fails.

        Verifies that the decorator's error handling ensures cleanup happens.
        """
        # Setup httpx_mock to return an error response
        httpx_mock.add_response(status_code=500, json={"error": "Server error"})

        with patch.object(SDAClient, "close") as mock_close:
            # Query should fail due to HTTP error
            try:
                await spatial_query(
                    "POINT(-94.68 42.03)",
                    table="mupolygon",
                )
            except Exception:  # HTTPStatusError raised by httpx client
                pass

        # Verify close was still called (for cleanup)
        assert mock_close.call_count == 1


# ============================================================================
# Tests for WKT validation
# ============================================================================


@pytest.mark.asyncio
class TestWKTValidation:
    """Test WKT geometry validation."""

    async def test_invalid_wkt_geometry_raises_error(self):
        """Test that invalid WKT raises ValueError."""
        with pytest.raises(ValueError, match="Invalid WKT geometry"):
            await spatial_query("INVALID(-94.68 42.03)")

    async def test_invalid_wkt_missing_parens_raises_error(self):
        """Test that WKT without parentheses raises ValueError."""
        with pytest.raises(ValueError, match="Invalid WKT geometry"):
            await spatial_query("POINT -94.68 42.03")

    async def test_invalid_wkt_random_string_raises_error(self):
        """Test that random string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid WKT geometry"):
            await spatial_query("not a geometry")


# ============================================================================
# Tests for SpatialQueryBuilder helper methods
# ============================================================================


class TestSpatialQueryBuilder:
    """Test internal SpatialQueryBuilder methods (non-async)."""

    def test_geometry_column_retrieval(self):
        """Test geometry column retrieval from builder."""
        builder = SpatialQueryBuilder()

        # Test mupolygon geometry column
        assert builder._get_geometry_column("mupolygon") == "p.mupolygongeo"

        # Test sapolygon geometry column
        assert builder._get_geometry_column("sapolygon") == "s.sapolygongeo"

        # Test non-spatial table raises error
        with pytest.raises(ValueError, match="does not have spatial data"):
            builder._get_geometry_column("mapunit")

        # Test unsupported table raises error
        with pytest.raises(
            ValueError,
            match=r"Unknown table: 'invalid_table'\. Supported spatial tables:",
        ):
            builder._get_geometry_column("invalid_table")  # type: ignore[arg-type]

    def test_query_unsupported_table_raises_value_error(self):
        """Test that SpatialQueryBuilder.query raises ValueError for unsupported table."""
        builder = SpatialQueryBuilder()
        with pytest.raises(
            ValueError,
            match=r"Unknown table: 'invalid_table'\. Supported spatial tables:",
        ):
            builder.query("POINT(-94.68 42.03)", table="invalid_table")  # type: ignore[arg-type]

    def test_spatial_predicate_mapping(self):
        """Test spatial predicate mapping."""
        builder = SpatialQueryBuilder()

        predicates = {
            "intersects": "STIntersects",
            "contains": "STContains",
            "within": "STWithin",
            "touches": "STTouches",
            "crosses": "STCrosses",
            "overlaps": "STOverlaps",
        }

        for relation, expected_predicate in predicates.items():
            result = builder._get_spatial_predicate(relation)
            assert result == expected_predicate

    def test_wkt_string_passthrough(self):
        """Test that WKT string passes through validation unchanged."""
        builder = SpatialQueryBuilder()
        wkt = "POINT(-94.68 42.03)"
        result = builder._geometry_to_wkt(wkt)

        assert result == wkt

    def test_bbox_dict_conversion(self):
        """Test bbox dict conversion to WKT polygon."""
        builder = SpatialQueryBuilder()
        bbox = {"xmin": -94.7, "ymin": 42.0, "xmax": -94.6, "ymax": 42.1}
        wkt = builder._geometry_to_wkt(bbox)

        # Should create a closed polygon
        assert "POLYGON" in wkt
        assert "-94.7" in wkt
        assert "42.0" in wkt
        assert "-94.6" in wkt
        assert "42.1" in wkt

    def test_bbox_invalid_keys_raises_error(self):
        """Test that bbox dict with invalid keys raises error."""
        bbox = {"x_min": -94.7, "y_min": 42.0, "x_max": -94.6, "y_max": 42.1}

        with pytest.raises(ValueError, match="xmin, ymin, xmax, ymax"):
            builder = SpatialQueryBuilder()
            builder._geometry_to_wkt(bbox)


# ============================================================================
# Tests for Shapely geometry input (conditional on shapely availability)
# ============================================================================


class TestShapelyGeometryInput:
    """Test shapely geometry input (skip if shapely not installed)."""

    def test_shapely_point_conversion(self):
        """Test conversion of shapely Point to WKT."""
        try:
            from shapely.geometry import Point
        except ImportError:
            pytest.skip("shapely not installed")

        builder = SpatialQueryBuilder()
        point = Point(-94.68, 42.03)
        wkt = builder._geometry_to_wkt(point)

        assert "POINT" in wkt
        assert "-94.68" in wkt
        assert "42.03" in wkt

    def test_shapely_polygon_conversion(self):
        """Test conversion of shapely Polygon to WKT."""
        try:
            from shapely.geometry import Polygon
        except ImportError:
            pytest.skip("shapely not installed")

        builder = SpatialQueryBuilder()
        polygon = Polygon([(-94.7, 42.0), (-94.6, 42.0), (-94.6, 42.1), (-94.7, 42.1)])
        wkt = builder._geometry_to_wkt(polygon)

        assert "POLYGON" in wkt
        assert "-94.7" in wkt or "94.7" in wkt
        assert "42.0" in wkt or "42" in wkt

    @pytest.mark.asyncio
    async def test_shapely_point_in_spatial_query(self, httpx_mock, mock_sda_response):
        """Test using shapely Point directly in spatial_query."""
        try:
            from shapely.geometry import Point
        except ImportError:
            pytest.skip("shapely not installed")

        httpx_mock.add_response(json=mock_sda_response)

        point = Point(-94.68, 42.03)
        result = await spatial_query(point)

        assert isinstance(result, SDAResponse)
        assert len(result.data) == 2


# ============================================================================
# Tests for table alias consistency in spatial queries
# ============================================================================


@pytest.mark.asyncio
class TestSpatialQueryTableAliases:
    """Test that all alias prefixes in SELECT are defined in FROM/JOIN clauses."""

    @pytest.mark.parametrize(
        "table,geom_col",
        [
            ("mupoint", None),
            ("muline", None),
            ("legend", "l.geom"),
            ("mapunit", "m.geom"),
        ],
    )
    async def test_select_aliases_defined_in_from_or_join(
        self, httpx_mock, mock_sda_response, table, geom_col
    ):
        """Assert that every alias prefix in SELECT is defined in FROM/JOIN clauses.

        Fixed tables (mupoint, muline, legend, mapunit) must define aliases in
        the FROM/JOIN clauses that match their default column selections.
        """
        httpx_mock.add_response(json=mock_sda_response)

        kwargs = {"table": table, "return_type": "tabular"}
        if geom_col is not None:
            kwargs["geom_column"] = geom_col

        result = await spatial_query("POINT(-94.68 42.03)", **kwargs)
        assert isinstance(result, SDAResponse)

        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        payload = json.loads(requests[0].content.decode("utf-8"))
        sql = payload["query"]

        # Parse SELECT and FROM/JOIN clauses
        select_match = re.search(
            r"SELECT\s+(?:DISTINCT\s+)?(.*?)\s+(FROM\s+.*)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        assert select_match is not None, f"Could not parse SQL: {sql}"
        select_part = select_match.group(1)
        from_part = select_match.group(2)

        # Extract alias prefixes used in SELECT (e.g., 'm.muname' -> 'm')
        select_aliases = set(
            re.findall(
                r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*\b",
                select_part,
            )
        )
        assert len(select_aliases) > 0, f"Expected aliases in SELECT for table {table}"

        # Extract table aliases defined in FROM and JOIN clauses
        # Matches 'FROM <table> <alias>' and 'JOIN <table> <alias>'
        from_join_aliases = set(
            re.findall(
                r"(?:FROM|JOIN)\s+[a-zA-Z_][a-zA-Z0-9_]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
                from_part,
                re.IGNORECASE,
            )
        )

        # Every alias used in SELECT must be defined in FROM/JOIN
        assert select_aliases.issubset(from_join_aliases), (
            f"Table {table}: undefined aliases {select_aliases - from_join_aliases} "
            f"found in SELECT: {select_part}. Defined aliases: {from_join_aliases}."
        )

    async def test_unsupported_table_raises_value_error(self):
        """Assert ValueError is raised when an unsupported or unaliased table is queried."""
        with pytest.raises(
            ValueError,
            match=r"Unknown table: 'nonexistent'\. Supported spatial tables:",
        ):
            await spatial_query("POINT(-94.68 42.03)", table="nonexistent")  # type: ignore[arg-type]

    async def test_unsupported_table_with_geom_col_raises_value_error(self):
        """Assert ValueError is raised for unsupported table even when geom_column is provided."""
        with pytest.raises(
            ValueError,
            match=r"Unknown table: 'nonexistent'\. Supported spatial tables:",
        ):
            await spatial_query(
                "POINT(-94.68 42.03)",
                table="nonexistent",  # type: ignore[arg-type]
                geom_column="geom",
            )
