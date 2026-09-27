"""Integration tests for spatial queries against Soil Data Access (SDA).

Proves spatial queries for point, line, and polygon SSURGO tables against
the live SDA web service using reliable client configuration.
"""

from typing import cast

import pytest

from soildb.base_client import ClientConfig
from soildb.client import SDAClient
from soildb.response import SDAResponse
from soildb.spatial import ReturnType, TableType, point_query, spatial_query


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table",
    ["mupoint", "muline", "featpoint", "featline", "mupolygon"],
)
@pytest.mark.parametrize("return_type", ["tabular", "spatial"])
async def test_spatial_query_bbox_tables_against_sda(
    table: str, return_type: str
) -> None:
    """Test spatial_query against SDA for various tables and return types.

    Executes spatial_query over a small bounding box with ClientConfig.reliable().
    Must complete without an SDAQueryError. When rows are returned, verifies
    expected columns are present in the response.
    """
    bbox = {"xmin": -93.7, "ymin": 42.0, "xmax": -93.6, "ymax": 42.1}
    typed_table = cast(TableType, table)
    typed_return_type = cast(ReturnType, return_type)

    async with SDAClient(config=ClientConfig.reliable()) as client:
        result = await spatial_query(
            bbox,
            table=typed_table,
            return_type=typed_return_type,
            client=client,
        )
        assert isinstance(result, SDAResponse)
        if len(result) > 0:
            if table == "mupolygon":
                expected_columns = ["mukey", "musym", "muname", "mukind", "areasymbol"]
            elif table in ("mupoint", "muline"):
                expected_columns = ["mukey", "musym", "muname"]
            elif table in ("featpoint", "featline"):
                expected_columns = ["featkey", "featsym"]
            else:
                expected_columns = []

            if return_type == "spatial":
                expected_columns.append("geometry")

            for col in expected_columns:
                assert col in result.columns


@pytest.mark.integration
@pytest.mark.asyncio
async def test_point_query_mupolygon_known_location() -> None:
    """Test point_query on mupolygon at known location returning rows.

    Reuses coordinates (latitude 42.0, longitude -93.6) from documentation examples
    and verifies at least one row is returned with expected columns.
    """
    async with SDAClient(config=ClientConfig.reliable()) as client:
        result = await point_query(
            latitude=42.0,
            longitude=-93.6,
            table="mupolygon",
            client=client,
        )
        assert isinstance(result, SDAResponse)
        assert len(result) >= 1
        for col in ["mukey", "muname", "musym"]:
            assert col in result.columns
