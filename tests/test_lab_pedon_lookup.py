"""Tests for explicit lab pedon lookup by identifier kind."""

import json
from typing import get_args

import pytest

from soildb import AmbiguousPedonError, fetch_labpedon, get_lab_pedon
from soildb.client import SDAClient
from soildb.convenience import (
    LabPedonIdColumn,
    _get_lab_pedon_key_then_id,
    get_lab_pedon_by_id,
)
from soildb.high_level import fetch_labpedon_by_id

COLUMNS = [
    "pedon_key",
    "upedonid",
    "latitude_decimal_degrees",
    "longitude_decimal_degrees",
]
TYPES = [
    "DataTypeName=int",
    "DataTypeName=varchar",
    "DataTypeName=float",
    "DataTypeName=float",
]


def _make_sda_json_response(*rows):
    """Create SDA JSON response in the correct format.

    Args:
        *rows: Variable number of rows, each a list/tuple matching COLUMNS length.

    Returns:
        JSON string matching SDA response format: {"Table": [columns, metadata, *rows]}
    """
    return json.dumps({"Table": [COLUMNS, TYPES, *[list(r) for r in rows]]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "what, expected",
    [
        ("pedon_key", "pedon_key = '12345'"),
        ("upedonid", "upedonid = '12345'"),
        ("pedoniid", "pedoniid = '12345'"),
        ("pedlabsampnum", "pedlabsampnum = '12345'"),
    ],
)
async def test_get_lab_pedon_filters_on_requested_column_only(
    what, expected, httpx_mock
):
    httpx_mock.add_response(text=_make_sda_json_response())

    async with SDAClient() as client:
        await get_lab_pedon("12345", what=what, client=client)

    # Verify exactly one request was made
    requests = httpx_mock.get_requests()
    assert len(requests) == 1

    # Verify the SQL in the request body contains the expected filter
    request_body = json.loads(requests[0].content.decode())
    assert expected in request_body["query"]


@pytest.mark.asyncio
async def test_get_lab_pedon_does_not_fall_back_when_key_not_found(httpx_mock):
    httpx_mock.add_response(text=_make_sda_json_response())

    async with SDAClient() as client:
        response = await get_lab_pedon("S1999NY061001", client=client)

    assert response.is_empty()

    # Verify exactly one request was made (no fallback)
    requests = httpx_mock.get_requests()
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_get_lab_pedon_rejects_unknown_identifier():
    async with SDAClient() as client:
        with pytest.raises(ValueError, match="what must be one of"):
            await get_lab_pedon("x", what="peiid", client=client)


@pytest.mark.asyncio
async def test_fetch_labpedon_raises_when_pedon_id_matches_several_pedons(httpx_mock):
    httpx_mock.add_response(
        text=_make_sda_json_response(
            [12345, "S1999NY061001", 42.0, -76.0], [67890, "S1999NY061001", 43.0, -75.0]
        )
    )

    async with SDAClient() as client:
        with pytest.raises(AmbiguousPedonError) as excinfo:
            await fetch_labpedon("S1999NY061001", what="upedonid", client=client)

    assert excinfo.value.pedon_keys == ["12345", "67890"]


@pytest.mark.asyncio
async def test_fetch_labpedon_returns_single_match(httpx_mock):
    httpx_mock.add_response(
        text=_make_sda_json_response([12345, "S1999NY061001", 42.0, -76.0])
    )

    async with SDAClient() as client:
        pedon = await fetch_labpedon(
            "S1999NY061001", what="upedonid", fill_horizons=False, client=client
        )

    assert pedon is not None
    assert pedon.pedon_key == "12345"
    assert pedon.pedon_id == "S1999NY061001"
    assert pedon.extra_fields["query_what"] == "upedonid"


@pytest.mark.asyncio
async def test_fetch_labpedon_returns_none_when_not_found(httpx_mock):
    httpx_mock.add_response(text=_make_sda_json_response())

    async with SDAClient() as client:
        result = await fetch_labpedon("99999", fill_horizons=False, client=client)

    assert result is None


@pytest.mark.asyncio
async def test_deprecated_lookups_emit_exactly_one_warning(httpx_mock):
    """Each *_by_id call should emit exactly one DeprecationWarning."""
    httpx_mock.add_response(text=_make_sda_json_response([12345, "A", 42.0, -76.0]))

    with pytest.warns(DeprecationWarning, match="get_lab_pedon") as record:
        async with SDAClient() as client:
            await get_lab_pedon_by_id("12345", client=client)
    assert len(record) == 1

    httpx_mock.add_response(text=_make_sda_json_response([12345, "A", 42.0, -76.0]))

    with pytest.warns(DeprecationWarning, match="fetch_labpedon") as record:
        async with SDAClient() as client:
            await fetch_labpedon_by_id(
                "12345",
                fill_horizons=False,
                client=client,
            )
    assert len(record) == 1


def test_lab_pedon_id_column_literal():
    assert get_args(LabPedonIdColumn) == (
        "pedon_key",
        "pedoniid",
        "upedonid",
        "pedlabsampnum",
    )


@pytest.mark.asyncio
async def test_get_lab_pedon_key_then_id_requires_client():
    with pytest.raises(TypeError, match="client is required"):
        await _get_lab_pedon_key_then_id("12345", client=None)


def test_deprecated_lookup_docstrings():
    assert ".. deprecated:: 0.9.0" in (get_lab_pedon_by_id.__doc__ or "")
    doc = fetch_labpedon_by_id.__doc__ or ""
    assert ".. deprecated:: 0.9.0" in doc
    assert "Args:" in doc
    assert "Returns:" in doc
