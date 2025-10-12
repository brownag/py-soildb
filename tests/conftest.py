"""
Test configuration and fixtures for soildb tests.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from soildb import SDAClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def sda_client():
    """Create a real SDAClient for integration tests."""
    client = SDAClient()
    yield client
    await client.close()


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = Mock(spec=SDAClient)
    client.execute = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_sda_response_json():
    """Sample SDA response in JSON format."""
    return """
    {
        "Table": [
            ["areasymbol", "mukey", "musym", "muname"],
            ["ColumnOrdinal=0,DataTypeName=varchar", "ColumnOrdinal=1,DataTypeName=varchar",
             "ColumnOrdinal=2,DataTypeName=varchar", "ColumnOrdinal=3,DataTypeName=varchar"],
            ["IA109", "123456", "55B", "Clarion loam, 2 to 5 percent slopes"],
            ["IA109", "123457", "138B", "Nicollet loam, 1 to 3 percent slopes"]
        ]
    }
    """


@pytest.fixture
def empty_sda_response_json():
    """Empty SDA response in JSON format."""
    return """
    {
        "Table": [
            ["areasymbol", "mukey"],
            ["ColumnOrdinal=0,DataTypeName=varchar", "ColumnOrdinal=1,DataTypeName=varchar"]
        ]
    }
    """


@pytest.fixture
def no_soilprofilecollection(monkeypatch):
    """
    Fixture to simulate that the 'soilprofilecollection' package is not installed.
    """
    import_orig = __import__

    def import_mock(name, *args, **kwargs):
        if name == "soilprofilecollection":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_mock)
