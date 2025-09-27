"""
Test configuration and fixtures for soildb tests.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from soildb import SDAClient


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
