"""
Tests for SDA response handling with type conversion.
"""

from datetime import datetime

import pytest

from soildb.exceptions import SDAResponseError
from soildb.response import SDAResponse


class TestSDAResponse:
    """Test SDAResponse parsing and conversion."""

    def test_parse_valid_response(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)

        assert len(response.columns) == 4
        assert response.columns == ["areasymbol", "mukey", "musym", "muname"]
        assert len(response) == 2
        assert not response.is_empty()

    def test_parse_empty_response(self, empty_sda_response_json):
        response = SDAResponse.from_json(empty_sda_response_json)

        assert len(response.columns) == 2
        assert len(response) == 0
        assert response.is_empty()

    def test_to_dict_with_type_conversion(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        records = response.to_dict()

        assert len(records) == 2
        assert records[0]["areasymbol"] == "IA109"
        assert records[0]["mukey"] == "123456"
        assert records[1]["musym"] == "138B"

    def test_column_types(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        types = response.get_column_types()

        assert types["areasymbol"] == "varchar"
        assert types["mukey"] == "varchar"

    def test_python_types(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        python_types = response.get_python_types()

        assert python_types["areasymbol"] == "string"
        assert python_types["mukey"] == "string"

    def test_invalid_json(self):
        with pytest.raises(SDAResponseError):
            SDAResponse.from_json("invalid json")

    def test_missing_table_key(self):
        with pytest.raises(SDAResponseError):
            SDAResponse.from_json('{"NotTable": []}')

    def test_iteration(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        rows = list(response)

        assert len(rows) == 2
        assert rows[0] == [
            "IA109",
            "123456",
            "55B",
            "Clarion loam, 2 to 5 percent slopes",
        ]

    def test_repr(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        repr_str = repr(response)

        assert "SDAResponse" in repr_str
        assert "columns=4" in repr_str
        assert "rows=2" in repr_str

    def test_str_with_types(self, sample_sda_response_json):
        response = SDAResponse.from_json(sample_sda_response_json)
        str_repr = str(response)

        assert "SDA Response:" in str_repr
        assert "Types:" in str_repr


class TestTypeConversion:
    """Test data type conversion functionality."""

    def test_numeric_conversion(self):
        """Test conversion of numeric data types."""
        numeric_response = {
            "Table": [
                ["intcol", "floatcol", "bitcol"],
                [
                    "ColumnOrdinal=0,DataTypeName=int",
                    "ColumnOrdinal=1,DataTypeName=float",
                    "ColumnOrdinal=2,DataTypeName=bit",
                ],
                ["123", "45.67", "1"],
                ["456", "89.01", "0"],
                [None, None, None],
            ]
        }

        response = SDAResponse(numeric_response)
        records = response.to_dict()

        # Check type conversion in dictionary format
        assert isinstance(records[0]["intcol"], int)
        assert records[0]["intcol"] == 123
        assert isinstance(records[0]["floatcol"], float)
        assert records[0]["floatcol"] == 45.67
        assert isinstance(records[0]["bitcol"], bool)
        assert records[0]["bitcol"] is True
        assert records[1]["bitcol"] is False

        # Check null handling
        assert records[2]["intcol"] is None
        assert records[2]["floatcol"] is None
        assert records[2]["bitcol"] is None

    def test_datetime_conversion(self):
        """Test datetime conversion."""
        datetime_response = {
            "Table": [
                ["datecol"],
                ["ColumnOrdinal=0,DataTypeName=datetime"],
                ["2020-01-15 10:30:00"],
                ["2019-12-20"],
            ]
        }

        response = SDAResponse(datetime_response)
        records = response.to_dict()

        assert isinstance(records[0]["datecol"], datetime)
        assert records[0]["datecol"].year == 2020
        assert records[0]["datecol"].month == 1
        assert records[0]["datecol"].day == 15


class TestDataFrameIntegration:
    """Test DataFrame conversion with type handling."""

    def test_to_pandas_with_types(self):
        """Test pandas conversion with proper types."""
        mixed_response = {
            "Table": [
                ["strcol", "intcol", "floatcol"],
                [
                    "ColumnOrdinal=0,DataTypeName=varchar",
                    "ColumnOrdinal=1,DataTypeName=int",
                    "ColumnOrdinal=2,DataTypeName=float",
                ],
                ["test1", "123", "45.67"],
                ["test2", "456", "89.01"],
            ]
        }

        response = SDAResponse(mixed_response)

        try:
            df = response.to_pandas()

            # Check that types are properly converted
            assert len(df) == 2
            assert df.iloc[0]["strcol"] == "test1"

            # Check numeric values (could be numpy types)
            intval = df.iloc[0]["intcol"]
            floatval = df.iloc[0]["floatcol"]

            # Check that the values are correct and numeric
            assert int(intval) == 123  # Convert to check value
            assert float(floatval) == 45.67

            # Check that dtypes are appropriate
            assert "int" in str(df["intcol"].dtype).lower()
            assert "float" in str(df["floatcol"].dtype).lower()

        except ImportError:
            pytest.skip("pandas not available")

    def test_type_conversion_disabled(self, sample_sda_response_json):
        """Test disabling type conversion."""
        response = SDAResponse.from_json(sample_sda_response_json)

        try:
            df = response.to_pandas(convert_types=False)
            # When type conversion is disabled, everything should be object/string
            assert len(df) == 2

        except ImportError:
            pytest.skip("pandas not available")


@pytest.fixture
def numeric_sda_response_json():
    """Sample SDA response with numeric data."""
    return """
    {
        "Table": [
            ["mukey", "muacres", "brockdepmin"],
            ["ColumnOrdinal=0,DataTypeName=varchar", "ColumnOrdinal=1,DataTypeName=float", "ColumnOrdinal=2,DataTypeName=int"],
            ["123456", "1234.5", "60"],
            ["123457", "2345.7", "48"]
        ]
    }
    """
