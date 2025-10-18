"""
Unit tests for schema inference functionality.
"""

import pytest
from unittest.mock import Mock

from soildb.schema_inference import infer_column_type, create_schema_from_sda_metadata, auto_register_schema
from soildb.type_processors import to_datetime, to_date, to_bool


class TestTypeInference:
    """Test T-SQL type inference functionality."""

    def test_infer_varchar_column(self):
        """Test inference of varchar columns."""
        python_type, processor, is_optional = infer_column_type("varchar(255)")
        assert python_type == str
        assert is_optional is False
        assert callable(processor)

    def test_infer_varchar_no_length(self):
        """Test inference of varchar without length specifier."""
        python_type, processor, is_optional = infer_column_type("varchar")
        assert python_type == str
        assert is_optional is False

    def test_infer_int_column(self):
        """Test inference of int columns."""
        python_type, processor, is_optional = infer_column_type("int")
        assert str(python_type).startswith("typing.Optional[int]")
        assert is_optional is True

    def test_infer_bigint_column(self):
        """Test inference of bigint columns."""
        python_type, processor, is_optional = infer_column_type("bigint")
        assert str(python_type).startswith("typing.Optional[int]")
        assert is_optional is True

    def test_infer_float_column(self):
        """Test inference of float columns."""
        python_type, processor, is_optional = infer_column_type("float")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_real_column(self):
        """Test inference of real columns."""
        python_type, processor, is_optional = infer_column_type("real")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_numeric_column(self):
        """Test inference of numeric columns."""
        python_type, processor, is_optional = infer_column_type("numeric(10,2)")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_decimal_column(self):
        """Test inference of decimal columns."""
        python_type, processor, is_optional = infer_column_type("decimal(8,3)")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_bit_column(self):
        """Test inference of bit columns."""
        python_type, processor, is_optional = infer_column_type("bit")
        assert python_type == bool
        assert is_optional is False

    def test_infer_datetime_column(self):
        """Test inference of datetime columns."""
        python_type, processor, is_optional = infer_column_type("datetime")
        assert str(python_type).startswith("typing.Optional[datetime.datetime]")
        assert is_optional is True

    def test_infer_datetime2_column(self):
        """Test inference of datetime2 columns."""
        python_type, processor, is_optional = infer_column_type("datetime2")
        assert str(python_type).startswith("typing.Optional[datetime.datetime]")
        assert is_optional is True

    def test_infer_date_column(self):
        """Test inference of date columns."""
        python_type, processor, is_optional = infer_column_type("date")
        assert str(python_type).startswith("typing.Optional[datetime.date]")
        assert is_optional is True

    def test_infer_text_column(self):
        """Test inference of text columns."""
        python_type, processor, is_optional = infer_column_type("text")
        assert python_type == str
        assert is_optional is False

    def test_infer_nvarchar_column(self):
        """Test inference of nvarchar columns."""
        python_type, processor, is_optional = infer_column_type("nvarchar(max)")
        assert python_type == str
        assert is_optional is False

    def test_infer_unknown_type(self):
        """Test inference of unknown types falls back to string."""
        python_type, processor, is_optional = infer_column_type("unknown_type")
        assert python_type == str
        assert is_optional is False

    def test_infer_int_column(self):
        """Test inference of int columns."""
        python_type, processor, is_optional = infer_column_type("int")
        assert str(python_type).startswith("typing.Optional[int]")
        assert is_optional is True

    def test_infer_bigint_column(self):
        """Test inference of bigint columns."""
        python_type, processor, is_optional = infer_column_type("bigint")
        assert str(python_type).startswith("typing.Optional[int]")
        assert is_optional is True

    def test_infer_float_column(self):
        """Test inference of float columns."""
        python_type, processor, is_optional = infer_column_type("float")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_real_column(self):
        """Test inference of real columns."""
        python_type, processor, is_optional = infer_column_type("real")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_numeric_column(self):
        """Test inference of numeric columns."""
        python_type, processor, is_optional = infer_column_type("numeric(10,2)")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_decimal_column(self):
        """Test inference of decimal columns."""
        python_type, processor, is_optional = infer_column_type("decimal(8,3)")
        assert str(python_type).startswith("typing.Optional[float]")
        assert is_optional is True

    def test_infer_bit_column(self):
        """Test inference of bit columns."""
        python_type, processor, is_optional = infer_column_type("bit")
        assert python_type == bool
        assert is_optional is False

    def test_infer_datetime_column(self):
        """Test inference of datetime columns."""
        python_type, processor, is_optional = infer_column_type("datetime")
        assert str(python_type).startswith("typing.Optional[datetime.datetime]")
        assert is_optional is True

    def test_infer_datetime2_column(self):
        """Test inference of datetime2 columns."""
        python_type, processor, is_optional = infer_column_type("datetime2")
        assert str(python_type).startswith("typing.Optional[datetime.datetime]")
        assert is_optional is True

    def test_infer_date_column(self):
        """Test inference of date columns."""
        python_type, processor, is_optional = infer_column_type("date")
        assert str(python_type).startswith("typing.Optional[datetime.date]")
        assert is_optional is True

    def test_infer_text_column(self):
        """Test inference of text columns."""
        python_type, processor, is_optional = infer_column_type("text")
        assert python_type == str
        assert is_optional is False

    def test_infer_nvarchar_column(self):
        """Test inference of nvarchar columns."""
        python_type, processor, is_optional = infer_column_type("nvarchar(max)")
        assert python_type == str
        assert is_optional is False

    def test_infer_unknown_type(self):
        """Test inference of unknown types falls back to string."""
        python_type, processor, is_optional = infer_column_type("unknown_type")
        assert python_type == str
        assert is_optional is False


class TestSchemaCreation:
    """Test schema creation from SDA metadata."""

    def test_create_schema_from_mock_response(self):
        """Test creating schema from mock SDA response."""
        from soildb.schema_system import TableSchema

        # Mock response with metadata
        mock_response = Mock()
        mock_response._raw_data = {
            "Table": [
                ["mukey", "muname", "iscomplete"],  # Column names
                [  # Metadata
                    "ColumnOrdinal=0,DataTypeName=int",
                    "ColumnOrdinal=1,DataTypeName=varchar",
                    "ColumnOrdinal=2,DataTypeName=bit"
                ]
            ]
        }

        schema = create_schema_from_sda_metadata(mock_response, "test_table")

        assert isinstance(schema, TableSchema)
        assert schema.name == "test_table"
        assert len(schema.columns) == 3

        # Check inferred types
        assert "mukey" in schema.columns
        assert "muname" in schema.columns
        assert "iscomplete" in schema.columns

        # Check mukey (int -> Optional[int])
        mukey_col = schema.columns["mukey"]
        assert str(mukey_col.type_hint).startswith("typing.Optional[int]")

        # Check muname (varchar -> str)
        muname_col = schema.columns["muname"]
        assert muname_col.type_hint == str

        # Check iscomplete (bit -> bool)
        iscomplete_col = schema.columns["iscomplete"]
        assert iscomplete_col.type_hint == bool

    def test_create_schema_empty_response(self):
        """Test creating schema from empty response."""
        mock_response = Mock()
        mock_response._raw_data = {"Table": []}

        schema = create_schema_from_sda_metadata(mock_response, "empty_table")

        assert schema.name == "empty_table"
        assert len(schema.columns) == 0

    def test_create_schema_missing_table(self):
        """Test creating schema when Table key is missing."""
        mock_response = Mock()
        mock_response._raw_data = {}

        schema = create_schema_from_sda_metadata(mock_response, "missing_table")

        assert schema.name == "missing_table"
        assert len(schema.columns) == 0


class TestAutoRegistration:
    """Test automatic schema registration."""

    def test_auto_register_new_schema(self):
        """Test registering a new schema."""
        from soildb.schema_system import SCHEMAS

        # Ensure test_table doesn't exist
        if "test_table" in SCHEMAS:
            del SCHEMAS["test_table"]

        # Mock response
        mock_response = Mock()
        mock_response._raw_data = {
            "Table": [
                ["test_col"],
                ["ColumnOrdinal=0,DataTypeName=varchar"]
            ]
        }

        result = auto_register_schema(mock_response, "test_table")

        assert result is True
        assert "test_table" in SCHEMAS
        assert len(SCHEMAS["test_table"].columns) == 1

        # Clean up
        if "test_table" in SCHEMAS:
            del SCHEMAS["test_table"]

    def test_auto_register_existing_schema(self):
        """Test that existing schemas are not overwritten."""
        from soildb.schema_system import SCHEMAS

        # Use an existing schema
        existing_columns = len(SCHEMAS["mapunit"].columns)

        # Mock response
        mock_response = Mock()
        mock_response._raw_data = {
            "Table": [
                ["new_col"],
                ["ColumnOrdinal=0,DataTypeName=int"]
            ]
        }

        result = auto_register_schema(mock_response, "mapunit")

        assert result is False  # Should not register
        assert len(SCHEMAS["mapunit"].columns) == existing_columns  # Should be unchanged

    def test_auto_register_invalid_response(self):
        """Test handling of invalid responses."""
        mock_response = Mock()
        mock_response._raw_data = None  # Invalid

        result = auto_register_schema(mock_response, "invalid_table")

        assert result is False


class TestTypeProcessors:
    """Test the new type processors."""

    def test_to_datetime_valid(self):
        """Test datetime processor with valid input."""
        result = to_datetime("2023-01-15 14:30:00")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_to_datetime_none(self):
        """Test datetime processor with None input."""
        result = to_datetime(None)
        assert result is None

    def test_to_datetime_empty_string(self):
        """Test datetime processor with empty string."""
        result = to_datetime("")
        assert result is None

    def test_to_datetime_invalid(self):
        """Test datetime processor with invalid input."""
        result = to_datetime("invalid")
        assert result is None

    def test_to_date_valid(self):
        """Test date processor with valid input."""
        result = to_date("2023-01-15")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_to_date_none(self):
        """Test date processor with None input."""
        result = to_date(None)
        assert result is None

    def test_to_bool_true_values(self):
        """Test bool processor with true values."""
        assert to_bool("true") is True
        assert to_bool("1") is True
        assert to_bool("yes") is True
        assert to_bool("t") is True
        assert to_bool(1) is True

    def test_to_bool_false_values(self):
        """Test bool processor with false values."""
        assert to_bool("false") is False
        assert to_bool("0") is False
        assert to_bool("no") is False
        assert to_bool("f") is False
        assert to_bool(0) is False
        assert to_bool(None) is False
        assert to_bool("") is False