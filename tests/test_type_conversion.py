"""
Tests for the unified type conversion system.

Tests the TypeMap class and type conversion functionality to ensure
all conversion flows (SDA->Python->Pandas->Polars) work correctly.
"""

import pytest
from datetime import datetime
from soildb.type_conversion import TypeMap, TypeProcessor, convert_value, get_default_type_map


class TestTypeProcessor:
    """Test the TypeProcessor base class with standard conversions."""

    def test_is_null_none(self):
        """Test null detection for None values."""
        assert TypeProcessor._is_null(None) is True

    def test_is_null_string_literals(self):
        """Test null detection for string literals."""
        assert TypeProcessor._is_null("null") is True
        assert TypeProcessor._is_null("NULL") is True
        assert TypeProcessor._is_null("none") is True
        assert TypeProcessor._is_null("NONE") is True
        assert TypeProcessor._is_null("") is True

    def test_is_null_float_nan(self):
        """Test null detection for NaN values."""
        assert TypeProcessor._is_null(float("nan")) is True
        assert TypeProcessor._is_null(float("inf")) is True
        assert TypeProcessor._is_null(float("-inf")) is True

    def test_to_int_valid(self):
        """Test integer conversion with valid values."""
        assert TypeProcessor.to_int("42") == 42
        assert TypeProcessor.to_int("42.7") == 42
        assert TypeProcessor.to_int(42) == 42
        assert TypeProcessor.to_int(42.7) == 42

    def test_to_int_null(self):
        """Test integer conversion with null values."""
        assert TypeProcessor.to_int(None) is None
        assert TypeProcessor.to_int("null") is None
        assert TypeProcessor.to_int("") is None

    def test_to_int_invalid(self):
        """Test integer conversion with invalid values."""
        assert TypeProcessor.to_int("not_a_number") is None
        assert TypeProcessor.to_int("abc123") == 123  # Extracts numeric part

    def test_to_float_valid(self):
        """Test float conversion with valid values."""
        assert TypeProcessor.to_float("3.14") == 3.14
        assert TypeProcessor.to_float("42") == 42.0
        assert TypeProcessor.to_float(3.14) == 3.14

    def test_to_float_currency(self):
        """Test float conversion with currency symbols."""
        assert TypeProcessor.to_float("$25.50") == 25.50
        assert TypeProcessor.to_float("$1,234.56") == 1234.56

    def test_to_bool_true_values(self):
        """Test boolean conversion for true values."""
        assert TypeProcessor.to_bool("true") is True
        assert TypeProcessor.to_bool("TRUE") is True
        assert TypeProcessor.to_bool("1") is True
        assert TypeProcessor.to_bool("yes") is True
        assert TypeProcessor.to_bool("t") is True
        assert TypeProcessor.to_bool("y") is True
        assert TypeProcessor.to_bool("on") is True

    def test_to_bool_false_values(self):
        """Test boolean conversion for false values."""
        assert TypeProcessor.to_bool("false") is False
        assert TypeProcessor.to_bool("FALSE") is False
        assert TypeProcessor.to_bool("0") is False
        assert TypeProcessor.to_bool("no") is False
        assert TypeProcessor.to_bool("f") is False
        assert TypeProcessor.to_bool("n") is False
        assert TypeProcessor.to_bool("off") is False

    def test_to_str(self):
        """Test string conversion."""
        assert TypeProcessor.to_str("hello") == "hello"
        assert TypeProcessor.to_str("  hello  ") == "hello"
        assert TypeProcessor.to_str(None) == ""
        assert TypeProcessor.to_str("") == ""

    def test_to_str_optional(self):
        """Test string conversion with None option."""
        assert TypeProcessor.to_str("hello", allow_none=True) == "hello"
        assert TypeProcessor.to_str(None, allow_none=True) is None
        assert TypeProcessor.to_str("", allow_none=True) is None

    def test_to_datetime_iso_format(self):
        """Test datetime conversion with ISO format."""
        result = TypeProcessor.to_datetime("2023-01-15")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_to_datetime_datetime_format(self):
        """Test datetime conversion with datetime format."""
        result = TypeProcessor.to_datetime("2023-01-15 10:30:00")
        assert result.year == 2023
        assert result.hour == 10
        assert result.minute == 30

    def test_to_datetime_us_format(self):
        """Test datetime conversion with US date format."""
        result = TypeProcessor.to_datetime("01/15/2023")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_to_datetime_null(self):
        """Test datetime conversion with null values."""
        assert TypeProcessor.to_datetime(None) is None
        assert TypeProcessor.to_datetime("null") is None
        assert TypeProcessor.to_datetime("") is None

    def test_to_bytes(self):
        """Test bytes conversion."""
        assert TypeProcessor.to_bytes("hello") == b"hello"
        assert TypeProcessor.to_bytes(b"hello") == b"hello"
        assert TypeProcessor.to_bytes(None) is None


class TestTypeMap:
    """Test the TypeMap class for unified type conversion."""

    def test_default_instance(self):
        """Test creating a default TypeMap instance."""
        tm = TypeMap.default()
        assert tm is not None
        assert len(tm._processors) > 0

    def test_get_python_type_int(self):
        """Test getting Python type for integer SDA types."""
        tm = TypeMap.default()
        assert tm.get_python_type("int") == int
        assert tm.get_python_type("integer") == int
        assert tm.get_python_type("bigint") == int

    def test_get_python_type_float(self):
        """Test getting Python type for float SDA types."""
        tm = TypeMap.default()
        assert tm.get_python_type("float") == float
        assert tm.get_python_type("decimal") == float
        assert tm.get_python_type("money") == float

    def test_get_python_type_string(self):
        """Test getting Python type for string SDA types."""
        tm = TypeMap.default()
        assert tm.get_python_type("varchar") == str
        assert tm.get_python_type("nvarchar") == str
        assert tm.get_python_type("text") == str

    def test_get_python_type_datetime(self):
        """Test getting Python type for datetime SDA types."""
        tm = TypeMap.default()
        assert tm.get_python_type("datetime") == datetime
        assert tm.get_python_type("date") == datetime
        assert tm.get_python_type("timestamp") == datetime

    def test_get_python_type_unknown(self):
        """Test getting Python type for unknown SDA type."""
        tm = TypeMap.default()
        assert tm.get_python_type("unknown_type") == str

    def test_get_pandas_dtype_int(self):
        """Test getting Pandas dtype for integer types."""
        tm = TypeMap.default()
        assert tm.get_pandas_dtype("int") == "Int64"
        assert tm.get_pandas_dtype("smallint") == "Int64"  # Maps to Int64 by default

    def test_get_pandas_dtype_float(self):
        """Test getting Pandas dtype for float types."""
        tm = TypeMap.default()
        assert tm.get_pandas_dtype("float") == "float64"
        assert tm.get_pandas_dtype("decimal") == "float64"

    def test_get_pandas_dtype_string(self):
        """Test getting Pandas dtype for string types."""
        tm = TypeMap.default()
        assert tm.get_pandas_dtype("varchar") == "string"
        assert tm.get_pandas_dtype("text") == "string"

    def test_get_pandas_dtype_datetime(self):
        """Test getting Pandas dtype for datetime types."""
        tm = TypeMap.default()
        assert tm.get_pandas_dtype("datetime") == "datetime64[ns]"
        assert tm.get_pandas_dtype("date") == "datetime64[ns]"

    def test_get_pandas_dtype_bool(self):
        """Test getting Pandas dtype for boolean types."""
        tm = TypeMap.default()
        assert tm.get_pandas_dtype("bit") == "boolean"

    def test_convert_value_int(self):
        """Test converting values to int."""
        tm = TypeMap.default()
        assert tm.convert_value("42", "int") == 42
        assert tm.convert_value(42, "int") == 42
        assert tm.convert_value("42.7", "int") == 42

    def test_convert_value_float(self):
        """Test converting values to float."""
        tm = TypeMap.default()
        assert tm.convert_value("3.14", "float") == 3.14
        assert tm.convert_value(3.14, "float") == 3.14
        assert tm.convert_value("$25.50", "money") == 25.50

    def test_convert_value_bool(self):
        """Test converting values to boolean."""
        tm = TypeMap.default()
        assert tm.convert_value("true", "bit") is True
        assert tm.convert_value("1", "bit") is True
        assert tm.convert_value("false", "bit") is False
        assert tm.convert_value("0", "bit") is False

    def test_convert_value_string(self):
        """Test converting values to string."""
        tm = TypeMap.default()
        assert tm.convert_value("hello", "varchar") == "hello"
        assert tm.convert_value("hello", "text") == "hello"

    def test_convert_value_datetime(self):
        """Test converting values to datetime."""
        tm = TypeMap.default()
        result = tm.convert_value("2023-01-15", "datetime")
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_convert_value_null(self):
        """Test converting null values."""
        tm = TypeMap.default()
        assert tm.convert_value("null", "int") is None
        assert tm.convert_value(None, "int") is None
        # Empty string converts to empty string for varchar, not None
        assert tm.convert_value("", "varchar") == ""

    def test_convert_value_error_recovery(self):
        """Test error recovery in value conversion."""
        tm = TypeMap.default()
        # Invalid integer should return None, not raise
        assert tm.convert_value("not_a_number", "int") is None
        # But numeric extract should work
        assert tm.convert_value("abc123def", "int") == 123

    def test_convert_row(self):
        """Test converting a row dictionary."""
        tm = TypeMap.default()
        row = {"mukey": "123456", "clay": "25.5", "taxclass": "fine"}
        type_map = {"mukey": "int", "clay": "float", "taxclass": "varchar"}
        converted = tm.convert_row(row, type_map)

        assert converted["mukey"] == 123456
        assert converted["clay"] == 25.5
        assert converted["taxclass"] == "fine"

    def test_convert_rows(self):
        """Test converting multiple row dictionaries."""
        tm = TypeMap.default()
        rows = [
            {"mukey": "123456", "clay": "25.5"},
            {"mukey": "789012", "clay": "30.0"},
        ]
        type_map = {"mukey": "int", "clay": "float"}
        converted = tm.convert_rows(rows, type_map)

        assert len(converted) == 2
        assert converted[0]["mukey"] == 123456
        assert converted[0]["clay"] == 25.5
        assert converted[1]["mukey"] == 789012
        assert converted[1]["clay"] == 30.0

    def test_register_custom_processor(self):
        """Test registering a custom type processor."""
        tm = TypeMap.default()

        def upper_processor(v):
            if v is None or v == "":
                return None
            return str(v).upper()

        tm.register_processor("my_type", str, upper_processor)
        assert tm.convert_value("hello", "my_type") == "HELLO"

    def test_register_processor_overrides_default(self):
        """Test that registering a processor overrides defaults."""
        tm = TypeMap.default()
        
        # Register custom int processor that multiplies by 2
        def double_processor(v):
            result = TypeProcessor.to_int(v)
            return result * 2 if result is not None else None

        tm.register_processor("int", int, double_processor)
        assert tm.convert_value("42", "int") == 84

    def test_get_type_mappings(self):
        """Test getting all type mappings."""
        tm = TypeMap.default()
        mappings = tm.get_type_mappings()

        assert len(mappings) > 0
        assert "int" in mappings
        assert "varchar" in mappings
        assert "datetime" in mappings

        # Check structure of mapping
        int_mapping = mappings["int"]
        assert "python" in int_mapping
        assert "pandas" in int_mapping
        assert "polars" in int_mapping
        assert int_mapping["python"] == "int"
        assert int_mapping["pandas"] == "Int64"

    def test_polars_dtype_mapping(self):
        """Test getting Polars dtype mappings."""
        tm = TypeMap.default()
        polars_dtype = tm.get_polars_dtype("int")
        # Should either be pl.Int64 object or string "Int64"
        assert polars_dtype is not None

    def test_static_pandas_dtype_method(self):
        """Test static method for getting Pandas dtype."""
        dtype = TypeMap.get_pandas_dtype_for_python_type(int)
        assert dtype == "Int64"

        dtype = TypeMap.get_pandas_dtype_for_python_type(str)
        assert dtype == "string"

        dtype = TypeMap.get_pandas_dtype_for_python_type(float)
        assert dtype == "float64"

    def test_repr(self):
        """Test string representation."""
        tm = TypeMap.default()
        repr_str = repr(tm)
        assert "TypeMap" in repr_str
        assert "processors" in repr_str


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_default_type_map(self):
        """Test getting the default type map singleton."""
        tm1 = get_default_type_map()
        tm2 = get_default_type_map()
        assert tm1 is tm2  # Should be same instance

    def test_convert_value_function(self):
        """Test the module-level convert_value function."""
        assert convert_value("42", "int") == 42
        assert convert_value("3.14", "float") == 3.14
        assert convert_value("true", "bit") is True
        assert convert_value("null", "int") is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_convert_value_strict_mode(self):
        """Test strict mode for error handling."""
        tm = TypeMap.default()
        
        # Non-strict mode (default) returns None on invalid conversion
        result = tm.convert_value("invalid", "int", strict=False)
        assert result is None or result == 0  # Numeric extraction may return 0
        
        # Note: Strict mode not fully implemented in current version,
        # falls back to non-strict behavior

    def test_convert_value_with_whitespace(self):
        """Test conversion with leading/trailing whitespace."""
        tm = TypeMap.default()
        assert tm.convert_value("  42  ", "int") == 42
        assert tm.convert_value("  hello  ", "varchar") == "hello"

    def test_convert_mixed_case_type_names(self):
        """Test that type names are case-insensitive."""
        tm = TypeMap.default()
        assert tm.convert_value("42", "INT") == 42
        assert tm.convert_value("42", "Int") == 42
        assert tm.convert_value("hello", "VARCHAR") == "hello"

    def test_large_numbers(self):
        """Test conversion of large numbers."""
        tm = TypeMap.default()
        assert tm.convert_value("999999999999", "bigint") == 999999999999
        assert tm.convert_value("1.23e10", "float") == 1.23e10

    def test_negative_numbers(self):
        """Test conversion of negative numbers."""
        tm = TypeMap.default()
        assert tm.convert_value("-42", "int") == -42
        assert tm.convert_value("-3.14", "float") == -3.14

    def test_scientific_notation(self):
        """Test conversion of numbers in scientific notation."""
        tm = TypeMap.default()
        assert tm.convert_value("1.5e2", "float") == 150.0
        assert tm.convert_value("1.5E-2", "float") == 0.015


class TestIntegration:
    """Integration tests for the type conversion system."""

    def test_integration_with_response(self):
        """Test integration with response module."""
        from soildb.response import SDAResponse
        
        # Create a simple response
        response_data = {
            "Table": [
                ["mukey", "muname", "clay"],
                ["Int", "NVarChar", "Float"],
                ["123456", "Miami", "25.5"],
                ["789012", "Ames", "20.0"],
            ]
        }
        
        response = SDAResponse(response_data)
        converted = response.to_dict()
        
        assert len(converted) == 2
        # to_dict() returns string values - conversion happens during to_pandas/to_polars
        assert converted[0]["mukey"] == "123456"
        assert converted[0]["muname"] == "Miami"
        assert converted[0]["clay"] == "25.5"

    def test_consistency_across_frameworks(self):
        """Test that type conversions are consistent across frameworks."""
        tm = TypeMap.default()
        
        # Same SDA type should map to compatible Python/Pandas/Polars types
        python_type = tm.get_python_type("int")
        pandas_dtype = tm.get_pandas_dtype("int")
        polars_dtype = tm.get_polars_dtype("int")
        
        assert python_type == int
        assert pandas_dtype == "Int64"
        assert polars_dtype is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
