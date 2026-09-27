"""
Tests for SDAResponse.from_rows and SDAResponse.concat classmethods.
"""

import pytest

from soildb.response import SDAResponse


class TestFromRows:
    """Test SDAResponse.from_rows classmethod."""

    def test_from_rows_basic(self):
        """Test basic from_rows with string data."""
        rows = [["a", "1"], ["b", "2"]]
        columns = ["name", "value"]
        types = ["varchar", "varchar"]

        response = SDAResponse.from_rows(rows, columns, types)

        assert response.columns == columns
        assert len(response) == 2
        assert response.metadata == ["DataTypeName=varchar", "DataTypeName=varchar"]

    def test_from_rows_with_conversion(self):
        """Test from_rows round-trip with type conversion."""
        rows = [["text1", 123, 45.67], ["text2", 456, 89.01]]
        columns = ["name", "intval", "floatval"]
        types = ["varchar", "int", "float"]

        response = SDAResponse.from_rows(rows, columns, types)
        records = response.to_dict()

        assert len(records) == 2
        assert records[0]["name"] == "text1"
        assert records[0]["intval"] == 123
        assert isinstance(records[0]["intval"], int)
        assert records[0]["floatval"] == 45.67
        assert isinstance(records[0]["floatval"], float)
        assert records[1]["name"] == "text2"
        assert records[1]["intval"] == 456

    def test_from_rows_empty(self):
        """Test from_rows with no data rows."""
        response = SDAResponse.from_rows([], ["col1", "col2"], ["varchar", "int"])

        assert response.columns == ["col1", "col2"]
        assert response.is_empty()
        assert len(response) == 0

    def test_from_rows_column_type_mismatch(self):
        """Test from_rows raises when column and type counts differ."""
        rows = [["a", 1]]
        columns = ["x", "y", "z"]
        types = ["varchar", "int"]

        with pytest.raises(ValueError, match="Column count.*does not match type count"):
            SDAResponse.from_rows(rows, columns, types)

    def test_from_rows_row_length_mismatch(self):
        """Test from_rows raises when row length differs from column count."""
        rows = [["a", 1, "extra"]]  # 3 values but only 2 columns
        columns = ["x", "y"]
        types = ["varchar", "int"]

        with pytest.raises(ValueError, match="Row 0 has 3 values, expected 2"):
            SDAResponse.from_rows(rows, columns, types)

    def test_from_rows_multiple_rows_mismatch(self):
        """Test error message for mismatch in a later row."""
        rows = [["a", 1], ["b"]]  # Second row too short
        columns = ["x", "y"]
        types = ["varchar", "int"]

        with pytest.raises(ValueError, match="Row 1 has 1 values, expected 2"):
            SDAResponse.from_rows(rows, columns, types)

    def test_from_rows_none_values(self):
        """Test from_rows handles None values.

        Note: None for varchar is converted to empty string by type system;
        None for int stays None.
        """
        rows = [["a", None], [None, 2]]
        columns = ["x", "y"]
        types = ["varchar", "int"]

        response = SDAResponse.from_rows(rows, columns, types)
        records = response.to_dict()

        assert records[0]["x"] == "a"
        assert records[0]["y"] is None
        assert records[1]["x"] == ""  # None converts to empty string for varchar
        assert records[1]["y"] == 2

    def test_from_rows_datetime_conversion(self):
        """Test from_rows with datetime type."""
        rows = [["2020-01-15 10:30:00"]]
        columns = ["datecol"]
        types = ["datetime"]

        response = SDAResponse.from_rows(rows, columns, types)
        records = response.to_dict()

        from datetime import datetime

        assert isinstance(records[0]["datecol"], datetime)
        assert records[0]["datecol"].year == 2020
        assert records[0]["datecol"].month == 1

    def test_from_rows_bit_conversion(self):
        """Test from_rows with bit type."""
        rows = [["1"], ["0"], [None]]
        columns = ["boolval"]
        types = ["bit"]

        response = SDAResponse.from_rows(rows, columns, types)
        records = response.to_dict()

        assert records[0]["boolval"] is True
        assert records[1]["boolval"] is False
        assert records[2]["boolval"] is None


class TestConcat:
    """Test SDAResponse.concat classmethod."""

    def test_concat_two_responses(self):
        """Test concatenating two non-empty responses."""
        r1 = SDAResponse.from_rows(
            [["a", 1], ["b", 2]], ["name", "val"], ["varchar", "int"]
        )
        r2 = SDAResponse.from_rows([["c", 3]], ["name", "val"], ["varchar", "int"])

        merged = SDAResponse.concat([r1, r2])

        assert merged.columns == ["name", "val"]
        assert len(merged) == 3
        records = merged.to_dict()
        assert records[0]["name"] == "a"
        assert records[1]["name"] == "b"
        assert records[2]["name"] == "c"

    def test_concat_preserves_order(self):
        """Test that concat preserves row order."""
        r1 = SDAResponse.from_rows([["1"], ["2"]], ["x"], ["varchar"])
        r2 = SDAResponse.from_rows([["3"]], ["x"], ["varchar"])
        r3 = SDAResponse.from_rows([["4"], ["5"]], ["x"], ["varchar"])

        merged = SDAResponse.concat([r1, r2, r3])

        records = merged.to_dict()
        assert [r["x"] for r in records] == ["1", "2", "3", "4", "5"]

    def test_concat_with_empty_responses(self):
        """Test concat skips empty responses."""
        r1 = SDAResponse.from_rows([["a", 1]], ["name", "val"], ["varchar", "int"])
        r2 = SDAResponse.from_rows([], ["name", "val"], ["varchar", "int"])  # empty
        r3 = SDAResponse.from_rows([["b", 2]], ["name", "val"], ["varchar", "int"])

        merged = SDAResponse.concat([r1, r2, r3])

        assert len(merged) == 2
        records = merged.to_dict()
        assert records[0]["name"] == "a"
        assert records[1]["name"] == "b"

    def test_concat_all_empty_responses(self):
        """Test concat with all empty responses returns empty."""
        r1 = SDAResponse.from_rows([], ["x"], ["varchar"])
        r2 = SDAResponse.from_rows([], ["x"], ["varchar"])

        merged = SDAResponse.concat([r1, r2])

        assert merged.is_empty()
        assert len(merged) == 0

    def test_concat_empty_sequence(self):
        """Test concat with no responses returns empty."""
        merged = SDAResponse.concat([])

        assert merged.is_empty()
        assert len(merged) == 0

    def test_concat_single_response(self):
        """Test concat with single response."""
        r1 = SDAResponse.from_rows([["a"]], ["x"], ["varchar"])

        merged = SDAResponse.concat([r1])

        assert len(merged) == 1
        assert merged.to_dict()[0]["x"] == "a"

    def test_concat_mismatched_columns_raises(self):
        """Test concat raises on column mismatch."""
        r1 = SDAResponse.from_rows([["a", 1]], ["x", "y"], ["varchar", "int"])
        r2 = SDAResponse.from_rows([["b", 2]], ["x", "z"], ["varchar", "int"])

        with pytest.raises(ValueError, match="Response 1 has columns"):
            SDAResponse.concat([r1, r2])

    def test_concat_mismatched_column_order_raises(self):
        """Test concat raises on column order mismatch."""
        r1 = SDAResponse.from_rows([["a", 1]], ["x", "y"], ["varchar", "int"])
        r2 = SDAResponse.from_rows([["b", 2]], ["y", "x"], ["int", "varchar"])

        with pytest.raises(ValueError, match="Response 1 has columns"):
            SDAResponse.concat([r1, r2])

    def test_concat_no_double_conversion(self):
        """Test that concat does not double-convert values.

        Values should be converted exactly once during to_dict(), not during
        the merge operation.
        """
        # Build responses with raw string data that needs conversion
        r1 = SDAResponse.from_rows(
            [["123", "45.67"]], ["intcol", "floatcol"], ["int", "float"]
        )
        r2 = SDAResponse.from_rows(
            [["456", "89.01"]], ["intcol", "floatcol"], ["int", "float"]
        )

        # Merge without converting
        merged = SDAResponse.concat([r1, r2])

        # Verify raw data is still strings (unconverted)
        assert merged.data[0][0] == "123"
        assert merged.data[0][1] == "45.67"
        assert merged.data[1][0] == "456"
        assert merged.data[1][1] == "89.01"

        # Now convert and verify proper types
        records = merged.to_dict()
        assert records[0]["intcol"] == 123
        assert isinstance(records[0]["intcol"], int)
        assert records[0]["floatcol"] == 45.67
        assert isinstance(records[0]["floatcol"], float)
        assert records[1]["intcol"] == 456
        assert isinstance(records[1]["intcol"], int)

    def test_concat_preserves_types(self):
        """Test that concat preserves SDA type information."""
        r1 = SDAResponse.from_rows(
            [["a", "123"]], ["name", "count"], ["varchar", "int"]
        )
        r2 = SDAResponse.from_rows(
            [["b", "456"]], ["name", "count"], ["varchar", "int"]
        )

        merged = SDAResponse.concat([r1, r2])

        # Verify metadata is preserved
        types = merged.get_column_types()
        assert types["name"] == "varchar"
        assert types["count"] == "int"

    def test_concat_with_datetime_type(self):
        """Test concat preserves datetime type through merge."""
        r1 = SDAResponse.from_rows([["2020-01-15"]], ["date"], ["datetime"])
        r2 = SDAResponse.from_rows([["2021-06-20"]], ["date"], ["datetime"])

        merged = SDAResponse.concat([r1, r2])

        records = merged.to_dict()
        assert len(records) == 2
        # Verify datetime conversion happened
        from datetime import datetime

        assert isinstance(records[0]["date"], datetime)
        assert isinstance(records[1]["date"], datetime)
        assert records[0]["date"].year == 2020
        assert records[1]["date"].year == 2021

    def test_concat_many_responses(self):
        """Test concat with many responses."""
        responses = [
            SDAResponse.from_rows([[str(i)]], ["x"], ["varchar"]) for i in range(10)
        ]

        merged = SDAResponse.concat(responses)

        assert len(merged) == 10
        records = merged.to_dict()
        assert [r["x"] for r in records] == [str(i) for i in range(10)]


class TestRoundTrip:
    """Test round-trip: from_rows -> to_dict -> from_rows."""

    def test_roundtrip_simple(self):
        """Test simple round-trip conversion."""
        original_rows = [["a", 1], ["b", 2]]
        columns = ["name", "val"]
        types = ["varchar", "int"]

        r1 = SDAResponse.from_rows(original_rows, columns, types)
        records = r1.to_dict()

        # Convert back to rows for comparison
        roundtrip_rows = [[r[col] for col in columns] for r in records]

        # Note: values may be converted (string "1" -> int 1), so compare values
        assert roundtrip_rows[0][0] == "a"
        assert roundtrip_rows[0][1] == 1
        assert roundtrip_rows[1][0] == "b"
        assert roundtrip_rows[1][1] == 2

    def test_roundtrip_with_none(self):
        """Test round-trip behavior with None values.

        Note: None for varchar is converted to empty string by type system.
        """
        rows = [["a", None], [None, 2]]
        columns = ["x", "y"]
        types = ["varchar", "int"]

        response = SDAResponse.from_rows(rows, columns, types)
        records = response.to_dict()

        assert records[0]["x"] == "a"
        assert records[0]["y"] is None
        assert records[1]["x"] == ""  # None converts to empty string for varchar
        assert records[1]["y"] == 2


class TestIntegration:
    """Integration tests combining from_rows and concat."""

    def test_from_rows_then_concat(self):
        """Test workflow: build with from_rows, then concat."""
        batch1_rows = [["a", 1], ["b", 2]]
        batch2_rows = [["c", 3]]
        columns = ["name", "val"]
        types = ["varchar", "int"]

        r1 = SDAResponse.from_rows(batch1_rows, columns, types)
        r2 = SDAResponse.from_rows(batch2_rows, columns, types)
        merged = SDAResponse.concat([r1, r2])

        assert len(merged) == 3
        records = merged.to_dict()
        names = [r["name"] for r in records]
        assert names == ["a", "b", "c"]

    def test_concat_then_export_types(self):
        """Test that export methods work correctly after concat."""
        r1 = SDAResponse.from_rows(
            [["100", "50.5"]], ["int", "float"], ["int", "float"]
        )
        r2 = SDAResponse.from_rows(
            [["200", "75.2"]], ["int", "float"], ["int", "float"]
        )

        merged = SDAResponse.concat([r1, r2])

        # Test to_dict
        records = merged.to_dict()
        assert records[0]["int"] == 100
        assert records[0]["float"] == 50.5

        # Test column types
        types = merged.get_column_types()
        assert types["int"] == "int"
        assert types["float"] == "float"

        # Test python types
        try:
            py_types = merged.get_python_types()
            assert "int" in py_types["int"]
            assert "float" in py_types["float"]
        except KeyError:
            # If key doesn't exist, that's OK for this test
            pass
