"""
Tests for SDA response handling with type conversion.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from soildb.exceptions import SDAResponseError
from soildb.response import SDAResponse

# Mock SDA response data for horizons
MOCK_HORIZON_DATA = {
    "Table": [
        ["chkey", "cokey", "hzdept_r", "hzdepb_r", "claytotal_r"],
        [
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=float",
        ],
        [1, 101, 0, 10, 25.0],
        [2, 101, 10, 30, 30.0],
        [3, 102, 0, 20, 15.0],
        [4, 102, 20, 50, 20.0],
    ]
}
MOCK_HORIZON_DATA_MISSING_COL = {
    "Table": [
        ["chkey", "cokey", "hzdepb_r", "claytotal_r"],  # Missing hzdept_r
        [
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=int",
            "DataTypeName=float",
        ],
        [1, 101, 10, 25.0],
        [2, 101, 30, 30.0],
    ]
}
MOCK_EMPTY_RESPONSE = {"Table": [["foo"], ["bar"]]}


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
        """Test that with convert_types=False, dtypes are not specifically converted."""
        response = SDAResponse.from_json(sample_sda_response_json)

        try:
            df = response.to_pandas(convert_types=False)
            # When type conversion is disabled, pandas will infer dtypes.
            # Since the raw data from to_dict() is already converted from strings,
            # we just check that the DataFrame is created.
            # All columns in sample_sda_response_json are strings, so they should be object/string.
            assert len(df) == 2
            assert all(pd.api.types.is_object_dtype(dtype) for dtype in df.dtypes)

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


@pytest.fixture
def mock_site_data():
    """Fixture for mock pandas DataFrame of site data."""
    return pd.DataFrame(
        {
            "cokey": [101, 102],
            "compname": ["CompA", "CompB"],
            "majcompflag": ["Yes", "Yes"],
        }
    )


class TestSoilProfileCollectionIntegration:
    """Tests for the to_soilprofilecollection method."""

    def test_to_soilprofilecollection_success(self):
        """Test successful conversion to SoilProfileCollection."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA)
        spc = response.to_soilprofilecollection()

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2  # 2 unique profiles (cokeys)
        assert len(spc.horizons) == 4
        assert spc.site.empty
        assert spc.hzidname == "chkey"
        assert spc.idname == "cokey"

    def test_to_soilprofilecollection_with_site_data(self, mock_site_data):
        """Test successful conversion with site data."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA)
        spc = response.to_soilprofilecollection(site_data=mock_site_data)

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2
        assert spc.site is not None
        assert len(spc.site) == 2
        assert "compname" in spc.site.columns

    def test_to_soilprofilecollection_missing_package(self, no_soilprofilecollection):
        """Test ImportError is raised if soilprofilecollection is not installed."""
        response = SDAResponse(MOCK_HORIZON_DATA)
        with pytest.raises(ImportError, match="pip install soildb\\[soil\\]"):
            response.to_soilprofilecollection()

    def test_to_soilprofilecollection_missing_required_columns(self):
        """Test SPCValidationError is raised if required columns are missing."""
        try:
            from soilprofilecollection import SoilProfileCollection  # noqa
            from soildb.spc_validator import SPCValidationError
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA_MISSING_COL)
        with pytest.raises(SPCValidationError):
            response.to_soilprofilecollection()

    def test_to_soilprofilecollection_custom_colnames(self, mock_site_data):
        """Test conversion with custom column names."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Create data with custom column names
        custom_data = json.loads(json.dumps(MOCK_HORIZON_DATA))  # Deep copy
        custom_data["Table"][0] = ["horizon_id", "site_id", "top", "bottom", "clay"]
        mock_site_data.rename(columns={"cokey": "site_id"}, inplace=True)

        response = SDAResponse(custom_data)
        spc = response.to_soilprofilecollection(
            site_data=mock_site_data,
            hz_id_col="horizon_id",
            site_id_col="site_id",
            hz_top_col="top",
            hz_bot_col="bottom",
        )

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2
        assert spc.hzidname == "horizon_id"
        assert spc.idname == "site_id"
        assert "compname" in spc.site.columns

    def test_to_soilprofilecollection_empty_response(self):
        """Test that an empty response raises SPCValidationError due to missing columns."""
        try:
            from soilprofilecollection import SoilProfileCollection  # noqa
            from soildb.spc_validator import SPCValidationError
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_EMPTY_RESPONSE)
        with pytest.raises(SPCValidationError):
            response.to_soilprofilecollection()

    def test_to_soilprofilecollection_with_preset_standard_sda(self):
        """Test conversion with preset='standard_sda'."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA)
        spc = response.to_soilprofilecollection(preset="standard_sda")

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2
        assert spc.idname == "cokey"
        assert spc.hzidname == "chkey"

    def test_to_soilprofilecollection_with_preset_lab_pedon(self):
        """Test conversion with preset='lab_pedon'."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Create lab pedon data with completely unique hzname values per profile
        # Note: In practice, hzname needs to be unique across the entire dataset
        lab_data = {
            "Table": [
                ["pedon_id", "hzname", "hzdept_r", "hzdepb_r", "claytotal_r"],
                [
                    "DataTypeName=varchar",
                    "DataTypeName=varchar",
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=float",
                ],
                ["P001", "hzname_1", 0, 10, 20.0],
                ["P001", "hzname_2", 10, 30, 30.0],
                ["P002", "hzname_3", 0, 15, 18.0],
            ]
        }
        response = SDAResponse(lab_data)
        spc = response.to_soilprofilecollection(preset="lab_pedon")

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2
        assert spc.idname == "pedon_id"
        assert spc.hzidname == "hzname"

    def test_to_soilprofilecollection_with_custom_preset(self):
        """Test conversion with CustomColumnConfig preset."""
        try:
            from soilprofilecollection import SoilProfileCollection

            from soildb.spc_presets import CustomColumnConfig
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Create data with custom column names
        custom_data = {
            "Table": [
                ["site_id", "horizon_id", "top_depth", "bottom_depth", "clay"],
                [
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=float",
                ],
                [101, 1, 0, 10, 25.0],
                [101, 2, 10, 30, 30.0],
                [102, 3, 0, 20, 15.0],
            ]
        }
        response = SDAResponse(custom_data)
        preset = CustomColumnConfig(
            site_id_col="site_id",
            horizon_id_col="horizon_id",
            horizon_top_col="top_depth",
            horizon_bottom_col="bottom_depth",
        )
        spc = response.to_soilprofilecollection(preset=preset)

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2
        assert spc.idname == "site_id"
        assert spc.hzidname == "horizon_id"

    def test_to_soilprofilecollection_with_depth_validation(self):
        """Test conversion with depth validation enabled."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA)
        spc = response.to_soilprofilecollection(validate_depths=True)

        assert isinstance(spc, SoilProfileCollection)
        assert len(spc) == 2

    def test_to_soilprofilecollection_with_invalid_depths(self):
        """Test conversion with invalid depth values raises error."""
        try:
            from soilprofilecollection import SoilProfileCollection  # noqa: F401

            from soildb.spc_validator import SPCValidationError
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Create data with invalid depth (top > bottom)
        invalid_data = {
            "Table": [
                ["chkey", "cokey", "hzdept_r", "hzdepb_r", "claytotal_r"],
                [
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=int",
                    "DataTypeName=float",
                ],
                [1, 101, 30, 10, 25.0],  # Invalid: top > bottom
                [2, 101, 10, 30, 30.0],
            ]
        }
        response = SDAResponse(invalid_data)

        # With validation enabled, should warn about invalid depth records and raise error
        with pytest.warns(
            UserWarning, match="Found 1 horizon records with invalid depth values"
        ):
            with pytest.raises(SPCValidationError, match="Depth validation failed"):
                response.to_soilprofilecollection(validate_depths=True)

    def test_to_soilprofilecollection_without_depth_validation(self):
        """Test conversion with depth validation disabled (but SPC still validates)."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Use valid data - note that SoilProfileCollection still validates internally
        response = SDAResponse(MOCK_HORIZON_DATA)

        # With validation disabled, we skip our validation but SPC still validates
        spc = response.to_soilprofilecollection(validate_depths=False)
        assert isinstance(spc, SoilProfileCollection)

    def test_validate_for_soilprofilecollection(self):
        """Test validate_for_soilprofilecollection returns validation report."""
        try:
            from soilprofilecollection import SoilProfileCollection  # noqa
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA)
        report = response.validate_for_soilprofilecollection()

        assert isinstance(report, dict)
        assert "data_summary" in report
        assert "validation_details" in report
        assert "errors" in report
        assert "warnings" in report

    def test_validate_for_soilprofilecollection_reports_missing_columns(self):
        """Test validate_for_soilprofilecollection detects missing columns."""
        try:
            from soilprofilecollection import SoilProfileCollection  # noqa
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        response = SDAResponse(MOCK_HORIZON_DATA_MISSING_COL)
        report = response.validate_for_soilprofilecollection()

        assert len(report["errors"]) > 0

    def test_backward_compatibility_old_parameters(self):
        """Test backward compatibility with original parameter names."""
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            pytest.skip("soilprofilecollection not installed")

        # Use original parameter names
        response = SDAResponse(MOCK_HORIZON_DATA)
        spc = response.to_soilprofilecollection(
            site_id_col="cokey",
            hz_id_col="chkey",
            hz_top_col="hzdept_r",
            hz_bot_col="hzdepb_r",
        )

        assert isinstance(spc, SoilProfileCollection)
        assert spc.idname == "cokey"
        assert spc.hzidname == "chkey"
