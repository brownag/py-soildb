"""
Tests for the metadata parsing module.
"""

import pytest

from soildb.metadata import (
    MetadataParseError,
    SurveyMetadata,
    extract_metadata_summary,
    parse_survey_metadata,
)


class TestSurveyMetadata:
    """Test SurveyMetadata class."""

    def test_init_with_valid_xml(self):
        """Test initialization with valid XML."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Soil Survey Geographic (SSURGO) Database for Test County, California</title>
                        <pubdate>20230315</pubdate>
                        <origin>U.S. Department of Agriculture, Natural Resources Conservation Service</origin>
                    </citeinfo>
                </citation>
                <descript>
                    <abstract>This is a test soil survey abstract.</abstract>
                    <purpose>Testing purposes only.</purpose>
                </descript>
                <spdom>
                    <bounding>
                        <westbc>-120.5</westbc>
                        <eastbc>-119.5</eastbc>
                        <northbc>38.5</northbc>
                        <southbc>37.5</southbc>
                    </bounding>
                </spdom>
                <keywords>
                    <theme>
                        <themekey>soil</themekey>
                        <themekey>agriculture</themekey>
                    </theme>
                    <place>
                        <placekey>California</placekey>
                        <placekey>Test County</placekey>
                    </place>
                </keywords>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(xml_content, "CA123")

        assert metadata.areasymbol == "CA123"
        assert (
            metadata.title
            == "Soil Survey Geographic (SSURGO) Database for Test County, California"
        )
        assert metadata.publication_date == "20230315"
        assert metadata.abstract == "This is a test soil survey abstract."
        assert metadata.purpose == "Testing purposes only."

        bbox = metadata.bounding_box
        assert bbox["west"] == -120.5
        assert bbox["east"] == -119.5
        assert bbox["north"] == 38.5
        assert bbox["south"] == 37.5

        assert "soil" in metadata.keywords
        assert "California" in metadata.keywords
        assert len(metadata.theme_keywords) == 2
        assert len(metadata.place_keywords) == 2

    def test_init_with_invalid_xml(self):
        """Test initialization with invalid XML."""
        invalid_xml = "This is not XML"

        with pytest.raises(MetadataParseError):
            SurveyMetadata(invalid_xml)

    def test_publication_date_parsing(self):
        """Test publication date parsing."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <pubdate>20230315</pubdate>
                    </citeinfo>
                </citation>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(xml_content)
        parsed_date = metadata.publication_date_parsed

        assert parsed_date is not None
        assert parsed_date.year == 2023
        assert parsed_date.month == 3
        assert parsed_date.day == 15

    def test_publication_date_parsing_alternative_format(self):
        """Test publication date parsing with alternative format."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <pubdate>2023-03-15</pubdate>
                    </citeinfo>
                </citation>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(xml_content)
        parsed_date = metadata.publication_date_parsed

        assert parsed_date is not None
        assert parsed_date.year == 2023
        assert parsed_date.month == 3
        assert parsed_date.day == 15

    def test_missing_elements(self):
        """Test handling of missing XML elements."""
        minimal_xml = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Test Title</title>
                    </citeinfo>
                </citation>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(minimal_xml)

        assert metadata.title == "Test Title"
        assert metadata.publication_date is None
        assert metadata.abstract is None
        assert metadata.contact_email is None
        assert metadata.bounding_box == {
            "west": None,
            "east": None,
            "north": None,
            "south": None,
        }

    def test_process_steps(self):
        """Test parsing of processing steps."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <dataqual>
                <lineage>
                    <procstep>
                        <procdesc>Initial data collection</procdesc>
                        <procdate>20230301</procdate>
                        <srcused>Field surveys</srcused>
                    </procstep>
                    <procstep>
                        <procdesc>Data processing and validation</procdesc>
                        <procdate>20230315</procdate>
                    </procstep>
                </lineage>
            </dataqual>
        </metadata>"""

        metadata = SurveyMetadata(xml_content)
        steps = metadata.get_process_steps()

        assert len(steps) == 2
        assert steps[0]["description"] == "Initial data collection"
        assert steps[0]["date"] == "20230301"
        assert steps[0]["source_used"] == "Field surveys"
        assert steps[1]["description"] == "Data processing and validation"
        assert steps[1]["date"] == "20230315"
        assert steps[1]["source_used"] is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Test Survey</title>
                        <pubdate>20230315</pubdate>
                    </citeinfo>
                </citation>
                <keywords>
                    <theme>
                        <themekey>soil</themekey>
                    </theme>
                </keywords>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(xml_content, "CA123")
        result = metadata.to_dict()

        assert result["areasymbol"] == "CA123"
        assert result["title"] == "Test Survey"
        assert result["publication_date"] == "20230315"
        assert result["publication_date_parsed"] == "2023-03-15T00:00:00"
        assert "keywords" in result
        assert "bounding_box" in result

    def test_string_representations(self):
        """Test string representations."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Test Survey</title>
                        <pubdate>20230315</pubdate>
                    </citeinfo>
                </citation>
            </idinfo>
        </metadata>"""

        metadata = SurveyMetadata(xml_content, "CA123")

        repr_str = repr(metadata)
        assert "SurveyMetadata" in repr_str
        assert "CA123" in repr_str
        assert "Test Survey" in repr_str

        str_str = str(metadata)
        assert "Survey Metadata: CA123" in str_str
        assert "Title: Test Survey" in str_str


class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_survey_metadata(self):
        """Test parse_survey_metadata function."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Test</title>
                    </citeinfo>
                </citation>
            </idinfo>
        </metadata>"""

        metadata = parse_survey_metadata(xml_content, "CA123")

        assert isinstance(metadata, SurveyMetadata)
        assert metadata.areasymbol == "CA123"
        assert metadata.title == "Test"

    def test_extract_metadata_summary(self):
        """Test extract_metadata_summary function."""
        xml_content = """<?xml version="1.0"?>
        <metadata>
            <idinfo>
                <citation>
                    <citeinfo>
                        <title>Test Survey</title>
                        <pubdate>20230315</pubdate>
                        <pubinfo>
                            <publish>NRCS</publish>
                        </pubinfo>
                    </citeinfo>
                </citation>
                <descript>
                    <abstract>This is a test abstract with some content.</abstract>
                </descript>
                <keywords>
                    <theme>
                        <themekey>soil</themekey>
                        <themekey>agriculture</themekey>
                    </theme>
                </keywords>
                <spdom>
                    <bounding>
                        <westbc>-120.0</westbc>
                        <eastbc>-119.0</eastbc>
                        <northbc>38.0</northbc>
                        <southbc>37.0</southbc>
                    </bounding>
                </spdom>
            </idinfo>
        </metadata>"""

        summary = extract_metadata_summary(xml_content)

        assert summary["title"] == "Test Survey"
        assert summary["publication_date"] == "20230315"
        assert summary["publisher"] == "NRCS"
        assert summary["keywords_count"] == 2
        assert summary["abstract_length"] > 0
        assert "bounding_box" in summary
        assert summary["bounding_box"]["west"] == -120.0

    def test_extract_metadata_summary_with_invalid_xml(self):
        """Test extract_metadata_summary with invalid XML."""
        invalid_xml = "Not XML"

        summary = extract_metadata_summary(invalid_xml)

        assert "error" in summary
        assert summary["error"] == "Failed to parse metadata"
