"""
XML metadata parsing for SSURGO survey area metadata from sacatalog.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

from .exceptions import SoilDBError


class MetadataParseError(SoilDBError):
    """Raised when XML metadata parsing fails."""

    pass


class SurveyMetadata:
    """
    Represents parsed metadata for a soil survey area.

    Parses FGDC-compliant XML metadata from sacatalog.fgdcmetadata column.
    Provides access to survey information including title, publication details,
    spatial bounds, keywords, and contact information.

    Example:
        >>> # Get metadata for a survey area
        >>> xml_data = client.get_sacatalog(areasymbol='IA015')['fgdcmetadata'].iloc[0]
        >>> metadata = SurveyMetadata(xml_data, areasymbol='IA015')
        >>>
        >>> # Access basic information
        >>> print(f"Title: {metadata.title}")
        >>> print(f"Published: {metadata.publication_date}")
        >>> print(f"Publisher: {metadata.publisher}")
        >>>
        >>> # Access spatial bounds
        >>> bounds = metadata.bounding_box
        >>> print(f"Covers area from {bounds['west']} to {bounds['east']}")
        >>>
        >>> # Access keywords and contact info
        >>> print(f"Keywords: {', '.join(metadata.keywords[:3])}")
        >>> print(f"Contact: {metadata.contact_person} ({metadata.contact_email})")
        >>>
        >>> # Convert to dictionary for further processing
        >>> data = metadata.to_dict()
    """

    def __init__(self, xml_content: str, areasymbol: Optional[str] = None):
        """
        Initialize from XML metadata content.

        Args:
            xml_content: Raw XML metadata string from fgdcmetadata column
            areasymbol: Optional area symbol for reference
        """
        self.areasymbol = areasymbol
        self.raw_xml = xml_content
        self._root: Optional[ET.Element] = None
        self._parse_xml()

    def _parse_xml(self) -> None:
        """Parse the XML content."""
        try:
            self._root = ET.fromstring(self.raw_xml)
        except ET.ParseError as e:
            raise MetadataParseError(f"Failed to parse XML metadata: {e}") from e

    def _find_text(self, xpath: str, default: Optional[str] = None) -> Optional[str]:
        """Find text content by XPath, return default if not found."""
        if self._root is None:
            return default
        element = self._root.find(xpath)
        return element.text.strip() if element is not None and element.text else default

    def _find_all_text(self, xpath: str) -> List[str]:
        """Find all text content matching XPath."""
        if self._root is None:
            return []
        elements = self._root.findall(xpath)
        return [
            elem.text.strip() for elem in elements if elem.text and elem.text.strip()
        ]

    @property
    def title(self) -> Optional[str]:
        """Survey area title from the metadata citation."""
        return self._find_text(".//idinfo/citation/citeinfo/title")

    @property
    def publication_date(self) -> Optional[str]:
        """Publication date from citation in YYYYMMDD format."""
        return self._find_text(".//idinfo/citation/citeinfo/pubdate")

    @property
    def publication_date_parsed(self) -> Optional[datetime]:
        """Publication date as datetime object."""
        pub_date = self.publication_date
        if not pub_date:
            return None
        try:
            return datetime.strptime(pub_date, "%Y%m%d")
        except ValueError:
            try:
                return datetime.strptime(pub_date, "%Y-%m-%d")
            except ValueError:
                return None

    @property
    def publisher(self) -> Optional[str]:
        """Publisher organization."""
        return self._find_text(".//idinfo/citation/citeinfo/pubinfo/publish")

    @property
    def publish_place(self) -> Optional[str]:
        """Publication place."""
        return self._find_text(".//idinfo/citation/citeinfo/pubinfo/pubplace")

    @property
    def origin(self) -> Optional[str]:
        """Originating organization."""
        return self._find_text(".//idinfo/citation/citeinfo/origin")

    @property
    def abstract(self) -> Optional[str]:
        """Survey abstract/description."""
        return self._find_text(".//idinfo/descript/abstract")

    @property
    def purpose(self) -> Optional[str]:
        """Survey purpose."""
        return self._find_text(".//idinfo/descript/purpose")

    @property
    def online_link(self) -> Optional[str]:
        """Online link to data."""
        return self._find_text(".//idinfo/citation/citeinfo/onlink")

    @property
    def access_constraints(self) -> Optional[str]:
        """Access constraints."""
        return self._find_text(".//idinfo/accconst")

    @property
    def use_constraints(self) -> Optional[str]:
        """Use constraints."""
        return self._find_text(".//idinfo/useconst")

    @property
    def contact_organization(self) -> Optional[str]:
        """Primary contact organization."""
        return self._find_text(".//idinfo/ptcontac/cntinfo/cntorgp/cntorg")

    @property
    def contact_person(self) -> Optional[str]:
        """Primary contact person."""
        return self._find_text(".//idinfo/ptcontac/cntinfo/cntorgp/cntper")

    @property
    def contact_position(self) -> Optional[str]:
        """Primary contact position."""
        return self._find_text(".//idinfo/ptcontac/cntinfo/cntpos")

    @property
    def contact_email(self) -> Optional[str]:
        """Primary contact email."""
        return self._find_text(".//idinfo/ptcontac/cntinfo/cntemail")

    @property
    def contact_phone(self) -> Optional[str]:
        """Primary contact phone."""
        return self._find_text(".//idinfo/ptcontac/cntinfo/cntvoice")

    @property
    def bounding_box(self) -> Dict[str, Optional[float]]:
        """Geographic bounding box coordinates in decimal degrees (WGS84)."""
        return {
            "west": self._parse_float(
                self._find_text(".//idinfo/spdom/bounding/westbc")
            ),
            "east": self._parse_float(
                self._find_text(".//idinfo/spdom/bounding/eastbc")
            ),
            "north": self._parse_float(
                self._find_text(".//idinfo/spdom/bounding/northbc")
            ),
            "south": self._parse_float(
                self._find_text(".//idinfo/spdom/bounding/southbc")
            ),
        }

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        """Parse string to float, return None if invalid."""
        if not value:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @property
    def keywords(self) -> List[str]:
        """Combined list of all theme and place keywords from the metadata."""
        theme_keywords = self._find_all_text(".//idinfo/keywords/theme/themekey")
        place_keywords = self._find_all_text(".//idinfo/keywords/place/placekey")
        return theme_keywords + place_keywords

    @property
    def theme_keywords(self) -> List[str]:
        """Theme-based keywords."""
        return self._find_all_text(".//idinfo/keywords/theme/themekey")

    @property
    def place_keywords(self) -> List[str]:
        """Place-based keywords."""
        return self._find_all_text(".//idinfo/keywords/place/placekey")

    @property
    def attribute_accuracy(self) -> Optional[str]:
        """Attribute accuracy information."""
        return self._find_text(".//dataqual/attracc/attraccr")

    @property
    def logical_consistency(self) -> Optional[str]:
        """Logical consistency information."""
        return self._find_text(".//dataqual/logic")

    @property
    def completeness(self) -> Optional[str]:
        """Completeness information."""
        return self._find_text(".//dataqual/complete")

    @property
    def horizontal_accuracy(self) -> Optional[str]:
        """Horizontal positional accuracy."""
        return self._find_text(".//dataqual/posacc/horizpa/horizpar")

    @property
    def scale_denominator(self) -> Optional[int]:
        """Map scale denominator."""
        scale_text = self._find_text(".//idinfo/citation/citeinfo/geoform")
        if not scale_text:
            return None
        # Try to extract scale from text like "1:24000"
        match = re.search(r"1:(\d+)", scale_text)
        if match:
            return int(match.group(1))
        return None

    @property
    def coordinate_system(self) -> Optional[str]:
        """Coordinate system name."""
        return self._find_text(".//spref/horizsys/geograph/geogunit")

    @property
    def datum(self) -> Optional[str]:
        """Geodetic datum."""
        return self._find_text(".//spref/horizsys/geodetic/horizdn")

    @property
    def ellipsoid(self) -> Optional[str]:
        """Reference ellipsoid."""
        return self._find_text(".//spref/horizsys/geodetic/ellips")

    def get_process_steps(self) -> List[Dict[str, Optional[str]]]:
        """Get all processing steps with dates and descriptions."""
        if self._root is None:
            return []

        steps = []
        for step_elem in self._root.findall(".//lineage/procstep"):
            step_info = {}

            # Get process description
            procdesc_elem = step_elem.find("procdesc")
            step_info["description"] = (
                procdesc_elem.text.strip()
                if procdesc_elem is not None and procdesc_elem.text
                else None
            )

            # Get process date
            procdate_elem = step_elem.find("procdate")
            step_info["date"] = (
                procdate_elem.text.strip()
                if procdate_elem is not None and procdate_elem.text
                else None
            )

            # Get source used
            srcused_elem = step_elem.find("srcused")
            step_info["source_used"] = (
                srcused_elem.text.strip()
                if srcused_elem is not None and srcused_elem.text
                else None
            )

            steps.append(step_info)

        return steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format.

        Returns:
            Dict containing all metadata fields including title, publication info,
            spatial bounds, keywords, contact details, and processing information.
        """
        bbox = self.bounding_box

        return {
            "areasymbol": self.areasymbol,
            "title": self.title,
            "publication_date": self.publication_date,
            "publication_date_parsed": (
                self.publication_date_parsed.isoformat()
                if self.publication_date_parsed
                else None
            ),
            "publisher": self.publisher,
            "publish_place": self.publish_place,
            "origin": self.origin,
            "abstract": self.abstract,
            "purpose": self.purpose,
            "online_link": self.online_link,
            "access_constraints": self.access_constraints,
            "use_constraints": self.use_constraints,
            "contact_organization": self.contact_organization,
            "contact_person": self.contact_person,
            "contact_position": self.contact_position,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "bounding_box": bbox,
            "keywords": self.keywords,
            "theme_keywords": self.theme_keywords,
            "place_keywords": self.place_keywords,
            "attribute_accuracy": self.attribute_accuracy,
            "logical_consistency": self.logical_consistency,
            "completeness": self.completeness,
            "horizontal_accuracy": self.horizontal_accuracy,
            "scale_denominator": self.scale_denominator,
            "coordinate_system": self.coordinate_system,
            "datum": self.datum,
            "ellipsoid": self.ellipsoid,
            "process_steps": self.get_process_steps(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SurveyMetadata(areasymbol='{self.areasymbol}', title='{self.title}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"Survey Metadata: {self.areasymbol}"]
        lines.append(f"Title: {self.title}")
        lines.append(f"Publication Date: {self.publication_date}")
        lines.append(f"Publisher: {self.publisher}")

        bbox = self.bounding_box
        if any(bbox.values()):
            lines.append(
                f"Bounds: ({bbox['west']}, {bbox['south']}) to ({bbox['east']}, {bbox['north']})"
            )

        if self.keywords:
            lines.append(
                f"Keywords: {', '.join(self.keywords[:5])}{'...' if len(self.keywords) > 5 else ''}"
            )

        return "\n".join(lines)


def parse_survey_metadata(
    xml_content: str, areasymbol: Optional[str] = None
) -> SurveyMetadata:
    """
    Parse XML metadata from sacatalog.fgdcmetadata column.

    Args:
        xml_content: Raw XML metadata string from fgdcmetadata column
        areasymbol: Optional area symbol for reference

    Returns:
        SurveyMetadata object with parsed information

    Raises:
        MetadataParseError: If XML parsing fails

    Example:
        >>> xml_data = client.get_sacatalog(areasymbol='IA015')['fgdcmetadata'].iloc[0]
        >>> metadata = parse_survey_metadata(xml_data, areasymbol='IA015')
        >>> print(metadata.title)
        Soil Survey Geographic (SSURGO) Database for Iowa County, Iowa
    """
    return SurveyMetadata(xml_content, areasymbol)


def extract_metadata_summary(xml_content: str) -> Dict[str, Any]:
    """
    Extract a summary of key metadata fields.

    Args:
        xml_content: Raw XML metadata string

    Returns:
        Dictionary with key metadata fields including title, publication_date,
        publisher, bounding_box, keywords_count, contact_email, and abstract_length

    Example:
        >>> xml_data = client.get_sacatalog(areasymbol='CA077')['fgdcmetadata'].iloc[0]
        >>> summary = extract_metadata_summary(xml_data)
        >>> print(f"Title: {summary['title']}")
        >>> print(f"Keywords: {summary['keywords_count']}")
        >>> print(f"Bounds: {summary['bounding_box']}")
    """
    try:
        metadata = parse_survey_metadata(xml_content)
        return {
            "title": metadata.title,
            "publication_date": metadata.publication_date,
            "publisher": metadata.publisher,
            "bounding_box": metadata.bounding_box,
            "keywords_count": len(metadata.keywords),
            "contact_email": metadata.contact_email,
            "abstract_length": len(metadata.abstract) if metadata.abstract else 0,
        }
    except MetadataParseError:
        return {"error": "Failed to parse metadata"}
