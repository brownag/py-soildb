"""
LDM (Lab Data Mart) table metadata and constants.

Provides table names, column mappings, and metadata for the Kellogg Soil Survey
Laboratory Data Mart (KSSL) accessible via Soil Data Access and SQLite snapshots.

Reference:
    https://jneme910.github.io/Lab_Data_Mart_Documentation/Documents/SDA_KSSL_Data_model.html
"""

from typing import Dict, List, Optional, Set

# ============================================================================
# LDM Table Names
# ============================================================================

# Core tables (always available)
PEDON_TABLE = "pedon"
SITE_TABLE = "site"
LAB_LAYER_TABLE = "lab_layer"

# Physical properties
LAB_PHYSICAL_PROPERTIES_TABLE = "lab_physical_properties"

# Chemical properties
LAB_CHEMICAL_PROPERTIES_TABLE = "lab_chemical_properties"

# Calculations and estimates
LAB_CALCULATIONS_TABLE = "lab_calculations_including_estimates_and_default_values"

# Rosetta Key data
LAB_ROSETTA_KEY_TABLE = "lab_rosetta_Key"

# Optional tables
LAB_MAJOR_TRACE_ELEMENTS_TABLE = "lab_major_and_trace_elements_and_oxides"
LAB_MINERALOGY_TABLE = "lab_mineralogy_glass_count_and_optical_properties"
LAB_MIR_TABLE = "lab_mir"
LAB_XRD_THERMAL_TABLE = "lab_xrd_and_thermal"


# ============================================================================
# Default Table Groups
# ============================================================================

DEFAULT_TABLES = [
    LAB_PHYSICAL_PROPERTIES_TABLE,
    LAB_CHEMICAL_PROPERTIES_TABLE,
    LAB_CALCULATIONS_TABLE,
    LAB_ROSETTA_KEY_TABLE,
]

OPTIONAL_TABLES = [
    LAB_MAJOR_TRACE_ELEMENTS_TABLE,
    LAB_MINERALOGY_TABLE,
    LAB_MIR_TABLE,
    LAB_XRD_THERMAL_TABLE,
]

ALL_TABLES = DEFAULT_TABLES + OPTIONAL_TABLES


# ============================================================================
# Table Groups by Category
# ============================================================================

PHYSICAL_PROPERTY_TABLES = [LAB_PHYSICAL_PROPERTIES_TABLE]

CHEMICAL_PROPERTY_TABLES = [LAB_CHEMICAL_PROPERTIES_TABLE]

CALCULATION_TABLES = [LAB_CALCULATIONS_TABLE]

SITE_PEDON_TABLES = [PEDON_TABLE, SITE_TABLE, LAB_LAYER_TABLE]

SPECTROSCOPY_TABLES = [LAB_MIR_TABLE]

MINERALOGY_TABLES = [LAB_MINERALOGY_TABLE]

TRACE_ELEMENT_TABLES = [LAB_MAJOR_TRACE_ELEMENTS_TABLE]

XRD_THERMAL_TABLES = [LAB_XRD_THERMAL_TABLE]


# ============================================================================
# Key Columns for Tables
# ============================================================================

TABLE_KEY_COLUMNS: Dict[str, str] = {
    # Core tables
    PEDON_TABLE: "pedon_key",
    SITE_TABLE: "site_key",
    LAB_LAYER_TABLE: "lab_layer_key",
    # Data tables
    LAB_PHYSICAL_PROPERTIES_TABLE: "lab_layer_key",
    LAB_CHEMICAL_PROPERTIES_TABLE: "lab_layer_key",
    LAB_CALCULATIONS_TABLE: "lab_layer_key",
    LAB_ROSETTA_KEY_TABLE: "lab_layer_key",
    LAB_MAJOR_TRACE_ELEMENTS_TABLE: "lab_layer_key",
    LAB_MINERALOGY_TABLE: "lab_layer_key",
    LAB_MIR_TABLE: "lab_layer_key",
    LAB_XRD_THERMAL_TABLE: "lab_layer_key",
}


# ============================================================================
# Valid Filtering Values
# ============================================================================

# Sample preparation codes
PREP_CODES = {
    "S": "Sieved",
    "D": "Dispersed",
    "C": "Crushed",
    "": "All prep codes",
}

# Analyzed size fractions
ANALYZED_SIZE_FRACTIONS = {
    "<2 mm": "Less than 2 mm (standard)",
    ">2 mm": "Greater than 2 mm",
    "2-5 mm": "2 to 5 mm",
    "": "All size fractions",
}

# Layer types (horizon classifications)
LAYER_TYPES = {
    "horizon": "Standard horizon",
    "layer": "Custom layer",
    "reporting layer": "Reporting layer",
}

# Area types (geographic classifications)
AREA_TYPES = {
    "ssa": "Soil Survey Area",
    "country": "Country",
    "state": "State",
    "county": "County",
    "mlra": "Major Land Resource Area",
    "nforest": "National Forest",
    "npark": "National Park",
}


# ============================================================================
# Common Query Columns
# ============================================================================

# Pedon and site identifier columns
ID_COLUMNS = {
    "pedlabsampnum": "Laboratory Pedon ID",
    "upedonid": "User Pedon ID",
    "pedon_key": "Pedon Key (internal)",
    "site_key": "Site Key (internal)",
    "siteiid": "Site Interpretation ID",
}

# Taxon name columns
TAXON_COLUMNS = {
    "corr_name": "Correlated Taxon Name",
    "samp_name": "Sampled As Taxon Name",
    "taxon_name": "Taxon Name",
}

# Location columns
LOCATION_COLUMNS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "area_code": "Area Code",
    "area_name": "Area Name",
}


# ============================================================================
# Chunking Parameters
# ============================================================================

# Default chunking strategy
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_BY_COLUMN = "pedon_key"
DEFAULT_MAX_RETRIES = 3


def get_table_description(table_name: str) -> str:
    """Get human-readable description of an LDM table.

    Args:
        table_name: Name of the LDM table

    Returns:
        Description of the table
    """
    descriptions = {
        LAB_PHYSICAL_PROPERTIES_TABLE: "Physical properties (texture, density, porosity, etc.)",
        LAB_CHEMICAL_PROPERTIES_TABLE: "Chemical properties (pH, CEC, organic matter, nutrients, etc.)",
        LAB_CALCULATIONS_TABLE: "Calculated properties and default estimates",
        LAB_ROSETTA_KEY_TABLE: "ROSETTA model water retention parameters",
        LAB_MAJOR_TRACE_ELEMENTS_TABLE: "Major and trace elements, oxides",
        LAB_MINERALOGY_TABLE: "Mineralogy, glass count, optical properties",
        LAB_MIR_TABLE: "Mid-Infrared spectroscopy data",
        LAB_XRD_THERMAL_TABLE: "X-ray diffraction and thermal analysis",
        PEDON_TABLE: "Pedon site information and identifiers",
        SITE_TABLE: "Site descriptive information",
        LAB_LAYER_TABLE: "Laboratory layer (horizon) information",
    }
    return descriptions.get(table_name, "Unknown table")


def is_valid_table(table_name: str) -> bool:
    """Check if a table name is valid.

    Args:
        table_name: Name of the table to check

    Returns:
        True if table is valid, False otherwise
    """
    return table_name in ALL_TABLES


def is_valid_prep_code(prep_code: str) -> bool:
    """Check if a prep code is valid.

    Args:
        prep_code: Preparation code to check

    Returns:
        True if prep_code is valid, False otherwise
    """
    return prep_code in PREP_CODES


def is_valid_size_fraction(size_frac: str) -> bool:
    """Check if a size fraction is valid.

    Args:
        size_frac: Size fraction to check

    Returns:
        True if size_frac is valid, False otherwise
    """
    return size_frac in ANALYZED_SIZE_FRACTIONS


def is_valid_layer_type(layer_type: str) -> bool:
    """Check if a layer type is valid.

    Args:
        layer_type: Layer type to check

    Returns:
        True if layer_type is valid, False otherwise
    """
    if layer_type is None:
        return True
    return layer_type in LAYER_TYPES


def is_valid_area_type(area_type: str) -> bool:
    """Check if an area type is valid.

    Args:
        area_type: Area type to check

    Returns:
        True if area_type is valid, False otherwise
    """
    if area_type is None:
        return True
    return area_type in AREA_TYPES


def validate_tables(tables: List[str]) -> bool:
    """Validate a list of table names.

    Args:
        tables: List of table names to validate

    Returns:
        True if all tables are valid, False otherwise
    """
    return all(is_valid_table(t) for t in tables)


def all_valid_prep_codes() -> Set[str]:
    """Get all valid prep codes.

    Returns:
        Set of valid prep code strings
    """
    return set(PREP_CODES.keys())


def all_valid_size_fractions() -> Set[str]:
    """Get all valid analyzed size fractions.

    Returns:
        Set of valid size fraction strings
    """
    return set(ANALYZED_SIZE_FRACTIONS.keys())


def all_valid_layer_types() -> Set[str]:
    """Get all valid layer types.

    Returns:
        Set of valid layer type strings (includes None)
    """
    return set(LAYER_TYPES.keys()) | {None}


def all_valid_area_types() -> Set[str]:
    """Get all valid area types.

    Returns:
        Set of valid area type strings (includes None)
    """
    return set(AREA_TYPES.keys()) | {None}
