"""
Utility functions for Henry Mount Soil Climate Database operations.

These utilities handle:
- Depth conversion between centimeters and inches
- Element code construction in AWDB format
- Henry-specific data transformations
"""

from typing import Optional


def cm_to_inches(depth_cm: float) -> float:
    """
    Convert depth from centimeters to inches.

    For below-ground measurements, depths are negative. For above-ground
    measurements, depths are positive.

    Args:
        depth_cm: Depth in centimeters. Negative values indicate below-ground.

    Returns:
        Depth in inches (preserves sign).

    Examples:
        >>> cm_to_inches(5.08)
        2.0
        >>> cm_to_inches(-10.16)
        -4.0
    """
    return depth_cm / 2.54


def inches_to_cm(depth_inches: float) -> float:
    """
    Convert depth from inches to centimeters.

    Args:
        depth_inches: Depth in inches.

    Returns:
        Depth in centimeters.

    Examples:
        >>> inches_to_cm(2.0)
        5.08
        >>> inches_to_cm(-4.0)
        -10.16
    """
    return depth_inches * 2.54


def construct_element_code(
    base_code: str,
    depth_cm: Optional[float] = None,
    ordinal: int = 1,
) -> str:
    """
    Construct an AWDB-style element code.

    Element codes follow the format: BASE_CODE:DEPTH_INCHES:ORDINAL
    where DEPTH_INCHES is negative for below-ground sensors.

    Henry provides depths as positive centimeters for below-ground sensors,
    which are converted to negative inches for AWDB compatibility.

    Args:
        base_code: Base element code (e.g., 'SMS' for soil moisture, 'STO' for soil temperature)
        depth_cm: Sensor depth in centimeters. Positive values represent below-ground depths.
        ordinal: Ordinal number for multiple sensors of same type (default: 1)

    Returns:
        Formatted element code string (e.g., 'SMS:-2:1')

    Examples:
        >>> construct_element_code('SMS', depth_cm=5.08, ordinal=1)
        'SMS:-2:1'
        >>> construct_element_code('STO', depth_cm=10.16, ordinal=1)
        'STO:-4:1'
        >>> construct_element_code('PREC', ordinal=1)
        'PREC::1'
    """
    if depth_cm is None:
        return f"{base_code}::{ordinal}"

    # Convert cm to inches and negate for below-ground sensors (AWDB convention)
    depth_inches = -int(round(abs(cm_to_inches(depth_cm))))
    return f"{base_code}:{depth_inches}:{ordinal}"


def parse_element_code(element_code: str) -> dict:
    """
    Parse an AWDB-style element code into its components.

    Args:
        element_code: Element code string (e.g., 'SMS:-2:1')

    Returns:
        Dictionary with keys:
        - 'base_code': Base code (str)
        - 'depth_inches': Depth in inches (int or None)
        - 'ordinal': Ordinal number (int)

    Examples:
        >>> parse_element_code('SMS:-2:1')
        {'base_code': 'SMS', 'depth_inches': -2, 'ordinal': 1}
        >>> parse_element_code('PREC::1')
        {'base_code': 'PREC', 'depth_inches': None, 'ordinal': 1}
    """
    parts = element_code.split(":")
    if len(parts) != 3:
        # Fallback for malformed codes
        return {
            "base_code": element_code,
            "depth_inches": None,
            "ordinal": 1,
        }

    base_code = parts[0]
    depth_inches = int(parts[1]) if parts[1] and parts[1] != "" else None
    ordinal = int(parts[2]) if parts[2] and parts[2] != "" else 1

    return {
        "base_code": base_code,
        "depth_inches": depth_inches,
        "ordinal": ordinal,
    }


def henry_variable_to_base_code(variable_name: str) -> str:
    """
    Map Henry variable names to AWDB base element codes.

    Args:
        variable_name: Henry variable name (e.g., 'soiltemp', 'soilVWC', 'airtemp')

    Returns:
        Base element code (e.g., 'STO', 'SMS', 'TOBS')

    Mapping:
        - soiltemp → STO (Soil Temperature)
        - soilVWC → SMS (Soil Moisture, volumetric water content)
        - airtemp → TOBS (Air Temperature Observed)
        - waterlevel → varies (stream stage or water table depth)
    """
    variable_map = {
        "soiltemp": "STO",  # Soil Temperature
        "soilvwc": "SMS",  # Soil Moisture (Volumetric Water Content)
        "soilvmc": "SMS",  # Soil Moisture (Volumetric Water Content)
        "airtemp": "TOBS",  # Air Temperature Observed
        "waterlevel": "WL",  # Water Level
    }

    # Case-insensitive lookup
    normalized = variable_name.lower()
    return variable_map.get(normalized, "UNKNOWN")


def parse_henry_timestamp(timestamp_str: str) -> str:
    """
    Normalize Henry timestamps to ISO8601 format.

    Henry API returns timestamps as space-separated datetime strings:
    'YYYY-MM-DD HH:MM:SS' (space, not T)

    This function preserves the datetime but ensures consistent formatting.

    Args:
        timestamp_str: Timestamp string from Henry API

    Returns:
        ISO8601 formatted timestamp (preserving original datetime, just normalized)

    Examples:
        >>> parse_henry_timestamp('2024-01-15 14:30:00')
        '2024-01-15T14:30:00'
    """
    # Henry format: YYYY-MM-DD HH:MM:SS (space-separated)
    # ISO format: YYYY-MM-DDTHH:MM:SS (T-separated)
    if isinstance(timestamp_str, str) and " " in timestamp_str:
        # Replace space with T for ISO8601
        return timestamp_str.replace(" ", "T", 1)
    return timestamp_str


__all__ = [
    "cm_to_inches",
    "inches_to_cm",
    "construct_element_code",
    "parse_element_code",
    "henry_variable_to_base_code",
    "parse_henry_timestamp",
]
