"""
Exceptions for Henry Mount Soil Climate Database operations.

These exceptions inherit from soildb base exceptions, allowing unified
exception handling across different data sources (AWDB, SDA, Henry, etc.).
"""

from soildb.exceptions import SoilDBError


class HenryError(SoilDBError):
    """Base exception for Henry-related errors."""

    pass


class HenryAPIError(HenryError):
    """Raised when Henry API returns an error response or unexpected format."""

    pass


class HenryNetworkError(HenryError):
    """Raised when network communication with Henry API fails."""

    pass


class HenryDataError(HenryError):
    """Raised when Henry data is malformed, invalid, or cannot be parsed."""

    pass


__all__ = [
    "HenryError",
    "HenryAPIError",
    "HenryNetworkError",
    "HenryDataError",
]
