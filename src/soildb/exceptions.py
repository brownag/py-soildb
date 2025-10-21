"""
Exception classes for soildb.
"""

from typing import Optional


class SoilDBError(Exception):
    """Base exception for all soildb errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)


class SDAConnectionError(SoilDBError):
    """Raised when there are connection issues with the SDA service."""

    def __str__(self) -> str:
        """Return detailed error message including SDA response details."""
        base_msg = "Failed to connect to USDA Soil Data Access service."
        if self.details:
            return f"{base_msg} SDA Response: {self.details}. Check your internet connection and try again."
        return f"{base_msg} Check your internet connection and try again."


class SDAQueryError(SoilDBError):
    """Raised when a query fails or returns invalid results."""

    def __init__(
        self, message: str, query: Optional[str] = None, details: Optional[str] = None
    ):
        self.query = query
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return detailed error message including query and SDA details."""
        parts = [self.message]
        if self.query:
            parts.append(f"Query: {self.query}")
        if self.details:
            parts.append(f"SDA Response: {self.details}")
        return "\n".join(parts)


class SDAMaintenanceError(SoilDBError):
    """Raised when the SDA service is under daily (or other) maintenance."""

    def __str__(self) -> str:
        """Return helpful maintenance message."""
        return "USDA Soil Data Access service is currently under maintenance. This typically occurs during off-hours (each day from 12:45 AM to 1 AM Central). Please try again in a few minutes."


class SDATimeoutError(SDAConnectionError):
    """Raised when a request to SDA times out."""

    def __str__(self) -> str:
        """Return helpful timeout message."""
        return "Request to USDA Soil Data Access service timed out. This may be due to network issues, high server load, or complex queries. Try increasing the timeout or simplifying your query."


class SDAResponseError(SDAQueryError):
    """Raised when SDA returns an invalid or unexpected response format."""

    def __str__(self) -> str:
        """Return helpful response error message."""
        return f"Received invalid response from USDA Soil Data Access service: {self.message}. This may indicate a service issue or malformed query. Check your query syntax and try again."


class SyncUsageError(SoilDBError):
    """Raised when .sync methods are used incorrectly."""

    def __str__(self) -> str:
        """Return helpful sync usage error message."""
        return f"{self.message}"
