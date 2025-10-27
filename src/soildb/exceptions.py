"""
Exception classes for soildb with restructured hierarchy.

Exception Hierarchy:
====================

SoilDBError (base for all soildb exceptions)
├── SDANetworkError (network/connection related)
│   ├── SDAConnectionError (connection failures)
│   ├── SDATimeoutError (request timeouts)
│   └── SDAMaintenanceError (service maintenance)
├── SDAQueryError (query execution failures)
│   └── SDAResponseError (invalid response format)
└── AWDBError (AWDB service errors)
    ├── AWDBConnectionError (connection failures)
    └── AWDBQueryError (query failures)

This hierarchy ensures:
- Connection/timeout errors inherit from SDANetworkError
- Query/response errors inherit from SDAQueryError
- Maintenance errors treated as network issues (temporary unavailability)
- Services can be caught at appropriate levels: catch SDANetworkError for all network issues
"""

from typing import Optional


class SoilDBError(Exception):
    """Base exception for all soildb errors.
    
    All soildb-specific exceptions inherit from this class, allowing
    code to catch any soildb error with `except SoilDBError`.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)


# ============================================================================
# SDA Service Exception Hierarchy
# ============================================================================


class SDANetworkError(SoilDBError):
    """Base for network-related SDA errors (connections, timeouts, maintenance).
    
    Catch this exception to handle all network-related issues with the SDA service.
    Includes temporary unavailability (maintenance) and connection problems.
    """

    pass


class SDAConnectionError(SDANetworkError):
    """Raised when there are connection issues with the SDA service.
    
    This includes HTTP errors, DNS failures, and other network connectivity problems.
    This does NOT include timeouts (see SDATimeoutError) or maintenance windows (see SDAMaintenanceError).
    """

    def __str__(self) -> str:
        """Return detailed error message including SDA response details."""
        base_msg = "Failed to connect to USDA Soil Data Access service."
        if self.details:
            return f"{base_msg} SDA Response: {self.details}. Check your internet connection and try again."
        return f"{base_msg} Check your internet connection and try again."


class SDATimeoutError(SDANetworkError):
    """Raised when a request to SDA times out.
    
    Timeouts can result from network latency, high server load, or queries
    that exceed the timeout threshold. This is semantically distinct from
    general connection errors and inherits from SDANetworkError.
    """

    def __str__(self) -> str:
        """Return helpful timeout message."""
        return "Request to USDA Soil Data Access service timed out. This may be due to network issues, high server load, or complex queries. Try increasing the timeout or simplifying your query."


class SDAMaintenanceError(SDANetworkError):
    """Raised when the SDA service is under maintenance.
    
    SDA undergoes daily maintenance typically from 12:45 AM to 1 AM Central Time.
    This exception indicates temporary unavailability rather than a permanent error.
    It inherits from SDANetworkError as the service is temporarily unreachable.
    """

    def __str__(self) -> str:
        """Return helpful maintenance message."""
        return "USDA Soil Data Access service is currently under maintenance. This typically occurs during off-hours (each day from 12:45 AM to 1 AM Central). Please try again in a few minutes."


class SDAQueryError(SoilDBError):
    """Raised when a query fails or returns invalid results.
    
    This includes SQL syntax errors, invalid table/column names, and other
    query-related failures. This does NOT include network errors (see SDANetworkError)
    or response parsing errors (see SDAResponseError).
    """

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


class SDAResponseError(SDAQueryError):
    """Raised when SDA returns an invalid or unexpected response format.
    
    This occurs when the SDA service returns a response that cannot be parsed
    or validated. Unlike SDAQueryError, this indicates the query was accepted
    but the response was malformed.
    """

    def __str__(self) -> str:
        """Return helpful response error message."""
        return f"Received invalid response from USDA Soil Data Access service: {self.message}. This may indicate a service issue or malformed query. Check your query syntax and try again."


# ============================================================================
# AWDB Service Exception Hierarchy
# ============================================================================


class AWDBError(SoilDBError):
    """Base exception for AWDB-related errors.
    
    All AWDB-specific exceptions inherit from this class, allowing code to
    catch any AWDB error with `except AWDBError`.
    """

    pass


class AWDBConnectionError(AWDBError):
    """Raised when there are connection issues with the AWDB service.
    
    This includes timeouts, network errors, rate limiting, and service unavailability.
    Mirror of SDANetworkError but for AWDB service.
    """

    pass


class AWDBQueryError(AWDBError):
    """Raised when an AWDB query fails or returns invalid results.
    
    This includes invalid parameters, missing data, and response parsing errors.
    Mirror of SDAQueryError but for AWDB service.
    """

    pass


