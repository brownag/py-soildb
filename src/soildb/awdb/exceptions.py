"""
Exceptions for AWDB operations.
"""


class AWDBError(Exception):
    """Base exception for AWDB-related errors."""

    pass


class AWDBConnectionError(AWDBError):
    """Error connecting to AWDB API."""

    pass


class AWDBQueryError(AWDBError):
    """Error in AWDB query or response parsing."""

    pass
