"""
Exceptions for AWDB operations.

These exceptions mirror the structure of SDA exceptions but for AWDB service operations.
Both inherit from SoilDBError, allowing unified exception handling across services.
"""

from soildb.exceptions import AWDBError, AWDBConnectionError, AWDBQueryError

# Re-export for backward compatibility and convenience
__all__ = ["AWDBError", "AWDBConnectionError", "AWDBQueryError"]
