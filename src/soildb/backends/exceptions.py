"""Backend-specific exceptions.

Provides exception hierarchy for all backend operations. Follows same
pattern as existing SDAError hierarchy in soildb.exceptions.
"""

from typing import Optional

from soildb.exceptions import SoilDBError


class BackendError(SoilDBError):
    """Base exception for all backend operations.

    All backend-specific exceptions inherit from this, allowing code to
    catch any backend error with `except BackendError`.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)


class BackendConnectionError(BackendError):
    """Raised when backend connection fails.

    Includes network errors, authentication failures, missing database files,
    and other connection-related issues.
    """

    def __str__(self) -> str:
        """Return detailed connection error message."""
        base_msg = "Failed to connect to data backend"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg


class BackendQueryError(BackendError):
    """Raised when query execution fails.

    Includes SQL errors, invalid columns/tables, timeouts, and other
    query execution failures.
    """

    def __str__(self) -> str:
        """Return detailed query error message."""
        base_msg = "Query execution failed"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg


class BackendSchemaError(BackendError):
    """Raised when schema introspection fails.

    Includes errors getting table/column information, invalid table names,
    and permission issues.
    """

    def __str__(self) -> str:
        """Return detailed schema error message."""
        base_msg = "Failed to retrieve schema information"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg
