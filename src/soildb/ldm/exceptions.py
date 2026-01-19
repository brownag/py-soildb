"""
Exception classes for LDM (Lab Data Mart) operations.

Exception Hierarchy:
====================

SoilDBError (inherited from main exceptions)
└── LDMError (base for all LDM errors)
    ├── LDMBackendError (backend-related errors)
    │   ├── LDMSQLiteError (SQLite database errors)
    │   ├── LDMSDAError (SDA service errors)
    │   └── LDMBackendSelectionError (invalid backend selection)
    ├── LDMQueryError (query construction/execution errors)
    │   ├── LDMParameterError (invalid parameters)
    │   └── LDMTableError (invalid table names)
    └── LDMResponseError (response parsing/conversion errors)
"""

from soildb.exceptions import SoilDBError


class LDMError(SoilDBError):
    """Base exception for all LDM-related errors.

    All LDM-specific exceptions inherit from this class, allowing code to catch
    any LDM error with `except LDMError`.
    """

    pass


# ============================================================================
# Backend Errors
# ============================================================================


class LDMBackendError(LDMError):
    """Base exception for LDM backend-related errors.

    Catch this exception to handle all issues related to backend selection and operations.
    """

    pass


class LDMSQLiteError(LDMBackendError):
    """Raised when there are issues with SQLite backend operations.

    This includes file not found, invalid database format, connection errors,
    and query execution failures on local SQLite databases.
    """

    def __str__(self) -> str:
        """Return detailed SQLite error message."""
        base_msg = "Error with SQLite LDM database"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg


class LDMSDAError(LDMBackendError):
    """Raised when there are issues with SDA backend operations.

    This includes query failures, timeout errors, and other SDA service issues
    when using the web service backend.
    """

    def __str__(self) -> str:
        """Return detailed SDA error message."""
        base_msg = "Error with SDA backend for LDM queries"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg


class LDMBackendSelectionError(LDMBackendError):
    """Raised when backend cannot be selected or initialized.

    This occurs when both dsn parameter and SDA are unavailable, or when
    there are conflicts in backend configuration.
    """

    pass


# ============================================================================
# Query Errors
# ============================================================================


class LDMQueryError(LDMError):
    """Raised when query construction or execution fails.

    This is the base class for query-related errors. Catch this to handle
    all query construction and execution issues.
    """

    pass


class LDMParameterError(LDMQueryError):
    """Raised when invalid parameters are provided to LDM functions.

    This includes invalid values for prep_code, analyzed_size_frac, layer_type,
    area_type, or conflicting parameters like both 'x' and 'WHERE'.
    """

    def __str__(self) -> str:
        """Return helpful parameter error message."""
        if "prep_code" in self.message:
            return f"{self.message}. Valid prep_codes are: S (sieved), D (dispersed), C (crushed), or empty string."
        elif "analyzed_size_frac" in self.message:
            return f"{self.message}. Valid size fractions are: '<2 mm', '>2 mm', '2-5 mm', or empty string."
        elif "layer_type" in self.message:
            return f"{self.message}. Valid layer_types are: 'horizon', 'layer', 'reporting layer', or None."
        elif "area_type" in self.message:
            return f"{self.message}. Valid area_types are: 'ssa', 'state', 'county', 'mlra', 'nforest', 'npark', or None."
        return self.message


class LDMTableError(LDMQueryError):
    """Raised when invalid LDM table names are specified.

    This occurs when a table name is not recognized as a valid LDM table.
    """

    def __str__(self) -> str:
        """Return helpful table error message."""
        return f"{self.message}. Use ldm.tables module to see valid table names."


# ============================================================================
# Response Errors
# ============================================================================


class LDMResponseError(LDMError):
    """Raised when response parsing or conversion fails.

    This includes issues converting SQLite results to SDAResponse format,
    type conversion failures, or unexpected response structures.
    """

    def __str__(self) -> str:
        """Return detailed response error message."""
        base_msg = "Error parsing or converting LDM response"
        if self.details:
            return f"{base_msg}: {self.details}"
        return base_msg
