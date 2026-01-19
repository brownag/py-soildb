"""Unit tests for LDM exceptions."""

import pytest

from soildb.ldm import (
    LDMBackendError,
    LDMBackendSelectionError,
    LDMError,
    LDMParameterError,
    LDMQueryError,
    LDMResponseError,
    LDMSDAError,
    LDMSQLiteError,
    LDMTableError,
)
from soildb.exceptions import SoilDBError


class TestLDMExceptions:
    """Test LDM exception hierarchy and functionality."""

    def test_ldm_error_is_soildb_error(self):
        """Test that LDMError inherits from SoilDBError."""
        assert issubclass(LDMError, SoilDBError)

    def test_ldm_backend_error_inheritance(self):
        """Test LDMBackendError inheritance."""
        assert issubclass(LDMBackendError, LDMError)

    def test_ldm_sqlite_error_inheritance(self):
        """Test LDMSQLiteError inheritance."""
        assert issubclass(LDMSQLiteError, LDMBackendError)

    def test_ldm_sda_error_inheritance(self):
        """Test LDMSDAError inheritance."""
        assert issubclass(LDMSDAError, LDMBackendError)

    def test_ldm_backend_selection_error_inheritance(self):
        """Test LDMBackendSelectionError inheritance."""
        assert issubclass(LDMBackendSelectionError, LDMBackendError)

    def test_ldm_query_error_inheritance(self):
        """Test LDMQueryError inheritance."""
        assert issubclass(LDMQueryError, LDMError)

    def test_ldm_parameter_error_inheritance(self):
        """Test LDMParameterError inheritance."""
        assert issubclass(LDMParameterError, LDMQueryError)

    def test_ldm_table_error_inheritance(self):
        """Test LDMTableError inheritance."""
        assert issubclass(LDMTableError, LDMQueryError)

    def test_ldm_response_error_inheritance(self):
        """Test LDMResponseError inheritance."""
        assert issubclass(LDMResponseError, LDMError)

    def test_ldm_error_message(self):
        """Test LDMError stores message."""
        error = LDMError("Test message")
        assert error.message == "Test message"

    def test_ldm_error_with_details(self):
        """Test LDMError stores message and details."""
        error = LDMError("Test message", details="Test details")
        assert error.message == "Test message"
        assert error.details == "Test details"

    def test_parameter_error_str(self):
        """Test LDMParameterError string representation."""
        error = LDMParameterError("Invalid prep_code")
        error_str = str(error)
        assert "prep_code" in error_str

    def test_sqlite_error_str(self):
        """Test LDMSQLiteError string representation."""
        error = LDMSQLiteError("Database error", details="File not found")
        error_str = str(error)
        assert "SQLite" in error_str

    def test_sda_error_str(self):
        """Test LDMSDAError string representation."""
        error = LDMSDAError("Query failed", details="Timeout")
        error_str = str(error)
        assert "SDA" in error_str

    def test_exception_catching_hierarchy(self):
        """Test that exceptions can be caught at appropriate levels."""
        # Can catch LDMSQLiteError as LDMBackendError
        error = LDMSQLiteError("Test")
        assert isinstance(error, LDMBackendError)
        assert isinstance(error, LDMError)
        assert isinstance(error, SoilDBError)

        # Can catch LDMParameterError as LDMQueryError
        error = LDMParameterError("Test")
        assert isinstance(error, LDMQueryError)
        assert isinstance(error, LDMError)

    def test_all_exceptions_exported(self):
        """Test that all LDM exceptions are exported from module."""
        from soildb import ldm

        assert hasattr(ldm, "LDMError")
        assert hasattr(ldm, "LDMBackendError")
        assert hasattr(ldm, "LDMSQLiteError")
        assert hasattr(ldm, "LDMSDAError")
        assert hasattr(ldm, "LDMQueryError")
        assert hasattr(ldm, "LDMParameterError")
        assert hasattr(ldm, "LDMTableError")
