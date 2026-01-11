# Changelog

## [0.6.0] - 2026-01-09

Release adding Web Soil Survey download capabilities for bulk SSURGO and STATSGO data acquisition.

### Added

- **Web Soil Survey Downloads**: Complete bulk download functionality for soil survey datasets
  - `download_wss()` function for downloading SSURGO and STATSGO data as ZIP files
  - Support for concurrent downloads with configurable concurrency limits
  - Automatic ZIP extraction and file organization into tabular/spatial directories
  - WSSClient class for low-level download operations
  - WSSDownloadError exception for download-specific error handling

## [0.5.0] - 2026-01-02

Release focusing on AWDB API improvements, HENRY database integration, and function renaming for semantic clarity.

### Breaking Changes

- **AWDB API Function Renaming**: Introduced semantically clearer function names
  - `find_stations_by_criteria` → `discover_stations`
  - `get_monitoring_station_data` → `get_property_data_near`
  - Deprecated older function names (backward compatibility maintained)
- **AWDB Parameter Renaming**: `begin_publication_date` → `start_publication_date` in AWDB client

### Added

- **HENRY Database Integration**: New functions for working with Henry Mount Soil Temperature and Water Database
- **Timezone Handling**: `_apply_station_timezone` helper function for correct timezone interpretation
  - Applies station metadata timezone offsets to naive datetime values
  - Aligns with ISO 8601 timestamp standards
- **Enhanced TimeSeriesDataPoint Model**: Added metadata fields
  - `element_code`: Element identifier
  - `variable_name`: Variable name
  - `station_timezone_offset`: Timezone offset information

### Changed

- **AWDB Client Enhancements**: `get_station_data` method now automatically fetches station metadata for timezone offset application
- **Hourly Data Support**: Improved handling of hourly data with proper timezone assignment

## [0.4.0] - 2025-12-10

Major release with API consolidation, improved developer experience, and production readiness.

**Breaking Changes**: See "Changed" section for migration details. Old API functions have straightforward replacements.

### Added

- **Unified spatial query API**: Single `spatial_query()` function for all geometry types
- **Unified bulk fetch API**: Single `fetch_by_keys()` function replacing specialized functions
- **Synchronous API wrapper**: `.sync()` decorator for all async functions for easier interactive use
- **SoilProfileCollection integration**: Export soil horizon data to soilprofilecollection objects
- **Soil water availability workflows**: Integrated AWDB + SDA for comprehensive soil-water analysis
- **Modular schema system**: Improved schema architecture for extensibility
- **Consolidated type mapping**: Unified approach to type conversion across all data types
- **Enhanced exception hierarchy**: Structured error types (SDANetworkError, SDAQueryError, AWDBError)
- **ResponseValidator**: Improved response parsing and validation
- **Error handling guide**: Comprehensive documentation on handling network and query errors
- **Troubleshooting guide**: Solutions for common issues and performance optimization

### Changed

- **Breaking**: Removed legacy `SpatialQuery` and `QueryBuilder` classes (use `Query()` instead)
- **Breaking**: Removed specialized fetch functions (use `fetch_by_keys()` instead)
  - `fetch_mapunit_polygon()` → `fetch_by_keys(..., "mupolygon")`
  - `fetch_component_by_mukey()` → `fetch_by_keys(..., "component", "mukey")`
  - `fetch_chorizon_by_cokey()` → `fetch_by_keys(..., "chorizon", "cokey")`
  - `fetch_survey_area_polygon()` → `fetch_by_keys(..., "sapolygon")`
- **Breaking**: Removed spatial query functions (use `spatial_query()` instead)
  - `query_mupolygon()`, `query_sapolygon()`, `query_featpoint()`, `query_featline()`
- **Breaking**: Removed `soildb.models` module (use `soildb.schema_system` instead)
- Sync API is now the primary documented interface (async available for advanced users)
- Improved fetch hierarchy with better pagination strategies
- Enhanced metadata parsing from fgdcmetadata column

### Fixed

- SDA empty response handling
- SDAResponse data_quality_score calculation
- Custom column name generalization in high-level schemas
- Type annotations and mypy compliance
- Response validation for edge cases

### Dependencies

- Added: hatchling (build system)
- Added: editables (editable installs)
- Added: build, quartodoc, quarto (documentation tools)
- Improved: pandas, polars, soilprofilecollection support

## [0.3.0] - 2025-11-15

Release focusing on sync API, AWDB integration, and response validation.

### Added

- **AWDB Integration**: Full support for NRCS Air and Water Database
  - `soildb.awdb` module for station queries and data retrieval
  - `find_stations_by_criteria()` for location-based station discovery
  - `get_monitoring_station_data()` for historical data access
  - Integration with soil data for water availability analysis
- **Synchronous wrappers**: `.sync()` versions of all async functions
  - Automatic SDAClient lifecycle management
  - No async/await syntax required
  - Easier for interactive and script use
- **ResponseValidator**: Improved response parsing
  - Better empty response handling
  - Type validation and coercion
  - Column name normalization
- **Automatic client management**: Client creation and cleanup handled automatically
- **Type processor documentation**: Comprehensive guide to type conversion system

### Changed

- Improved error handling with better validation
- Enhanced response parsing for edge cases
- Better integration between modules

### Fixed

- Empty soilprofilecollection conversion
- Test markers and integration tests
- Unicode and debug statement cleanup

## [0.2.0] - 2025-10-19

Major release with schema-driven architecture.

### Added

- Schema system for data processing and type validation
- Dynamic model generation from schemas at runtime
- Extra fields support for custom columns
- Optional pandas/polars dependencies
- Enhanced type safety and validation

### Changed

- High-level functions now use schema-based processing
- Fetch layer uses centralized schema column definitions
- Improved error handling and exceptions

### Fixed

- Organic carbon calculations with CaCO3 correction
- Custom column handling in `extra_fields`
- Pandas dtype comparison logic

## [0.1.0] - 2025-09-28

Initial release of soildb Python package.

### Added

- SDA client for querying USDA soil data
- Query builder for custom SQL queries
- DataFrame export (pandas and polars)
- Spatial queries with bounding boxes and points
- Bulk data fetching with pagination
- Example scripts for soil analysis

