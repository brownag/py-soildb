# Changelog

## [0.2.0] - 2025-10-17

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