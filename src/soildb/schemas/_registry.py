"""
Schema registry with lazy-loading mechanism.

Loads schemas on-demand rather than all at startup.
Provides schema versioning and tracking.
"""

from typing import Callable, Dict, Optional

from ._base import TableSchema

# Lazy-loading registry
_SCHEMAS_CACHE: Dict[str, TableSchema] = {}
_SCHEMA_LOADERS: Dict[
    str, Callable[[], TableSchema]
] = {}  # Will be populated with lazy loaders


def _register_loader(table_name: str, loader_func: Callable[[], TableSchema]) -> None:
    """Register a lazy loader for a schema."""
    _SCHEMA_LOADERS[table_name] = loader_func


def _load_schema(table_name: str) -> Optional[TableSchema]:
    """Load a schema from its module (lazy-loaded on first access)."""
    if table_name in _SCHEMAS_CACHE:
        return _SCHEMAS_CACHE[table_name]

    if table_name not in _SCHEMA_LOADERS:
        return None

    # Load the schema
    schema = _SCHEMA_LOADERS[table_name]()
    _SCHEMAS_CACHE[table_name] = schema
    return schema


def get_schema(table_name: str) -> Optional[TableSchema]:
    """Get a schema by table name, loading it if necessary.

    Args:
        table_name: Name of the table/schema to retrieve

    Returns:
        TableSchema if found, None otherwise
    """
    return _load_schema(table_name)


def list_available_schemas() -> list:
    """List all available schema table names."""
    return sorted(_SCHEMA_LOADERS.keys())


def register_schemas() -> None:
    """Register all schema loaders. Call this once on module import."""
    from .chorizon import AGGREGATE_HORIZON_SCHEMA, CHORIZON_SCHEMA
    from .component import COMPONENT_SCHEMA, MAP_UNIT_COMPONENT_SCHEMA
    from .mapunit import MAPUNIT_SCHEMA, SOIL_MAP_UNIT_SCHEMA
    from .pedon import PEDON_HORIZON_SCHEMA, PEDON_SCHEMA
    from .property import HORIZON_PROPERTY_SCHEMA
    from .spatial import MUPOLYGON_SCHEMA

    # Register loaders
    _SCHEMA_LOADERS["mapunit"] = lambda: MAPUNIT_SCHEMA
    _SCHEMA_LOADERS["component"] = lambda: COMPONENT_SCHEMA
    _SCHEMA_LOADERS["chorizon"] = lambda: CHORIZON_SCHEMA
    _SCHEMA_LOADERS["pedon"] = lambda: PEDON_SCHEMA
    _SCHEMA_LOADERS["pedon_horizon"] = lambda: PEDON_HORIZON_SCHEMA
    _SCHEMA_LOADERS["horizon_property"] = lambda: HORIZON_PROPERTY_SCHEMA
    _SCHEMA_LOADERS["aggregate_horizon"] = lambda: AGGREGATE_HORIZON_SCHEMA
    _SCHEMA_LOADERS["map_unit_component"] = lambda: MAP_UNIT_COMPONENT_SCHEMA
    _SCHEMA_LOADERS["soil_map_unit"] = lambda: SOIL_MAP_UNIT_SCHEMA
    _SCHEMA_LOADERS["mupolygon"] = lambda: MUPOLYGON_SCHEMA


# Convenience access to commonly-used schemas
def get_mapunit_schema() -> TableSchema:
    """Get the mapunit schema."""
    return _load_schema("mapunit")  # type: ignore[return-value]


def get_component_schema() -> TableSchema:
    """Get the component schema."""
    return _load_schema("component")  # type: ignore[return-value]


def get_chorizon_schema() -> TableSchema:
    """Get the chorizon schema."""
    return _load_schema("chorizon")  # type: ignore[return-value]


def get_pedon_schema() -> TableSchema:
    """Get the pedon schema."""
    return _load_schema("pedon")  # type: ignore[return-value]


def get_pedon_horizon_schema() -> TableSchema:
    """Get the pedon_horizon schema."""
    return _load_schema("pedon_horizon")  # type: ignore[return-value]


# Initialize schemas on module import
register_schemas()
