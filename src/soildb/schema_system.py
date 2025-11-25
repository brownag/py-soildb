"""
Schema-driven column mapping system for flexible data structures.

This module provides a completely automatic system for mapping database columns
to dataclass fields with minimal hardcoded logic.

REFACTORED ARCHITECTURE:
The schema definitions have been refactored into modular files in the schemas/
subdirectory to reduce monolithic definitions from 900+ lines to ~200 lines:

- schemas/_base.py: ColumnSchema, TableSchema classes
- schemas/_registry.py: Lazy-loading registry
- schemas/mapunit.py: Mapunit and soil_map_unit schemas
- schemas/component.py: Component and map_unit_component schemas
- schemas/chorizon.py: Chorizon and aggregate_horizon schemas
- schemas/pedon.py: Pedon and pedon_horizon schemas
- schemas/property.py: Horizon property schema
- schemas/spatial.py: Mupolygon and spatial schemas

BACKWARD COMPATIBILITY:
- SCHEMAS dictionary still works via lazy-loading (no change needed in code using it)
- get_schema() function works as before
- All dataclass creation functions unchanged

USAGE:
    from soildb.schema_system import get_schema, SCHEMAS, create_dynamic_dataclass
    schema = get_schema("mapunit")
    schema_dict = SCHEMAS["component"]
    
    from soildb.schemas import list_available_schemas
    available = list_available_schemas()

For detailed schema design documentation, see: docs/SCHEMA_SYSTEM.md
"""

from dataclasses import asdict, dataclass, field, make_dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore

# Import base classes and registry from modular schema system
from .schemas import ColumnSchema, TableSchema, get_schema as schemas_get_schema
from .schemas import list_available_schemas
from .schemas._registry import _load_schema, _SCHEMA_LOADERS


class _LazySchemaDict(dict):
    """Dictionary wrapper that lazy-loads schemas on access.
    
    Provides backward compatibility with code that uses SCHEMAS["table_name"].
    Schemas are loaded on-demand rather than all at startup, improving performance.
    """

    def __getitem__(self, key: str) -> TableSchema:
        """Get schema by table name, loading if necessary."""
        schema = _load_schema(key)
        if schema is None:
            raise KeyError(f"Schema not found for table: {key}")
        return schema

    def __contains__(self, key: Any) -> bool:
        """Check if schema exists."""
        return key in _SCHEMA_LOADERS

    def keys(self) -> list:
        """Get all available schema table names."""
        return sorted(_SCHEMA_LOADERS.keys())

    def get(self, key: str, default: Any = None) -> Optional[TableSchema]:
        """Get schema or return default."""
        return _load_schema(key) or default

    def items(self) -> list:
        """Get all schema items (loads all schemas)."""
        return [(k, _load_schema(k)) for k in sorted(_SCHEMA_LOADERS.keys())]

    def values(self) -> list:
        """Get all schema values (loads all schemas)."""
        return [_load_schema(k) for k in sorted(_SCHEMA_LOADERS.keys())]

    def __iter__(self):
        """Iterate over schema table names."""
        return iter(sorted(_SCHEMA_LOADERS.keys()))

    def __len__(self) -> int:
        """Get count of available schemas."""
        return len(_SCHEMA_LOADERS)


# Maintain backward compatibility - SCHEMAS dictionary now uses lazy-loading
SCHEMAS = _LazySchemaDict()


def get_schema(table_name: str) -> Optional[TableSchema]:
    """Get schema for a table.
    
    This function provides backward compatibility with existing code.
    Schemas are lazy-loaded on first access.
    
    Args:
        table_name: Name of the table schema to retrieve
        
    Returns:
        TableSchema if found, None otherwise
        
    Example:
        schema = get_schema("mapunit")
        if schema:
            print(f"Default columns: {schema.get_default_columns()}")
    """
    return _load_schema(table_name)


def add_column_to_schema(table_name: str, column_schema: ColumnSchema) -> None:
    """Add a new column to an existing schema.
    
    This allows dynamic schema extension at runtime.
    
    Args:
        table_name: Name of the table
        column_schema: ColumnSchema object to add
        
    Example:
        from soildb.schema_system import add_column_to_schema, ColumnSchema
        from soildb.type_processors import to_optional_str
        
        new_col = ColumnSchema(
            "custom_field", str, to_optional_str,
            default=True, field_name="custom_field"
        )
        add_column_to_schema("mapunit", new_col)
    """
    schema = _load_schema(table_name)
    if schema:
        schema.columns[column_schema.name] = column_schema


def create_dynamic_dataclass(
    schema: TableSchema, name: str, base_class: Optional[Type] = None
) -> Type[Any]:
    """Create a dataclass dynamically from a schema.

    Args:
        schema: The table schema to create the dataclass from
        name: Name for the new dataclass
        base_class: Optional base class to inherit from (for complex models)
        
    Returns:
        A new dataclass type with methods for accessing extra_fields

    Example:
        from soildb.schema_system import get_schema, create_dynamic_dataclass
        
        schema = get_schema("mapunit")
        MapUnit = create_dynamic_dataclass(schema, "MapUnit")
        
        # Create instances from row data
        data = {"mukey": "123456", "muname": "Miami", "extra_fields": {}}
        mu = MapUnit(**data)
    """
    from dataclasses import Field

    fields: List[tuple[str, Any, Any]] = []

    # Add base fields
    for fname, default_value in schema.base_fields.items():
        if fname == "extra_fields":
            fields.append((fname, Dict[str, Any], field(default_factory=dict)))
        elif isinstance(default_value, list):
            fields.append((fname, List[Any], field(default_factory=list)))
        else:
            fields.append((fname, type(default_value), default_value))

    # Add schema-defined fields with proper defaults for Optional types
    for col_schema in schema.columns.values():
        if col_schema.field_name and col_schema.field_name not in [
            f[0] for f in fields
        ]:
            # For Optional types or fields, use None as default if not explicitly set
            default_val = None
            # Check if type hint is Optional by looking at string representation
            type_str = str(col_schema.type_hint)
            if "Optional" in type_str or "Union" in type_str or "None" in type_str:
                default_val = None
            elif col_schema.field_name == "unit":
                # Special case: unit field defaults to ""
                default_val = ""
            else:
                default_val = None

            fields.append((col_schema.field_name, col_schema.type_hint, default_val))

    # Define utility methods
    def get_extra_field(self: Any, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self: Any, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self: Any) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())

    def to_dict(self: Any) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    if base_class:
        # Create new methods dict
        base_dict = {}
        base_dict["get_extra_field"] = get_extra_field  # type: ignore
        base_dict["has_extra_field"] = has_extra_field  # type: ignore
        base_dict["list_extra_fields"] = list_extra_fields  # type: ignore
        base_dict["to_dict"] = to_dict  # type: ignore

        # Create the class
        DynamicClass = type(name, (base_class,), base_dict)
        return DynamicClass
    else:
        # Use make_dataclass for the basic structure
        dataclass_fields = []
        for fname, ftype, default_val in fields:
            if isinstance(default_val, Field):
                dataclass_fields.append((fname, ftype, default_val))
            else:
                dataclass_fields.append((fname, ftype, default_val))

        DynamicClass = make_dataclass(name, dataclass_fields)

        # Add methods to the class
        DynamicClass.get_extra_field = get_extra_field  # type: ignore
        DynamicClass.has_extra_field = has_extra_field  # type: ignore
        DynamicClass.list_extra_fields = list_extra_fields  # type: ignore
        DynamicClass.to_dict = to_dict  # type: ignore

        return DynamicClass


@dataclass
class PedonData:
    """
    A complete pedon with site information and laboratory-analyzed horizons.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard pedon fields.
    """

    pedon_key: str  # Primary key
    pedon_id: str  # User pedon ID
    taxonname: Optional[str] = None  # Soil taxonomic name (series or higher level)
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Classification
    taxclname: Optional[str] = None  # Full taxonomic class name
    # Horizons
    horizons: List[Any] = field(default_factory=list)
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the pedon to a dictionary."""
        d = asdict(self)
        d["horizons"] = [h.to_dict() for h in self.horizons]
        return d

    def get_horizon_by_depth(self, depth: float) -> Optional[Any]:
        """Get the horizon that contains the specified depth."""
        for horizon in self.horizons:
            if (
                horizon.top_depth is not None
                and horizon.bottom_depth is not None
                and horizon.top_depth <= depth < horizon.bottom_depth
            ):
                return horizon
        return None

    def get_profile_depth(self) -> float:
        """Get the total depth of the pedon profile."""
        if not self.horizons:
            return 0.0
        valid_depths = [
            h.bottom_depth for h in self.horizons if h.bottom_depth is not None
        ]
        return max(valid_depths) if valid_depths else 0.0

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


# Create dynamic dataclasses from schemas
PedonHorizon = create_dynamic_dataclass(get_schema("pedon_horizon"), "PedonHorizon")
HorizonProperty = create_dynamic_dataclass(
    get_schema("horizon_property"), "HorizonProperty"
)
AggregateHorizon = create_dynamic_dataclass(
    get_schema("aggregate_horizon"), "AggregateHorizon"
)
MapUnitComponent = create_dynamic_dataclass(
    get_schema("map_unit_component"), "MapUnitComponent"
)
SoilMapUnit = create_dynamic_dataclass(get_schema("soil_map_unit"), "SoilMapUnit")


# Export all dynamically created models
__all__ = [
    "ColumnSchema",
    "TableSchema",
    "PedonData",
    "PedonHorizon",
    "HorizonProperty",
    "AggregateHorizon",
    "MapUnitComponent",
    "SoilMapUnit",
    "SCHEMAS",
    "get_schema",
    "add_column_to_schema",
    "create_dynamic_dataclass",
    "list_available_schemas",
]
