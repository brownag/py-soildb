"""
Data Models for the soildb package.

These dataclasses provide structured, object-oriented representations for
common soil science data entities like Map Units, Components, and Horizons.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HorizonProperty:
    """Represents a soil property for a single horizon with low, rv, and high values.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard property fields.
    """

    property_name: str
    rv: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    unit: str = ""
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


@dataclass
class AggregateHorizon:
    """Represents an aggregate 'component horizon' with statistical summaries.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard horizon fields.
    """

    horizon_key: str  # chkey
    horizon_name: str
    top_depth: float
    bottom_depth: float
    properties: List[HorizonProperty] = field(default_factory=list)
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["properties"] = [p.to_dict() for p in self.properties]
        return d

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


@dataclass
class MapUnitComponent:
    """Represents a single component of a soil map unit.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard component fields.
    """

    component_key: str  # cokey
    component_name: str
    component_percentage: float
    is_major_component: bool
    taxonomic_class: Optional[str] = None
    drainage_class: Optional[str] = None
    local_phase: Optional[str] = None
    hydric_rating: Optional[str] = None
    component_kind: Optional[str] = None
    aggregate_horizons: List[AggregateHorizon] = field(default_factory=list)
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["aggregate_horizons"] = [h.to_dict() for h in self.aggregate_horizons]
        return d

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


@dataclass
class SoilMapUnit:
    """
    A complete, structured representation of a soil map unit, including its
    components and their aggregate horizons.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard map unit fields.
    """

    map_unit_key: str  # mukey
    map_unit_name: str
    map_unit_symbol: Optional[str] = None
    survey_area_symbol: Optional[str] = None
    survey_area_name: Optional[str] = None
    components: List[MapUnitComponent] = field(default_factory=list)
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire nested structure to a dictionary."""
        d = asdict(self)
        d["components"] = [c.to_dict() for c in self.components]
        return d

    def get_major_components(self) -> List[MapUnitComponent]:
        """Returns only the major components of the map unit."""
        return [c for c in self.components if c.is_major_component]

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


@dataclass
class PedonHorizon:
    """Represents a single horizon from a pedon with laboratory data.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard horizon fields.
    """

    pedon_key: str  # Foreign key to pedon
    layer_key: str  # Unique layer identifier
    horizon_name: str  # Horizon designation
    layer_sequence: Optional[int] = None  # Horizon sequence number
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    # Physical properties
    sand_total: Optional[float] = None
    silt_total: Optional[float] = None
    clay_total: Optional[float] = None
    texture_lab: Optional[str] = None
    # Chemical properties
    ph_h2o: Optional[float] = None
    organic_carbon: Optional[float] = None
    calcium_carbonate: Optional[float] = None
    # Physical properties
    bulk_density_third_bar: Optional[float] = None
    le_third_fifteen_lt2_mm: Optional[float] = None
    water_content_tenth_bar: Optional[float] = None
    water_content_third_bar: Optional[float] = None
    water_content_fifteen_bar: Optional[float] = None
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_extra_field(self, key: str) -> Any:
        """Get an extra field value by key."""
        return self.extra_fields.get(key)

    def has_extra_field(self, key: str) -> bool:
        """Check if an extra field exists."""
        return key in self.extra_fields

    def list_extra_fields(self) -> List[str]:
        """List all extra field keys."""
        return list(self.extra_fields.keys())


@dataclass
class PedonData:
    """
    A complete pedon with site information and laboratory-analyzed horizons.

    This dataclass includes an extra_fields dictionary to store arbitrary user-defined
    properties beyond the standard pedon fields.
    """

    pedon_key: str  # Primary key
    pedon_id: str  # User pedon ID
    series: Optional[str] = None  # Soil series name
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Classification
    soil_classification: Optional[str] = None  # Full soil classification
    # Horizons
    horizons: List[PedonHorizon] = field(default_factory=list)
    # --- ADDED ---
    # Dictionary for arbitrary user-defined properties.
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the pedon to a dictionary."""
        d = asdict(self)
        d["horizons"] = [h.to_dict() for h in self.horizons]
        return d

    def get_horizon_by_depth(self, depth: float) -> Optional[PedonHorizon]:
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
