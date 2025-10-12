"""
Data Models for the soildb package.

These dataclasses provide structured, object-oriented representations for
common soil science data entities like Map Units, Components, and Horizons.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class HorizonProperty:
    """Represents a soil property for a single horizon with low, rv, and high values."""
    property_name: str
    rv: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    unit: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateHorizon:
    """Represents an aggregate 'component horizon' with statistical summaries."""
    horizon_key: str  # chkey
    horizon_name: str
    top_depth: float
    bottom_depth: float
    properties: List[HorizonProperty] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['properties'] = [p.to_dict() for p in self.properties]
        return d


@dataclass
class MapUnitComponent:
    """Represents a single component of a soil map unit."""
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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['aggregate_horizons'] = [h.to_dict() for h in self.aggregate_horizons]
        return d


@dataclass
class SoilMapUnit:
    """
    A complete, structured representation of a soil map unit, including its
    components and their aggregate horizons.
    """
    map_unit_key: str  # mukey
    map_unit_name: str
    map_unit_symbol: Optional[str] = None
    survey_area_symbol: Optional[str] = None
    survey_area_name: Optional[str] = None
    components: List[MapUnitComponent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire nested structure to a dictionary."""
        d = asdict(self)
        d['components'] = [c.to_dict() for c in self.components]
        return d

    def get_major_components(self) -> List[MapUnitComponent]:
        """Returns only the major components of the map unit."""
        return [c for c in self.components if c.is_major_component]


@dataclass
class PedonHorizon:
    """Represents a single horizon from a pedon with laboratory data."""
    pedon_key: str  # Foreign key to pedon
    layer_key: str  # Unique layer identifier
    layer_sequence: int  # Horizon sequence number
    horizon_name: str  # Horizon designation
    top_depth: float
    bottom_depth: float
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PedonData:
    """
    A complete pedon with site information and laboratory-analyzed horizons.
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
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the pedon to a dictionary."""
        d = asdict(self)
        d['horizons'] = [h.to_dict() for h in self.horizons]
        return d

    def get_horizon_by_depth(self, depth: float) -> Optional[PedonHorizon]:
        """Get the horizon that contains the specified depth."""
        for horizon in self.horizons:
            if horizon.top_depth <= depth < horizon.bottom_depth:
                return horizon
        return None

    def get_profile_depth(self) -> float:
        """Get the total depth of the pedon profile."""
        if not self.horizons:
            return 0.0
        return max(h.bottom_depth for h in self.horizons)