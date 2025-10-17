"""
Data Models for the soildb package.

These dataclasses provide structured, object-oriented representations for
common soil science data entities like Map Units, Components, and Horizons.

Most models are dynamically generated from schemas in schema_system.py for
flexibility and maintainability. See schema_system.py for schema definitions.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Import dynamically generated models from schema_system
from .schema_system import (
    AggregateHorizon,  # type: ignore
    HorizonProperty,  # type: ignore
    MapUnitComponent,  # type: ignore
    PedonHorizon,  # type: ignore
    SoilMapUnit,  # type: ignore
)


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
    horizons: List[Any] = field(default_factory=list)
    # --- ADDED ---
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


# Export all models for public API
__all__ = [
    "AggregateHorizon",
    "HorizonProperty",
    "MapUnitComponent",
    "PedonData",
    "PedonHorizon",
    "SoilMapUnit",
]
