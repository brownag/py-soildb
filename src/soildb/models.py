"""
Data Models for the soildb package.

DEPRECATED: This module is maintained for backward compatibility only.
All models have been consolidated into schema_system.py.

New code should import directly from schema_system:
    from soildb.schema_system import (
        PedonData,
        AggregateHorizon,
        HorizonProperty,
        MapUnitComponent,
        PedonHorizon,
        SoilMapUnit,
    )

This compatibility layer will be removed in v0.4.0.
"""

import warnings

# Import all models from schema_system (single source of truth)
from .schema_system import (
    AggregateHorizon,  # type: ignore
    HorizonProperty,  # type: ignore
    MapUnitComponent,  # type: ignore
    PedonData,  # type: ignore
    PedonHorizon,  # type: ignore
    SoilMapUnit,  # type: ignore
)

# Show deprecation warning when this module is imported
warnings.warn(
    "The 'soildb.models' module is deprecated and will be removed in v0.4.0. "
    "Please import directly from 'soildb.schema_system' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export all models for public API
__all__ = [
    "AggregateHorizon",
    "HorizonProperty",
    "MapUnitComponent",
    "PedonData",
    "PedonHorizon",
    "SoilMapUnit",
]
