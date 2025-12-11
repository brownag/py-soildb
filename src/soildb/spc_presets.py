"""
Presets for common SoilProfileCollection column configurations.

This module provides preset configurations for converting SDA query results
to SoilProfileCollection objects. Use these presets to quickly configure
column mappings for common query types without manual configuration.

Example:
    >>> from soildb.spc_presets import StandardSDAHorizonColumns
    >>> from soildb import SDAClient, Query
    >>>
    >>> # Use preset for standard SDA horizon query
    >>> preset = StandardSDAHorizonColumns()
    >>>
    >>> async with SDAClient() as client:
    ...     query = Query().select(
    ...         *preset.get_all_columns()
    ...     ).from_("chorizon").limit(100)
    ...     response = await client.execute(query)
    ...     spc = response.to_soilprofilecollection(**preset.get_config())
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ColumnConfig:
    """Configuration for SoilProfileCollection column mapping.

    Attributes:
        site_id_col: Column name for site/component identifier
        horizon_id_col: Column name for unique horizon identifier
        horizon_top_col: Column name for horizon top depth (in inches)
        horizon_bottom_col: Column name for horizon bottom depth (in inches)
        description: Human-readable description of this configuration
        required_columns: List of all required columns for this configuration
        optional_columns: List of commonly included optional columns
    """

    site_id_col: str
    horizon_id_col: str
    horizon_top_col: str
    horizon_bottom_col: str
    description: str = ""
    required_columns: Optional[List[str]] = None
    optional_columns: Optional[List[str]] = None

    def get_config_dict(self) -> Dict[str, str]:
        """Get configuration as dictionary for to_soilprofilecollection()."""
        return {
            "site_id_col": self.site_id_col,
            "hz_id_col": self.horizon_id_col,
            "hz_top_col": self.horizon_top_col,
            "hz_bot_col": self.horizon_bottom_col,
        }

    def get_required_columns(self) -> List[str]:
        """Get list of all required columns for this configuration."""
        if self.required_columns is None:
            return [
                self.site_id_col,
                self.horizon_id_col,
                self.horizon_top_col,
                self.horizon_bottom_col,
            ]
        return self.required_columns

    def get_all_columns(self) -> List[str]:
        """Get list of all required + optional columns."""
        required = self.get_required_columns()
        optional = self.optional_columns or []
        return list(dict.fromkeys(required + optional))  # Remove duplicates, preserve order


class StandardSDAHorizonColumns(ColumnConfig):
    """
    Standard SDA horizon columns (USDA-NRCS standard naming).

    Use this preset when querying the chorizon table with standard columns.

    Required columns:
    - cokey: Component key (site identifier)
    - chkey: Child horizon key (unique horizon identifier)
    - hzdept_r: Horizon top depth (representative, in inches)
    - hzdepb_r: Horizon bottom depth (representative, in inches)

    Common optional columns:
    - hzname: Horizon name (e.g., "A", "Bt")
    - claytotal_r: Total clay percentage
    - sandtotal_r: Total sand percentage
    - silttotal_r: Total silt percentage
    - awc_r: Available water capacity (in/in)
    - ksat_r: Saturated hydraulic conductivity (µm/s)

    Example:
        >>> preset = StandardSDAHorizonColumns()
        >>> # Get just required columns
        >>> required = preset.get_required_columns()
        >>> # Get all common columns
        >>> all_cols = preset.get_all_columns()
    """

    def __init__(self) -> None:
        super().__init__(
            site_id_col="cokey",
            horizon_id_col="chkey",
            horizon_top_col="hzdept_r",
            horizon_bottom_col="hzdepb_r",
            description="Standard USDA-NRCS horizon columns from chorizon table",
            required_columns=["cokey", "chkey", "hzdept_r", "hzdepb_r"],
            optional_columns=[
                "hzname", "claytotal_r", "sandtotal_r", "silttotal_r",
                "awc_r", "ksat_r", "om_r", "dbthirdbar_r", "wthirdbar_r"
            ],
        )


class LabPedonHorizonColumns(ColumnConfig):
    """
    Laboratory pedon horizon columns (USDA soil characterization data).

    Use this preset when working with pedon/site horizon data from soil
    characterization laboratories.

    Required columns:
    - siteiid: Site IISYM (site identifier)
    - pedon_id: Pedon identifier (used as site ID for SPC)
    - hzname: Horizon name (serves as unique identifier)
    - hzdept_r: Horizon top depth
    - hzdepb_r: Horizon bottom depth

    Note: Lab pedon data may have different column names. Verify with your
    data source and use CustomColumnConfig if names differ.

    Example:
        >>> preset = LabPedonHorizonColumns()
        >>> config = preset.get_config_dict()
    """

    def __init__(self) -> None:
        super().__init__(
            site_id_col="pedon_id",
            horizon_id_col="hzname",
            horizon_top_col="hzdept_r",
            horizon_bottom_col="hzdepb_r",
            description="Laboratory pedon soil characterization horizon columns",
            required_columns=["pedon_id", "hzname", "hzdept_r", "hzdepb_r"],
            optional_columns=[
                "siteiid", "total_carbon_ncs", "carbonate_carbon_volumetric",
                "claytotal_psa", "sandtotal_psa", "silttotal_psa",
                "organic_carbon_walkley_black"
            ],
        )


class PedonSiteHorizonColumns(ColumnConfig):
    """
    Pedon site horizon columns (for pedon_id + pedon_key scenario).

    Use this when you have both pedon_id (site identifier) and pedon_key
    or pedon_key_horizon (unique horizon identifier).

    Required columns:
    - pedon_id: Pedon identifier (site ID)
    - pedon_key_horizon: Unique horizon key
    - hzdept_r: Horizon top depth (in inches)
    - hzdepb_r: Horizon bottom depth (in inches)

    Example:
        >>> preset = PedonSiteHorizonColumns()
    """

    def __init__(self) -> None:
        super().__init__(
            site_id_col="pedon_id",
            horizon_id_col="pedon_key_horizon",
            horizon_top_col="hzdept_r",
            horizon_bottom_col="hzdepb_r",
            description="Pedon site horizon columns using pedon_key_horizon",
            required_columns=["pedon_id", "pedon_key_horizon", "hzdept_r", "hzdepb_r"],
            optional_columns=["hzname", "hzdom", "hzsubname", "distinctness", "topography"],
        )


class MapunitComponentHorizonColumns(ColumnConfig):
    """
    Map unit component horizon columns (for mapunit -> component -> horizon workflow).

    Use this when you have joined mapunit, component, and chorizon tables.

    Required columns:
    - mukey: Map unit key (mapped to site_id)
    - cokey: Component key (horizon identifier)
    - hzdept_r: Horizon top depth
    - hzdepb_r: Horizon bottom depth

    Note: This configuration groups by mukey, so each map unit becomes a "site"
    in the SoilProfileCollection.

    Example:
        >>> preset = MapunitComponentHorizonColumns()
    """

    def __init__(self) -> None:
        super().__init__(
            site_id_col="mukey",
            horizon_id_col="cokey",
            horizon_top_col="hzdept_r",
            horizon_bottom_col="hzdepb_r",
            description="Map unit component horizon columns (mapunit -> component -> horizon join)",
            required_columns=["mukey", "cokey", "hzdept_r", "hzdepb_r"],
            optional_columns=["musym", "muname", "compname", "comppct_r", "hzname"],
        )


class CustomColumnConfig(ColumnConfig):
    """
    Custom column configuration for non-standard data sources.

    Use this when your data has different column names than the standard
    SDA or lab naming conventions.

    Args:
        site_id_col: Column name for site/profile identifier
        horizon_id_col: Column name for unique horizon identifier
        horizon_top_col: Column name for top depth
        horizon_bottom_col: Column name for bottom depth
        description: Optional description of this configuration
        required_columns: Optional list of all required columns
        optional_columns: Optional list of commonly included columns

    Example:
        >>> # For custom data with different column names
        >>> preset = CustomColumnConfig(
        ...     site_id_col="profile_id",
        ...     horizon_id_col="layer_id",
        ...     horizon_top_col="depth_from_cm",
        ...     horizon_bottom_col="depth_to_cm",
        ...     description="Custom profile data in centimeters"
        ... )
    """

    def __init__(
        self,
        site_id_col: str,
        horizon_id_col: str,
        horizon_top_col: str,
        horizon_bottom_col: str,
        description: str = "Custom column configuration",
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None,
    ):
        super().__init__(
            site_id_col=site_id_col,
            horizon_id_col=horizon_id_col,
            horizon_top_col=horizon_top_col,
            horizon_bottom_col=horizon_bottom_col,
            description=description,
            required_columns=required_columns,
            optional_columns=optional_columns,
        )


# Registry of all built-in presets
PRESET_REGISTRY = {
    "standard_sda": StandardSDAHorizonColumns,
    "lab_pedon": LabPedonHorizonColumns,
    "pedon_site": PedonSiteHorizonColumns,
    "mapunit_component": MapunitComponentHorizonColumns,
}


def get_preset(name: str) -> ColumnConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Name of the preset ('standard_sda', 'lab_pedon', 'pedon_site',
              'mapunit_component')

    Returns:
        ColumnConfig instance

    Raises:
        ValueError: If preset name is not found

    Example:
        >>> preset = get_preset('standard_sda')
        >>> config = preset.get_config_dict()
    """
    if name not in PRESET_REGISTRY:
        available = ", ".join(PRESET_REGISTRY.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    return PRESET_REGISTRY[name]()


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions

    Example:
        >>> presets = list_presets()
        >>> for name, desc in presets.items():
        ...     print(f"{name}: {desc}")
    """
    presets = {}
    for name, preset_class in PRESET_REGISTRY.items():
        preset = preset_class()
        presets[name] = preset.description
    return presets
