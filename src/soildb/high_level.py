"""
High-level functions for the soildb package that return structured
data objects from soildb.models.
"""

from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd

from . import (
    fetch_chorizon_by_cokey,
    fetch_component_by_mukey,
    fetch_pedon_horizons,
    get_lab_pedon_by_id,
    get_lab_pedons_by_bbox,
    get_mapunit_by_point,
)
from .client import SDAClient
from .models import (
    AggregateHorizon,
    HorizonProperty,
    MapUnitComponent,
    PedonData,
    SoilMapUnit,
)
from .schema_system import get_schema, PedonHorizon  # type: ignore


def _create_pedon_horizon_from_row(
    pedon_key: str, h_row: Any, requested_columns: Optional[List[str]] = None
) -> Any:
    """
    Create a PedonHorizon object from a horizon data row.

    This consolidates the column extraction logic used in pedon fetching functions.
    Now supports preserving extra columns beyond the default set.

    Args:
        pedon_key: The pedon key for this horizon
        h_row: The data row containing horizon information
        requested_columns: List of columns that were requested (for tracking extra columns)
    """
    # Get the pedon horizon schema
    schema = get_schema("pedon_horizon")
    if not schema:
        raise ValueError("Pedon horizon schema not found")

    # Process the row using the schema
    processed = schema.process_row(h_row, requested_columns)

    # Handle special processing for organic_carbon (combine total_carbon_ncs and organic_carbon_walkley_black)
    total_carbon_ncs = processed.get("total_carbon_ncs")
    organic_carbon_wb = processed.get("organic_carbon_walkley_black")

    if pd.notna(total_carbon_ncs):
        processed["organic_carbon"] = float(total_carbon_ncs)  # type: ignore
    elif pd.notna(organic_carbon_wb):
        processed["organic_carbon"] = float(organic_carbon_wb)  # type: ignore
    else:
        processed["organic_carbon"] = None

    # Remove the raw carbon columns from processed data
    processed.pop("total_carbon_ncs", None)
    processed.pop("organic_carbon_walkley_black", None)

    # Create the PedonHorizon object
    return PedonHorizon(  # type: ignore
        pedon_key=pedon_key,
        **{
            k: v for k, v in processed.items() if k not in ["extra_fields", "pedon_key"]
        },
        extra_fields=processed.get("extra_fields", {}),
    )


async def fetch_mapunit_struct_by_point(
    latitude: float,
    longitude: float,
    fill_components: bool = True,
    fill_horizons: bool = True,
    component_columns: Optional[List[str]] = None,
    horizon_columns: Optional[List[str]] = None,
    client: Optional[SDAClient] = None,
) -> SoilMapUnit:  # type: ignore
    """
    Fetch a structured SoilMapUnit object for a specific geographic location.

    This function orchestrates multiple queries to build a complete, nested
    object representing the map unit, its components, and their aggregate horizons.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        fill_components: If True, fetch component data for the map unit.
        fill_horizons: If True, fetch aggregate horizon data for each component.
        component_columns: List of component columns to fetch. If None, uses default columns.
                          Extra columns beyond defaults will be stored in component extra_fields.
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance.

    Returns:
        A SoilMapUnit object.
    """
    # Step 1: Get map unit data
    mu_response = await get_mapunit_by_point(longitude, latitude, client=client)
    mu_df = mu_response.to_pandas()

    if mu_df.empty:
        raise ValueError(f"No map unit found at location ({latitude}, {longitude})")

    # Step 2: Create the base SoilMapUnit object
    first_row = mu_df.iloc[0]
    mukey = str(first_row["mukey"])

    # Track column information
    metadata = {
        "query_location": {"latitude": latitude, "longitude": longitude},
        "query_date": datetime.now().isoformat(),
    }

    # Add column tracking if custom columns were requested
    if component_columns or horizon_columns:
        metadata["requested_columns"] = {
            "component_columns": component_columns,
            "horizon_columns": horizon_columns,
        }
        # Get default columns from schemas
        comp_schema = get_schema("component")
        hz_schema = get_schema("chorizon")
        metadata["default_columns"] = {
            "component_columns": comp_schema.get_default_columns()
            if comp_schema
            else [],
            "horizon_columns": hz_schema.get_default_columns() if hz_schema else [],
        }

    map_unit = SoilMapUnit(  # type: ignore
        map_unit_key=mukey,
        map_unit_name=str(first_row["muname"]),
        map_unit_symbol=str(first_row.get("musym", "")),
        survey_area_symbol=str(first_row.get("lkey", "")),
        survey_area_name=str(first_row.get("areaname", "")),
        extra_fields=metadata,
    )

    if not fill_components:
        return map_unit  # type: ignore

    # Step 3: Fetch component data for this map unit
    comp_schema = get_schema("component")
    if not comp_schema:
        raise ValueError("Component schema not found")

    comp_columns = component_columns or comp_schema.get_default_columns()
    comp_response = await fetch_component_by_mukey(
        mukey,
        columns=comp_columns,
        client=client,
    )
    comp_df = comp_response.to_pandas()

    if comp_df.empty:
        # No components found, return map unit without components
        return map_unit  # type: ignore

    # Step 4: Create MapUnitComponent objects
    components = []
    for _, row in comp_df.iterrows():
        # Process row using schema
        processed = comp_schema.process_row(row, component_columns)

        components.append(
            MapUnitComponent(
                component_key=processed["component_key"],
                component_name=processed["component_name"],
                component_percentage=processed["component_percentage"],
                is_major_component=processed["is_major_component"],
                taxonomic_class=processed.get("taxonomic_class"),
                drainage_class=processed.get("drainage_class"),
                local_phase=processed.get("local_phase"),
                hydric_rating=processed.get("hydric_rating"),
                component_kind=processed.get("component_kind"),
                extra_fields=processed.get("extra_fields", {}),
            )
        )
    map_unit.components = components

    if not fill_horizons or not components:
        return map_unit  # type: ignore

    # Step 4: Fetch and attach aggregate horizons for all components in one call
    all_cokeys: List[Union[str, int]] = [c.component_key for c in components]
    hz_schema = get_schema("chorizon")
    if not hz_schema:
        raise ValueError("Chorizon schema not found")

    hz_columns = horizon_columns or hz_schema.get_default_columns()
    # Ensure cokey is always included for grouping
    if "cokey" not in hz_columns:
        hz_columns.insert(0, "cokey")

    horizons_df = (
        await fetch_chorizon_by_cokey(
            all_cokeys,
            columns=hz_columns,
            client=client,
        )
    ).to_pandas()

    if not horizons_df.empty:
        horizons_df["cokey"] = horizons_df["cokey"].astype(str)
        comp_map = {c.component_key: c for c in map_unit.components}
        for cokey, comp_horizons_df in horizons_df.groupby("cokey"):
            component = comp_map.get(cokey)
            if not component:
                continue
            for _, h_row in comp_horizons_df.iterrows():
                # Process horizon row using schema
                processed = hz_schema.process_row(h_row, horizon_columns)

                properties = []
                # Extract properties from processed data
                prop_data = {
                    "clay": ("claytotal_r", "claytotal_l", "claytotal_h", "%"),
                    "sand": ("sandtotal_r", "sandtotal_l", "sandtotal_h", "%"),
                    "organic_matter": ("om_r", "om_l", "om_h", "%"),
                    "ph": ("ph1to1h2o_r", "ph1to1h2o_l", "ph1to1h2o_h", "pH"),
                }

                for name, (rv_key, low_key, high_key, unit) in prop_data.items():
                    if rv_key in processed and processed[rv_key] is not None:
                        properties.append(
                            HorizonProperty(
                                property_name=name,
                                low=processed.get(low_key),
                                rv=processed[rv_key],
                                high=processed.get(high_key),
                                unit=unit,
                            )
                        )

                component.aggregate_horizons.append(
                    AggregateHorizon(
                        horizon_key=processed["horizon_key"],
                        horizon_name=processed["horizon_name"],
                        top_depth=processed["top_depth"],
                        bottom_depth=processed["bottom_depth"],
                        properties=properties,
                        extra_fields=processed.get("extra_fields", {}),
                    )
                )

    return map_unit  # type: ignore


async def fetch_pedon_struct_by_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    fill_horizons: bool = True,
    horizon_columns: Optional[List[str]] = None,
    client: Optional[SDAClient] = None,
) -> List[PedonData]:
    """
    Fetch structured pedon data within a bounding box.

    Returns a list of PedonData objects with site information and laboratory-analyzed horizons.

    Args:
        min_x: Western boundary (longitude)
        min_y: Southern boundary (latitude)
        max_x: Eastern boundary (longitude)
        max_y: Northern boundary (latitude)
        fill_horizons: If True, fetch horizon data for each pedon
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance.

    Returns:
        List of PedonData objects
    """
    # Step 1: Get pedon site data
    site_response = await get_lab_pedons_by_bbox(
        min_x, min_y, max_x, max_y, client=client
    )
    site_df = site_response.to_pandas()

    if site_df.empty:
        return []

    # Step 2: Create base PedonData objects
    pedons = []
    pedon_schema = get_schema("pedon")
    if not pedon_schema:
        raise ValueError("Pedon schema not found")

    for _, row in site_df.iterrows():
        # Process row using schema
        processed = pedon_schema.process_row(row)

        # Track column information
        metadata = {
            "query_bbox": {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            },
            "query_date": datetime.now().isoformat(),
        }

        # Add column tracking if custom columns were requested
        if horizon_columns:
            metadata["requested_columns"] = {
                "horizon_columns": horizon_columns,
            }
            hz_schema = get_schema("pedon_horizon")
            metadata["default_columns"] = {
                "horizon_columns": hz_schema.get_default_columns() if hz_schema else [],
            }

        # Merge metadata into extra_fields
        processed["extra_fields"].update(metadata)

        pedon = PedonData(
            pedon_key=processed["pedon_key"],
            pedon_id=processed["pedon_id"],
            series=processed.get("series"),
            latitude=processed.get("latitude"),
            longitude=processed.get("longitude"),
            soil_classification=processed.get("soil_classification"),
            extra_fields=processed.get("extra_fields", {}),
        )
        pedons.append(pedon)

    if not fill_horizons or not pedons:
        return pedons

    # Step 3: Fetch horizons for all pedons
    pedon_keys = [p.pedon_key for p in pedons]
    horizons_df = (await fetch_pedon_horizons(pedon_keys, client=client)).to_pandas()

    if not horizons_df.empty:
        # Convert pedon_key to string for matching
        horizons_df["pedon_key"] = horizons_df["pedon_key"].astype(str)
        pedon_map = {p.pedon_key: p for p in pedons}

        for pedon_key, pedon_horizons_df in horizons_df.groupby("pedon_key"):
            pedon_obj = pedon_map.get(pedon_key)
            if pedon_obj is None:
                continue
            pedon_obj.horizons = [
                _create_pedon_horizon_from_row(pedon_key, h_row, horizon_columns)
                for _, h_row in pedon_horizons_df.iterrows()
            ]

    return pedons


async def fetch_pedon_struct_by_id(
    pedon_id: str,
    fill_horizons: bool = True,
    horizon_columns: Optional[List[str]] = None,
    client: Optional[SDAClient] = None,
) -> Optional[PedonData]:
    """
    Fetch structured pedon data for a specific pedon.

    Returns a PedonData object with site information and laboratory-analyzed horizons.

    Args:
        pedon_id: Pedon key or user pedon ID
        fill_horizons: If True, fetch horizon data for the pedon
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance.

    Returns:
        PedonData object or None if not found
    """
    # Step 1: Get pedon site data
    site_response = await get_lab_pedon_by_id(pedon_id, client=client)
    site_df = site_response.to_pandas()

    if site_df.empty:
        return None

    # Step 2: Create PedonData object
    row = site_df.iloc[0]
    pedon_schema = get_schema("pedon")
    if not pedon_schema:
        raise ValueError("Pedon schema not found")

    processed = pedon_schema.process_row(row)

    # Track column information
    metadata = {
        "query_pedon_id": pedon_id,
        "query_date": datetime.now().isoformat(),
    }

    # Add column tracking if custom columns were requested
    if horizon_columns:
        metadata["requested_columns"] = {  # type: ignore
            "horizon_columns": horizon_columns,
        }
        hz_schema = get_schema("pedon_horizon")
        metadata["default_columns"] = {  # type: ignore
            "horizon_columns": hz_schema.get_default_columns() if hz_schema else [],
        }

    # Merge metadata into extra_fields
    processed["extra_fields"].update(metadata)

    pedon = PedonData(
        pedon_key=processed["pedon_key"],
        pedon_id=processed["pedon_id"],
        series=processed.get("series"),
        latitude=processed.get("latitude"),
        longitude=processed.get("longitude"),
        soil_classification=processed.get("soil_classification"),
        extra_fields=processed.get("extra_fields", {}),
    )

    if not fill_horizons:
        return pedon

    # Step 3: Fetch horizons
    pedon_key = pedon.pedon_key
    horizons_df = (await fetch_pedon_horizons(pedon_key, client=client)).to_pandas()

    if not horizons_df.empty:
        horizons = []
        for _, h_row in horizons_df.iterrows():
            horizon = _create_pedon_horizon_from_row(pedon_key, h_row, horizon_columns)
            horizons.append(horizon)
        pedon.horizons = horizons

    return pedon
