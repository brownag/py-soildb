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
    PedonHorizon,
    SoilMapUnit,
)


def _to_optional_float(value: Any) -> Optional[float]:
    """Converts a value to a float, returning None if it's NaN or None."""
    return float(value) if pd.notna(value) else None


def _to_optional_int(value: Any) -> Optional[int]:
    """Converts a value to an int, returning None if it's NaN or None."""
    return int(value) if pd.notna(value) else None


def _create_pedon_horizon_from_row(pedon_key: str, h_row: Any) -> PedonHorizon:
    """
    Create a PedonHorizon object from a horizon data row.

    This consolidates the column extraction logic used in pedon fetching functions.
    """
    total_carbon_ncs = h_row.get("total_carbon_ncs")
    organic_carbon_wb = h_row.get("organic_carbon_walkley_black")

    if pd.notna(total_carbon_ncs):
        organic_carbon = float(total_carbon_ncs)
    elif pd.notna(organic_carbon_wb):
        organic_carbon = float(organic_carbon_wb)
    else:
        organic_carbon = None

    return PedonHorizon(
        pedon_key=pedon_key,
        layer_key=str(h_row["layer_key"]),
        layer_sequence=_to_optional_int(h_row.get("layer_sequence")),
        horizon_name=str(h_row["hzn_desgn"]),
        top_depth=_to_optional_float(h_row.get("hzn_top")),
        bottom_depth=_to_optional_float(h_row.get("hzn_bot")),
        sand_total=_to_optional_float(h_row.get("sand_total")),
        silt_total=_to_optional_float(h_row.get("silt_total")),
        clay_total=_to_optional_float(h_row.get("clay_total")),
        texture_lab=str(h_row.get("texture_lab", "")),
        ph_h2o=_to_optional_float(h_row.get("ph_h2o")),
        organic_carbon=organic_carbon,
        calcium_carbonate=_to_optional_float(h_row.get("caco3_lt_2_mm")),
        bulk_density_third_bar=_to_optional_float(h_row.get("bulk_density_third_bar")),
        le_third_fifteen_lt2_mm=_to_optional_float(
            h_row.get("le_third_fifteen_lt2_mm")
        ),
        water_content_tenth_bar=_to_optional_float(
            h_row.get("water_retention_tenth_bar")
        ),
        water_content_third_bar=_to_optional_float(
            h_row.get("water_retention_third_bar")
        ),
        water_content_fifteen_bar=_to_optional_float(
            h_row.get("water_retention_15_bar")
        ),
    )


async def fetch_mapunit_struct_by_point(
    latitude: float,
    longitude: float,
    fill_components: bool = True,
    fill_horizons: bool = True,
    client: Optional[SDAClient] = None,
) -> SoilMapUnit:
    """
    Fetch a structured SoilMapUnit object for a specific geographic location.

    This function orchestrates multiple queries to build a complete, nested
    object representing the map unit, its components, and their aggregate horizons.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        fill_components: If True, fetch component data for the map unit.
        fill_horizons: If True, fetch aggregate horizon data for each component.
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
    map_unit = SoilMapUnit(
        map_unit_key=mukey,
        map_unit_name=str(first_row["muname"]),
        map_unit_symbol=str(first_row.get("musym", "")),
        survey_area_symbol=str(first_row.get("lkey", "")),
        survey_area_name=str(first_row.get("areaname", "")),
        metadata={
            "query_location": {"latitude": latitude, "longitude": longitude},
            "query_date": datetime.now().isoformat(),
        },
    )

    if not fill_components:
        return map_unit

    # Step 3: Fetch component data for this map unit
    comp_response = await fetch_component_by_mukey(
        mukey,
        columns=[
            "mukey",
            "cokey",
            "compname",
            "comppct_r",
            "majcompflag",
            "localphase",
            "drainagecl",
            "taxclname",
            "hydricrating",
            "compkind",
        ],
        client=client,
    )
    comp_df = comp_response.to_pandas()

    if comp_df.empty:
        # No components found, return map unit without components
        return map_unit

    # Step 4: Create MapUnitComponent objects
    components = []
    for _, row in comp_df.iterrows():
        components.append(
            MapUnitComponent(
                component_key=str(row["cokey"]),
                component_name=str(row["compname"]),
                component_percentage=float(row.get("comppct_r", 0)),
                is_major_component=row.get("majcompflag", "No").lower() == "yes",
                taxonomic_class=str(row.get("taxclname", "")),
                drainage_class=str(row.get("drainagecl", "")),
                local_phase=str(row.get("localphase", "")),
                hydric_rating=str(row.get("hydricrating", "")),
                component_kind=str(row.get("compkind", "")),
            )
        )
    map_unit.components = components

    if not fill_horizons or not components:
        return map_unit

    # Step 4: Fetch and attach aggregate horizons for all components in one call
    all_cokeys: List[Union[str, int]] = [c.component_key for c in components]
    horizons_df = (
        await fetch_chorizon_by_cokey(
            all_cokeys,
            columns=[
                "cokey",
                "chkey",
                "hzname",
                "hzdept_r",
                "hzdepb_r",
                "claytotal_l",
                "claytotal_r",
                "claytotal_h",
                "sandtotal_l",
                "sandtotal_r",
                "sandtotal_h",
                "om_l",
                "om_r",
                "om_h",
                "ph1to1h2o_l",
                "ph1to1h2o_r",
                "ph1to1h2o_h",
            ],
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
                properties = []
                prop_map = {
                    "clay": ("claytotal_l", "claytotal_r", "claytotal_h", "%"),
                    "sand": ("sandtotal_l", "sandtotal_r", "sandtotal_h", "%"),
                    "organic_matter": ("om_l", "om_r", "om_h", "%"),
                    "ph": ("ph1to1h2o_l", "ph1to1h2o_r", "ph1to1h2o_h", "pH"),
                }
                for name, (low, r, h, u) in prop_map.items():
                    if r in h_row and pd.notna(h_row[r]):
                        properties.append(
                            HorizonProperty(
                                property_name=name,
                                low=_to_optional_float(h_row.get(low)),
                                rv=float(h_row[r]),
                                high=_to_optional_float(h_row.get(h)),
                                unit=u,
                            )
                        )

                component.aggregate_horizons.append(
                    AggregateHorizon(
                        horizon_key=str(h_row["chkey"]),
                        horizon_name=str(h_row["hzname"]),
                        top_depth=float(h_row["hzdept_r"]),
                        bottom_depth=float(h_row["hzdepb_r"]),
                        properties=properties,
                    )
                )

    return map_unit


async def fetch_pedon_struct_by_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    fill_horizons: bool = True,
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
    for _, row in site_df.iterrows():
        pedon = PedonData(
            pedon_key=str(row["pedon_key"]),
            pedon_id=str(row.get("upedonid", "")),
            series=str(row.get("corr_name", "")),
            latitude=_to_optional_float(row.get("latitude_decimal_degrees")),
            longitude=_to_optional_float(row.get("longitude_decimal_degrees")),
            soil_classification=str(row.get("taxonname", "")),
            metadata={
                "query_bbox": {
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                },
                "query_date": datetime.now().isoformat(),
            },
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
            pedon = pedon_map.get(pedon_key)
            if pedon is None:
                continue
            pedon.horizons = [
                _create_pedon_horizon_from_row(pedon_key, h_row)
                for _, h_row in pedon_horizons_df.iterrows()
            ]

    return pedons


async def fetch_pedon_struct_by_id(
    pedon_id: str,
    fill_horizons: bool = True,
    client: Optional[SDAClient] = None,
) -> Optional[PedonData]:
    """
    Fetch structured pedon data for a specific pedon.

    Returns a PedonData object with site information and laboratory-analyzed horizons.

    Args:
        pedon_id: Pedon key or user pedon ID
        fill_horizons: If True, fetch horizon data for the pedon
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
    pedon = PedonData(
        pedon_key=str(row["pedon_key"]),
        pedon_id=str(row.get("upedonid", "")),
        series=str(row.get("corr_name", "")),
        latitude=_to_optional_float(row.get("latitude_decimal_degrees")),
        longitude=_to_optional_float(row.get("longitude_decimal_degrees")),
        soil_classification=str(row.get("taxonname", "")),
        metadata={
            "query_pedon_id": pedon_id,
            "query_date": datetime.now().isoformat(),
        },
    )

    if not fill_horizons:
        return pedon

    # Step 3: Fetch horizons
    pedon_key = pedon.pedon_key
    horizons_df = (await fetch_pedon_horizons(pedon_key, client=client)).to_pandas()

    if not horizons_df.empty:
        horizons = []
        for _, h_row in horizons_df.iterrows():
            horizon = _create_pedon_horizon_from_row(pedon_key, h_row)
            horizons.append(horizon)
        pedon.horizons = horizons

    return pedon
