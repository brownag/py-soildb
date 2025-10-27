"""
High-level functions for structured soil data access.

This module provides an object-oriented interface for accessing nested soil data
structures. Choose this module when you want:
- Nested relationships (Map Unit → Components → Horizons)
- An OOP interface with dataclass objects
- Automatic fetching and combining of related data
- Attributes instead of dictionary keys

ALTERNATIVES BY USE CASE:
- Want DataFrame for pandas operations? → Use convenience.get_*() functions
- Want raw SDA response? → Use convenience.get_*() functions
- Want low-level control? → Use Query() builder or query_templates
- Want spatial queries? → Use spatial_query()

API TIER REFERENCE (in order of preference):
1. HIGH-LEVEL (this module):
   - fetch_mapunit_struct_by_point()
   - fetch_pedon_struct_by_bbox()
   - fetch_pedon_struct_by_id()

2. MID-LEVEL (convenience):
   - get_mapunit_by_point()
   - get_mapunit_by_bbox()
   - get_lab_pedons_by_bbox()
   - get_lab_pedon_by_id()

3. LOW-LEVEL:
   - Query() / query_templates
   - spatial_query()

QUICK START:
```python
# Get structured map unit with components and horizons
mapunit = await fetch_mapunit_struct_by_point(latitude=42.0, longitude=-93.6)
for component in mapunit.components:
    print(f"{component.component_name}: {component.component_percentage}%")

# Get pedons with nested horizons
pedons = await fetch_pedon_struct_by_bbox(-94, 41.5, -93, 42.5)
for pedon in pedons:
    depth = pedon.get_profile_depth()
    print(f"Pedon {pedon.pedon_id}: {depth} cm deep")

# Get raw data for DataFrame operations
response = await get_mapunit_by_point(-93.6, 42.0)
df = response.to_pandas()
df.to_csv('mapunit.csv')
```
"""

from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd

from . import (
    fetch_pedon_horizons,
    get_lab_pedon_by_id,
    get_lab_pedons_by_bbox,
    get_mapunit_by_point,
)
from .fetch import fetch_by_keys
from .client import SDAClient
from .schema_system import (  # type: ignore
    AggregateHorizon,
    HorizonProperty,
    MapUnitComponent,
    PedonData,
    PedonHorizon,
    SoilMapUnit,
    get_schema,
)
from .utils import add_sync_version


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
    extra = processed.get("extra_fields", {})
    total_carbon_ncs = extra.get("total_carbon_ncs")
    organic_carbon_wb = extra.get("organic_carbon_walkley_black")
    caco3_lt_2_mm = extra.get("caco3_lt_2_mm")

    if pd.notna(total_carbon_ncs):
        # Calculate organic carbon from total carbon, subtracting carbonate carbon
        organic_carbon = float(total_carbon_ncs)
        if pd.notna(caco3_lt_2_mm):
            # Subtract carbon from calcium carbonate (CaCO3 is 12% carbon by weight)
            carbonate_carbon = float(caco3_lt_2_mm) * 0.12
            organic_carbon = max(
                0, organic_carbon - carbonate_carbon
            )  # Ensure non-negative

        processed["organic_carbon"] = organic_carbon
        extra.pop("total_carbon_ncs", None)
        extra.pop("organic_carbon_walkley_black", None)
        extra.pop("caco3_lt_2_mm", None)
    elif pd.notna(organic_carbon_wb):
        processed["organic_carbon"] = float(organic_carbon_wb)
        extra.pop("organic_carbon_walkley_black", None)
        extra.pop("total_carbon_ncs", None)
        extra.pop("caco3_lt_2_mm", None)
    else:
        processed["organic_carbon"] = None

    # Create the PedonHorizon object
    return PedonHorizon(  # type: ignore
        pedon_key=pedon_key,
        **{
            k: v for k, v in processed.items() if k not in ["extra_fields", "pedon_key"]
        },
        extra_fields=processed.get("extra_fields", {}),
    )


@add_sync_version
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

    Returns a NESTED OBJECT MODEL with Map Unit → Components → Aggregate Horizons.

    **USAGE EXAMPLES:**

    ```python
    # Get map unit with components and horizons
    mapunit = await fetch_mapunit_struct_by_point(
        latitude=42.0, 
        longitude=-93.6
    )
    
    # Access map unit properties
    print(f"Map Unit: {mapunit.map_unit_name}")
    print(f"Symbol: {mapunit.map_unit_symbol}")
    
    # Iterate through components
    for component in mapunit.components:
        print(f"  Component: {component.component_name}")
        print(f"  Percentage: {component.component_percentage}%")
        
        # Access component horizons
        for horizon in component.horizons:
            print(f"    Horizon {horizon.designation}: {horizon.texture_class}")
    
    # Get only map unit (no components/horizons)
    mapunit = await fetch_mapunit_struct_by_point(
        latitude=42.0, 
        longitude=-93.6,
        fill_components=False
    )
    
    # Compare with raw data API
    response = await get_mapunit_by_point(longitude=-93.6, latitude=42.0)
    df = response.to_pandas()
    # Now you have a DataFrame for pandas operations
    ```

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        fill_components: If True, fetch component data for the map unit.
                        Default: True. Set False for map unit only.
        fill_horizons: If True, fetch aggregate horizon data for each component.
                      Default: True. Only used if fill_components=True.
        component_columns: List of component columns to fetch. If None, uses default columns.
                          Extra columns beyond defaults will be stored in component extra_fields.
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance. Creates new client if not provided.

    Returns:
        SoilMapUnit: Object with nested components and horizons (if fill_* options enabled).
    
    Raises:
        ValueError: If no map unit found at specified location.
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
    comp_response = await fetch_by_keys(
        mukey,
        "component",
        "mukey",
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
        extra_fields = processed.get("extra_fields", {})

        # Helper function to get field value from processed or extra_fields
        def get_field(
            field_name: str,
            processed_dict: dict,
            extra_fields_dict: dict,
            alt_names: Optional[List[str]] = None,
        ) -> Any:
            if field_name in processed_dict:
                return processed_dict[field_name]
            if alt_names:
                for alt in alt_names:
                    if alt in extra_fields_dict:
                        return extra_fields_dict[alt]
            return extra_fields_dict.get(field_name)

        components.append(
            MapUnitComponent(
                component_key=get_field(
                    "component_key", processed, extra_fields, ["cokey"]
                ),
                component_name=get_field(
                    "component_name", processed, extra_fields, ["compname"]
                ),
                component_percentage=get_field(
                    "component_percentage", processed, extra_fields, ["comppct_r"]
                ),
                is_major_component=get_field(
                    "is_major_component", processed, extra_fields, ["majcompflag"]
                ),
                taxonomic_class=get_field(
                    "taxonomic_class", processed, extra_fields, ["taxclname"]
                ),
                drainage_class=get_field(
                    "drainage_class", processed, extra_fields, ["drainagecl"]
                ),
                local_phase=get_field(
                    "local_phase", processed, extra_fields, ["localphase"]
                ),
                hydric_rating=get_field(
                    "hydric_rating", processed, extra_fields, ["hydricrating"]
                ),
                component_kind=get_field(
                    "component_kind", processed, extra_fields, ["compkind"]
                ),
                extra_fields=extra_fields,
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
        await fetch_by_keys(
            all_cokeys,
            "chorizon",
            "cokey",
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
                extra_fields = processed.get("extra_fields", {})

                properties = []
                # Extract properties from extra_fields (since property columns have field_name=None)
                prop_data = {
                    "clay": ("claytotal_r", "claytotal_l", "claytotal_h", "%"),
                    "sand": ("sandtotal_r", "sandtotal_l", "sandtotal_h", "%"),
                    "organic_matter": ("om_r", "om_l", "om_h", "%"),
                    "ph": ("ph1to1h2o_r", "ph1to1h2o_l", "ph1to1h2o_h", "pH"),
                }

                for name, (rv_key, low_key, high_key, unit) in prop_data.items():
                    rv = extra_fields.get(rv_key)
                    if rv is not None:
                        low = extra_fields.get(low_key)
                        high = extra_fields.get(high_key)
                        properties.append(
                            HorizonProperty(
                                property_name=name,
                                low=low,
                                rv=rv,
                                high=high,
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
                        extra_fields=extra_fields,
                    )
                )

    return map_unit  # type: ignore


@add_sync_version
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
    
    **USAGE EXAMPLES:**

    ```python
    # Get all pedons in bounding box with horizons
    pedons = await fetch_pedon_struct_by_bbox(
        min_x=-94.0,
        min_y=41.5, 
        max_x=-93.0,
        max_y=42.5
    )
    
    # Analyze pedon structure
    for pedon in pedons:
        print(f"Pedon {pedon.pedon_id}:")
        print(f"  Taxon: {pedon.taxonname}")
        print(f"  Total depth: {pedon.get_profile_depth()} cm")
        
        # Access horizons
        for horizon in pedon.horizons:
            print(f"    {horizon.designation}: {horizon.texture_class}")
            for prop in horizon.properties:
                print(f"      {prop.property_name}: {prop.representative_value}")
    
    # Get pedons without horizons
    pedons = await fetch_pedon_struct_by_bbox(
        min_x=-94.0,
        min_y=41.5,
        max_x=-93.0,
        max_y=42.5,
        fill_horizons=False
    )
    
    # Compare with raw data API
    response = await get_lab_pedons_by_bbox(-94, 41.5, -93, 42.5)
    df = response.to_pandas()
    # Now use pandas for analysis: df.groupby(), df.plot(), etc.
    ```

    Args:
        min_x: Western boundary (longitude)
        min_y: Southern boundary (latitude)
        max_x: Eastern boundary (longitude)
        max_y: Northern boundary (latitude)
        fill_horizons: If True, fetch horizon data for each pedon. Default: True.
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance. Creates new client if not provided.

    Returns:
        List of PedonData objects (empty list if no pedons found).
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
            taxonname=processed.get("taxonname"),
            latitude=processed.get("latitude"),
            longitude=processed.get("longitude"),
            taxclname=processed.get("taxclname"),
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


@add_sync_version
async def fetch_pedon_struct_by_id(
    pedon_id: str,
    fill_horizons: bool = True,
    horizon_columns: Optional[List[str]] = None,
    client: Optional[SDAClient] = None,
) -> Optional[PedonData]:
    """
    Fetch structured pedon data for a specific pedon.

    Returns a PedonData object with site information and laboratory-analyzed horizons.

    **USAGE EXAMPLES:**

    ```python
    # Get single pedon with all horizons
    pedon = await fetch_pedon_struct_by_id("S1999NY061001")
    
    if pedon:
        print(f"Pedon: {pedon.pedon_id}")
        print(f"Taxon: {pedon.taxonname}")
        print(f"Location: ({pedon.latitude}, {pedon.longitude})")
        print(f"Total depth: {pedon.get_profile_depth()} cm")
        
        # Analyze horizons
        for horizon in pedon.horizons:
            print(f"  {horizon.designation}: {horizon.texture_class}")
            # Access horizon properties
            depth_at = pedon.get_horizon_by_depth(50)  # Get horizon at 50 cm
            if depth_at:
                print(f"    At 50 cm depth: {depth_at.designation}")
    
    # Get pedon without horizons
    pedon = await fetch_pedon_struct_by_id(
        "S1999NY061001",
        fill_horizons=False
    )
    
    # Compare with raw data API
    response = await get_lab_pedon_by_id("S1999NY061001")
    df = response.to_pandas()
    # Now use pandas for operations
    ```

    Args:
        pedon_id: Pedon key or user pedon ID
        fill_horizons: If True, fetch horizon data for the pedon. Default: True.
        horizon_columns: List of horizon columns to fetch. If None, uses default columns.
                        Extra columns beyond defaults will be stored in horizon extra_fields.
        client: Optional SDA client instance. Creates new client if not provided.

    Returns:
        PedonData object if found, None if not found.
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
        taxonname=processed.get("taxonname"),
        latitude=processed.get("latitude"),
        longitude=processed.get("longitude"),
        taxclname=processed.get("taxclname"),
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
