"""
Helper functions for soil water availability modeling with AWDB data.

This module provides workflows that combine:
- SDA soil properties (texture, available water capacity, depth to water table, permeability)
- AWDB environmental monitoring (soil moisture, precipitation, soil temperature from SCAN/SNOTEL)

Use these functions to:
- Validate SDA available water capacity predictions against SCAN field measurements
- Monitor soil water status and stress conditions
- Assess seasonal water availability dynamics
- Optimize irrigation scheduling based on soil and weather data
"""

from typing import Any, Dict, List, Optional

from .awdb import convenience as awdb_convenience
from .awdb.exceptions import AWDBError
from .client import SDAClient
from .query import Query
from .response import SDAResponse



async def get_component_water_properties(
    areasymbol: str,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Fetch soil water-related properties for a survey area.

    Properties (horizon-level) include:
    - Available water capacity (awc_r) - water available for plant growth
    - Saturated hydraulic conductivity (ksat_r) - drainage/infiltration
    - Organic matter (om_r) - affects water holding capacity
    - Texture (clay, silt, sand percentages)
    
    Also includes component and mapunit info.

    Args:
        areasymbol: NRCS survey area symbol (e.g., 'IA109')
        client: Optional SDAClient instance

    Returns:
        SDAResponse with horizon-level water properties
    """
    close_client = False
    if client is None:
        client = SDAClient()
        close_client = True

    try:
        query = (
            Query()
            .select(
                "mu.mukey",
                "mu.muname",
                "mu.areasymbol",
                "c.cokey",
                "c.compname",
                "c.comppct_r",
                "ch.chkey",
                "ch.hzname",
                "ch.hzdept_r",
                "ch.hzdepb_r",
                "ch.claytotal_r",
                "ch.silttotal_r",
                "ch.sandtotal_r",
                "ch.awc_r",  # Available water capacity (inches/inch of soil)
                "ch.ksat_r",  # Saturated hydraulic conductivity
                "ch.om_r",  # Organic matter
            )
            .from_("mapunit mu")
            .inner_join("component c", "mu.mukey = c.mukey")
            .inner_join("chorizon ch", "c.cokey = ch.cokey")
            .where(f"mu.areasymbol = '{areasymbol}'")
            .order_by("c.cokey, ch.hzdept_r")
        )

        return await client.execute(query)
    finally:
        if close_client:
            await client.aclose()


async def get_water_table_depth(
    areasymbol: str,
    use_mapunit_agg: bool = True,
    client: Optional[SDAClient] = None,
) -> SDAResponse:
    """
    Get water table depth information for a survey area.

    Can query from two sources:
    1. muaggatt (mapunit-level aggregated attributes) - simpler, representative values
    2. comonth (component-month data) - component-level seasonal variation

    Args:
        areasymbol: NRCS survey area symbol (e.g., 'IA109')
        use_mapunit_agg: If True, use muaggatt (mapunit level). If False, use comonth (component level)
        client: Optional SDAClient instance

    Returns:
        SDAResponse with water table depth data
    """
    close_client = False
    if client is None:
        client = SDAClient()
        close_client = True

    try:
        if use_mapunit_agg:
            # Mapunit-level aggregated attributes (representative values)
            query = (
                Query()
                .select(
                    "mu.mukey",
                    "mu.muname",
                    "mu.areasymbol",
                    "ma.wtdepannmin",  # Min depth to water (annual)
                    "ma.wtdepannmax",  # Max depth to water (annual)
                    "ma.flodfreqdcd",  # Flood frequency code
                    "ma.flodfreqmntn",  # Flood frequency month name
                )
                .from_("mapunit mu")
                .left_join("muaggatt ma", "mu.mukey = ma.mukey")
                .where(f"mu.areasymbol = '{areasymbol}'")
            )
        else:
            # Component-month level (seasonal variation in water table)
            query = (
                Query()
                .select(
                    "mu.mukey",
                    "mu.muname",
                    "mu.areasymbol",
                    "c.cokey",
                    "c.compname",
                    "c.comppct_r",
                    "cm.month",
                    "cm.wtdepannmin",  # Min depth to water for this month
                    "cm.wtdepannmax",  # Max depth to water for this month
                    "cm.flodfreqcl",  # Flood frequency class
                )
                .from_("mapunit mu")
                .inner_join("component c", "mu.mukey = c.mukey")
                .left_join("comonth cm", "c.cokey = cm.cokey")
                .where(f"mu.areasymbol = '{areasymbol}'")
                .order_by("c.cokey, cm.month")
            )

        return await client.execute(query)
    finally:
        if close_client:
            await client.aclose()


async def get_scan_soil_moisture_profile(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
) -> Dict[str, Any]:
    """
    Fetch soil moisture measurements from nearby SCAN station.

    SCAN stations typically measure soil moisture at multiple depths
    (common: 2, 4, 8, 16, 20 inches below surface).

    Args:
        latitude: Target latitude
        longitude: Target longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_distance_km: Max search distance for SCAN station

    Returns:
        Dictionary with soil moisture time series at available depths
    """
    try:
        # Get monitoring data from nearest SCAN station
        data = await awdb_convenience.get_monitoring_station_data(
            latitude=latitude,
            longitude=longitude,
            property_name="soil_moisture",
            start_date=start_date,
            end_date=end_date,
            max_distance_km=max_distance_km,
            network_codes=["SCAN"],  # Focus on SCAN network
            auto_select_sensor=True,
        )
        return data
    except AWDBError as e:
        raise AWDBError(
            f"Could not fetch SCAN soil moisture data: {e}"
        ) from e


async def get_scan_soil_temperature_profile(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
) -> Dict[str, Any]:
    """
    Fetch soil temperature measurements from nearby SCAN station.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_distance_km: Max search distance for SCAN station

    Returns:
        Dictionary with soil temperature time series at available depths
    """
    try:
        data = await awdb_convenience.get_monitoring_station_data(
            latitude=latitude,
            longitude=longitude,
            property_name="soil_temp",
            start_date=start_date,
            end_date=end_date,
            max_distance_km=max_distance_km,
            network_codes=["SCAN"],
            auto_select_sensor=True,
        )
        return data
    except AWDBError as e:
        raise AWDBError(
            f"Could not fetch SCAN soil temperature data: {e}"
        ) from e


async def get_precipitation_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    max_distance_km: float = 50.0,
) -> Dict[str, Any]:
    """
    Fetch precipitation data from nearby monitoring station.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_distance_km: Max search distance

    Returns:
        Dictionary with precipitation time series
    """
    try:
        data = await awdb_convenience.get_monitoring_station_data(
            latitude=latitude,
            longitude=longitude,
            property_name="precipitation",
            start_date=start_date,
            end_date=end_date,
            max_distance_km=max_distance_km,
            auto_select_sensor=True,
        )
        return data
    except AWDBError as e:
        raise AWDBError(
            f"Could not fetch precipitation data: {e}"
        ) from e


async def estimate_water_availability(
    areasymbol: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Estimate soil water availability by combining SDA predictions with AWDB measurements.

    Compares:
    - SDA available water capacity (potential from horizons)
    - SDA depth to water table (groundwater proximity)
    - AWDB soil moisture measurements (actual from SCAN)
    - Precipitation input
    - Soil temperature (affects plant availability)

    Args:
        areasymbol: Survey area symbol
        latitude: Representative latitude
        longitude: Representative longitude
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Dictionary with water availability analysis
    """
    # Fetch SDA soil properties (horizons)
    soil_response = await get_component_water_properties(areasymbol)
    soil_df = soil_response.to_pandas()

    if soil_df.empty:
        raise ValueError(f"No soil data found for {areasymbol}")

    # Fetch water table depth (mapunit-level aggregates)
    wtable_response = await get_water_table_depth(areasymbol, use_mapunit_agg=True)
    wtable_df = wtable_response.to_pandas()

    # Fetch AWDB monitoring data
    moisture_data = await get_scan_soil_moisture_profile(
        latitude, longitude, start_date, end_date
    )
    precip_data = await get_precipitation_data(latitude, longitude, start_date, end_date)
    temp_data = await get_scan_soil_temperature_profile(
        latitude, longitude, start_date, end_date
    )

    # Calculate summary statistics from horizon data
    soil_summary = {
        "n_horizons": len(soil_df),
        "n_components": soil_df["cokey"].nunique(),
        "avg_awc_in_per_in": soil_df["awc_r"].mean(),
        "avg_ksat": soil_df["ksat_r"].mean(),
        "avg_organic_matter": soil_df["om_r"].mean(),
        "texture_distribution": {
            "avg_clay_pct": soil_df["claytotal_r"].mean(),
            "avg_silt_pct": soil_df["silttotal_r"].mean(),
            "avg_sand_pct": soil_df["sandtotal_r"].mean(),
        },
        "depth_range": {
            "shallow_ft": soil_df["hzdept_r"].min() / 12,  # Convert to feet
            "deep_ft": soil_df["hzdepb_r"].max() / 12,
        }
    }

    # Add water table depth if available
    if not wtable_df.empty:
        soil_summary["water_table"] = {
            "min_depth_ft": wtable_df["wtdepannmin"].mean() / 12 if "wtdepannmin" in wtable_df.columns else None,
            "max_depth_ft": wtable_df["wtdepannmax"].mean() / 12 if "wtdepannmax" in wtable_df.columns else None,
        }

    # Extract measurements from AWDB data
    moisture_points = moisture_data.get("data_points", [])
    precip_points = precip_data.get("data_points", [])
    temp_points = temp_data.get("data_points", [])

    return {
        "survey_area": areasymbol,
        "monitoring_station": moisture_data.get("site_name", "Unknown"),
        "distance_km": moisture_data.get("metadata", {}).get("distance_km", None),
        "soil_properties": soil_summary,
        "awdb_measurements": {
            "soil_moisture": {
                "n_observations": len(moisture_points),
                "site_id": moisture_data.get("site_id"),
                "network": moisture_data.get("metadata", {}).get("network"),
            },
            "precipitation": {
                "n_observations": len(precip_points),
                "site_id": precip_data.get("site_id"),
                "total_inches": sum(
                    p.get("value", 0) for p in precip_points if p.get("value")
                ),
            },
            "soil_temperature": {
                "n_observations": len(temp_points),
                "site_id": temp_data.get("site_id"),
            },
        },
        "analysis_period": {
            "start_date": start_date,
            "end_date": end_date,
        },
    }


def get_water_stress_categories(
    awc_in_per_in: float, current_moisture_pct: float
) -> Dict[str, Any]:
    """
    Assess soil water stress condition given AWC and current moisture.

    Args:
        awc_in_per_in: Available water capacity (inches/inch of soil)
        current_moisture_pct: Current soil moisture as percentage

    Returns:
        Dictionary with stress category and recommendations
    """
    # Convert AWC to percentage (assume typical relationship)
    # AWC ranges typically 0.05-0.30 in/in, corresponds roughly to 5-30% available
    awc_pct = awc_in_per_in * 100

    if current_moisture_pct >= awc_pct * 0.75:
        stress_level = "optimal"
        description = "Water-filled pores available for plant uptake"
    elif current_moisture_pct >= awc_pct * 0.5:
        stress_level = "moderate"
        description = "Adequate water but plant available water declining"
    elif current_moisture_pct >= awc_pct * 0.25:
        stress_level = "mild_stress"
        description = "Plants begin to experience water stress"
    else:
        stress_level = "severe_stress"
        description = "Severe water stress; irrigation recommended if applicable"

    return {
        "stress_level": stress_level,
        "description": description,
        "awc_pct": awc_pct,
        "current_moisture_pct": current_moisture_pct,
        "plant_available_pct": current_moisture_pct,
    }


def should_use_awdb_for_water_analysis() -> bool:
    """
    Determine if AWDB data is necessary for water availability analysis.

    Returns:
        True - AWDB is essential for validating and monitoring water availability
    """
    return True


def get_recommended_awdb_depths_for_soil(clay_pct: float) -> List[int]:
    """
    Get recommended SCAN soil depths based on soil texture.

    Args:
        clay_pct: Percentage clay content

    Returns:
        List of recommended soil depths in inches (negative values)
    """
    # Sandy soils: focus on shallow measurements (fast drainage)
    # Clay soils: focus on deeper measurements (slower drainage, water retention)
    if clay_pct < 20:
        return [-2, -4]  # Sandy
    elif clay_pct < 35:
        return [-2, -4, -8]  # Loamy
    else:
        return [-2, -4, -8, -16, -20]  # Clay
