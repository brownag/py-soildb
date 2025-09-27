"""
Spatial analysis examples using soildb.

Demonstrates working with geographic soil data including spatial queries,
area calculations, and mapping.
"""

import asyncio

from soildb import spatial_query

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point

    SPATIAL_LIBS = True
except ImportError:
    print("GeoPandas/Shapely not available. Install with:")
    print("pip install soildb[spatial]")
    SPATIAL_LIBS = False


async def point_buffer_analysis():
    """Analyze soils within a buffer around a point."""
    print("=== Point Buffer Analysis ===")

    if not SPATIAL_LIBS:
        print("Spatial libraries required for this example")
        return

    # Center point (Ames, Iowa)
    center_lon, center_lat = -93.6319, 42.0308
    buffer_degrees = 0.01  # ~1km buffer

    # Create buffer polygon
    center_point = Point(center_lon, center_lat)
    buffer_poly = center_point.buffer(buffer_degrees)

    # Query map unit polygons within buffer
    response = await spatial_query(
        geometry=buffer_poly.wkt,
        table="mupolygon",
        spatial_relation="intersects",
        return_type="spatial",
    )

    if response.data:
        gdf = response.to_geodataframe()

        print(f"Found {len(gdf)} map units within buffer")
        print(
            f"Total area: {gdf.geometry.to_crs('EPSG:5070').area.sum():.6f} square meters"
        )

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(ax=ax, column="musym", alpha=0.7, legend=True, cmap="tab20")

        # Add buffer boundary
        buffer_gdf = gpd.GeoDataFrame([1], geometry=[buffer_poly], crs="EPSG:4326")
        buffer_gdf.plot(ax=ax, color="red", alpha=0.3, edgecolor="red", linewidth=2)

        # Add center point
        center_gdf = gpd.GeoDataFrame([1], geometry=[center_point], crs="EPSG:4326")
        center_gdf.plot(ax=ax, color="red", markersize=100, marker="x")

        ax.set_title(f"Map Units Within {buffer_degrees:.3f}° of Ames, Iowa")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.show()

        return gdf
    else:
        print("No map units found in buffer")
        return None


async def survey_area_boundaries():
    """Visualize survey area boundaries."""
    print("\n=== Survey Area Boundaries ===")

    if not SPATIAL_LIBS:
        print("Spatial libraries required for this example")
        return

    # Iowa counties
    areasymbols = ["IA015", "IA109", "IA113", "IA169"]  # Boone, Polk, Story, Story

    # Get survey area polygons - need to query each area separately
    all_gdfs = []
    for _areasymbol in areasymbols:
        # Create a simple bbox for each area (this is simplified)
        response = await spatial_query(
            geometry={
                "xmin": -94,
                "ymin": 41,
                "xmax": -93,
                "ymax": 42,
            },  # Rough Iowa bbox
            table="sapolygon",
            return_type="spatial",
        )
        if response.data:
            gdf = response.to_geodataframe()
            # Filter to specific area symbol
            area_gdf = gdf[gdf["areasymbol"].isin(areasymbols)]
            if not area_gdf.empty:
                all_gdfs.append(area_gdf)

    if all_gdfs:
        gdf = gpd.pd.concat(all_gdfs, ignore_index=True)

    if response.data:
        gdf = response.to_geodataframe()

        print(f"Found {len(gdf)} survey area polygons")

        # Plot survey areas
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            ax=ax,
            column="areasymbol",
            alpha=0.7,
            legend=True,
            edgecolor="black",
            linewidth=1,
        )

        # Add labels
        for _idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                row["areasymbol"],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_title("Iowa County Survey Areas")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.show()

        return gdf
    else:
        print("No survey areas found")
        return None


async def soil_comparison_by_region():
    """Compare soil properties across different regions."""
    print("\n=== Soil Comparison by Region ===")

    # Define study areas (small bounding boxes)
    regions = {
        "Central Iowa": {"xmin": -93.7, "ymin": 42.0, "xmax": -93.6, "ymax": 42.1},
        "Eastern Iowa": {"xmin": -91.6, "ymin": 41.5, "xmax": -91.5, "ymax": 41.6},
        "Western Iowa": {"xmin": -95.9, "ymin": 41.5, "xmax": -95.8, "ymax": 41.6},
    }

    region_data = {}

    for region_name, bbox in regions.items():
        print(f"Querying {region_name}...")

        # Get map unit polygons
        response = await spatial_query(
            geometry=bbox,
            table="mupolygon",
            spatial_relation="intersects",
            return_type="tabular",
        )

        if response.data:
            df = response.to_pandas()
            region_data[region_name] = df
            print(f"Found {len(df)} map units")
        else:
            print(f"No data found for {region_name}")

    # Compare regions
    if region_data:
        print("\nRegion Comparison:")
        for region, df in region_data.items():
            unique_symbols = df["musym"].nunique()
            print(f"{region}: {len(df)} polygons, {unique_symbols} unique map units")

    return region_data


async def watershed_analysis():
    """Analyze soils within a watershed boundary."""
    print("\n=== Watershed Analysis (Simplified) ===")

    # Simulate a simple watershed polygon (in reality, you'd load from shapefile)
    # This is a rough approximation of a small watershed
    watershed_wkt = """POLYGON((-93.65 42.02, -93.60 42.02, -93.58 42.05,
                                -93.62 42.08, -93.67 42.06, -93.65 42.02))"""

    # Query map units within watershed
    response = await spatial_query(
        geometry=watershed_wkt,
        table="mupolygon",
        spatial_relation="intersects",
        return_type="spatial",
    )

    if response.data:
        df = response.to_pandas()
        print(f"Found {len(df)} map units in watershed")

        # Analyze map unit distribution
        mu_counts = df["musym"].value_counts()
        print("\nTop 5 map units by frequency:")
        for musym, count in mu_counts.head().items():
            print(f"  {musym}: {count} polygons")

        if SPATIAL_LIBS:
            gdf = response.to_geodataframe()

            # Simple visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(ax=ax, column="musym", alpha=0.7, legend=True)
            ax.set_title("Map Units in Simulated Watershed")
            plt.tight_layout()
            plt.show()

        return df
    else:
        print("No map units found in watershed")
        return None


async def main():
    """Run all spatial analysis examples."""
    print("Spatial Analysis Examples")
    print("=" * 50)

    # Run examples
    await point_buffer_analysis()
    await survey_area_boundaries()
    await soil_comparison_by_region()
    await watershed_analysis()

    print("\n" + "=" * 50)
    print("Spatial analysis examples completed!")

    if not SPATIAL_LIBS:
        print("\nTo enable full spatial functionality, install:")
        print("pip install soildb[spatial]")


if __name__ == "__main__":
    asyncio.run(main())
