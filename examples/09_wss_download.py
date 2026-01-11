"""
Example of downloading SSURGO data from Web Soil Survey.

This example shows how to:
1. Download specific survey areas as ZIP files
2. Download all survey areas for a state
3. Download STATSGO data
4. Handle the downloaded files
"""

import asyncio
import tempfile
from pathlib import Path

from soildb import download_wss


async def main():
    """Demonstrate WSS download functionality."""

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("=== Downloading Specific Survey Areas ===")

        try:
            # Download specific survey areas
            print("Downloading IA109 and IA113 survey areas...")
            paths = await download_wss(
                areasymbols=["IA109", "IA113"],
                dest_dir=temp_path / "ssurgo_specific",
                extract=True,
                remove_zip=False  # Keep ZIP files for inspection
            )

            print(f"Downloaded {len(paths)} survey areas:")
            for path in paths:
                print(f"  - {path}")
                if path.is_dir():
                    # List contents
                    tabular = path / "tabular"
                    spatial = path / "spatial"
                    if tabular.exists():
                        txt_files = list(tabular.glob("*.txt"))
                        print(f"    Tabular files: {len(txt_files)}")
                    if spatial.exists():
                        shp_files = list(spatial.glob("*.shp"))
                        print(f"    Spatial files: {len(shp_files)}")

        except Exception as e:
            print(f"Error downloading specific areas: {e}")

        print("\n=== Downloading All Hawaii Survey Areas ===")

        try:
            # Download all Iowa survey areas using WHERE clause
            print("Downloading all Hawaii (HI*) survey areas...")
            paths = await download_wss(
                where_clause="areasymbol LIKE 'HI%'",
                dest_dir=temp_path / "ssurgo_hawaii",
                extract=True,
                remove_zip=True  # Remove ZIP files after extraction
            )

            print(f"Downloaded {len(paths)} Hawaii survey areas")

        except Exception as e:
            print(f"Error downloading Iowa areas: {e}")

        print("\n=== Downloading STATSGO Data ===")

        try:
            # Download STATSGO data for Iowa
            print("Downloading STATSGO data for Iowa...")
            paths = await download_wss(
                areasymbols=["IA"],
                db="STATSGO",
                dest_dir=temp_path / "statsgo_iowa",
                extract=True
            )

            print(f"Downloaded {len(paths)} STATSGO areas:")
            for path in paths:
                print(f"  - {path}")

        except Exception as e:
            print(f"Error downloading STATSGO data: {e}")

        print("\n=== Sync Usage Example ===")

        try:
            # Demonstrate synchronous usage
            print("Downloading using sync API...")
            paths = download_wss.sync(
                areasymbols=["IA109"],
                dest_dir=temp_path / "sync_example",
                extract=True
            )

            print(f"Sync download completed: {len(paths)} areas")

        except Exception as e:
            print(f"Error with sync download: {e}")

if __name__ == "__main__":
    asyncio.run(main())
