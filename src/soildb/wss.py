"""
Web Soil Survey (WSS) download functionality.

Provides functions to download complete SSURGO and STATSGO datasets
from the USDA-NRCS Web Soil Survey portal as ZIP files.
"""

import asyncio
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import httpx

from .client import SDAClient
from .exceptions import SoilDBError
from .utils import add_sync_version

logger = logging.getLogger(__name__)


class WSSDownloadError(SoilDBError):
    """Exception raised for Web Soil Survey download errors."""
    pass


class WSSClient:
    """Client for downloading data from Web Soil Survey."""

    BASE_URL = "https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache"

    def __init__(self, timeout: float = 300.0):
        """
        Initialize WSS client.

        Args:
            timeout: Request timeout in seconds (default: 5 minutes for large downloads)
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "WSSClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is available."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def download_zip(
        self,
        url: str,
        dest_path: Union[str, Path],
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> Path:
        """
        Download a ZIP file from WSS.

        Args:
            url: Download URL
            dest_path: Destination file path
            progress_callback: Optional callback for download progress

        Returns:
            Path to downloaded file
        """
        await self._ensure_client()
        assert self._client is not None  # Should be set by _ensure_client

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(dest_path, "wb") as f:
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            progress_callback(progress, downloaded, total_size)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise WSSDownloadError(f"Dataset not found at URL: {url}") from e
            raise WSSDownloadError(f"HTTP error downloading {url}: {e}") from e
        except Exception as e:
            raise WSSDownloadError(f"Error downloading {url}: {e}") from e

        return dest_path

    @staticmethod
    def extract_zip(
        zip_path: Union[str, Path],
        extract_dir: Union[str, Path],
        remove_zip: bool = False
    ) -> Path:
        """
        Extract a ZIP file to a directory, handling nested directory structures.

        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to
            remove_zip: Whether to remove ZIP file after extraction

        Returns:
            Path to extraction directory
        """
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)

        if not zip_path.exists():
            raise WSSDownloadError(f"ZIP file not found: {zip_path}")

        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as e:
            raise WSSDownloadError(f"Invalid ZIP file {zip_path}: {e}") from e

        if remove_zip:
            zip_path.unlink()

        return extract_dir

    @staticmethod
    def organize_extracted_files(extract_dir: Union[str, Path]) -> None:
        """
        Organize extracted SSURGO files into tabular and spatial subdirectories.

        Args:
            extract_dir: Directory containing extracted files
        """
        extract_dir = Path(extract_dir)

        # Create subdirectories
        tabular_dir = extract_dir / "tabular"
        spatial_dir = extract_dir / "spatial"

        tabular_dir.mkdir(exist_ok=True)
        spatial_dir.mkdir(exist_ok=True)

        # Move files based on extension, but exclude metadata files
        for file_path in extract_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.name.lower()
                # Exclude known metadata files from tabular organization
                is_metadata = (
                    filename.startswith(('readme', 'metadata', 'soil_metadata')) or
                    'metadata' in filename
                )

                if file_path.suffix.lower() in [".txt", ".csv"] and not is_metadata:
                    file_path.rename(tabular_dir / file_path.name)
                elif file_path.suffix.lower() in [".shp", ".shx", ".dbf", ".prj", ".sbn", ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".ixs", ".mxs", ".atx", ".shp.xml"]:
                    file_path.rename(spatial_dir / file_path.name)
            # Skip existing tabular/ and spatial/ subdirectories - they're already correctly organized


def build_ssurgo_url(areasymbol: str, saverest: Union[str, Any]) -> str:
    """
    Build Web Soil Survey download URL for SSURGO data.

    Args:
        areasymbol: Survey area symbol (e.g., 'IA109')
        saverest: Save date (string or pandas Timestamp)

    Returns:
        Download URL string
    """
    # Convert to string if it's a pandas Timestamp
    if hasattr(saverest, 'strftime'):
        date_str = saverest.strftime('%Y-%m-%d')
    else:
        # Assume it's already a string in YYYY-MM-DD format
        date_str = str(saverest)
    return f"{WSSClient.BASE_URL}/SSA/wss_SSA_{areasymbol}_[{date_str}].zip"


def build_statsgo_url(areasymbol: str, saverest: Union[str, Any]) -> str:
    """
    Build Web Soil Survey download URL for STATSGO data.

    Args:
        areasymbol: Survey area symbol (e.g., 'IA' for state, 'US' for national)
        saverest: Save date (string or pandas Timestamp)

    Returns:
        Download URL string
    """
    # Convert to string if it's a pandas Timestamp
    if hasattr(saverest, 'strftime'):
        date_str = saverest.strftime('%Y-%m-%d')
    else:
        # Assume it's already a string in YYYY-MM-DD format
        date_str = str(saverest)
    return f"{WSSClient.BASE_URL}/STATSGO2/wss_gsmsoil_{areasymbol}_[{date_str}].zip"


@add_sync_version
async def download_wss(
    areasymbols: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    dest_dir: Union[str, Path] = "./ssurgo_data",
    extract: bool = True,
    remove_zip: bool = False,
    include_templates: bool = False,
    db: str = "SSURGO",
    client: Optional[SDAClient] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    max_concurrent: int = 3,
) -> List[Path]:
    """
    Download SSURGO or STATSGO ZIP files for specified survey areas from Web Soil Survey.

    Args:
        areasymbols: List of area symbols to download. If None, uses where_clause.
        where_clause: SQL WHERE clause to filter sacatalog table (e.g., "areasymbol LIKE 'IA%'")
        dest_dir: Directory to save downloaded files
        extract: Whether to extract ZIP files after download
        remove_zip: Whether to remove ZIP files after extraction
        include_templates: Whether to include MS Access template databases (SSURGO only)
        db: Database type - "SSURGO" or "STATSGO"
        client: Optional SDA client. If not provided, creates temporary client.
        progress_callback: Optional callback function for download progress
        max_concurrent: Maximum number of concurrent downloads

    Returns:
        List of paths to downloaded/extracted directories

    Examples:
        # Download specific survey areas
        paths = await download_wss(areasymbols=["IA109", "IA113"])

        # Download all Iowa survey areas
        paths = await download_wss(where_clause="areasymbol LIKE 'IA%'")

        # Download STATSGO for Iowa
        paths = await download_wss(
            areasymbols=["IA"],
            db="STATSGO",
            dest_dir="./statsgo_data"
        )
    """
    if db not in ["SSURGO", "STATSGO"]:
        raise ValueError("db must be 'SSURGO' or 'STATSGO'")

    if areasymbols is None and where_clause is None:
        raise ValueError("Either areasymbols or where_clause must be provided")

    # Create SDA client if not provided
    if client is None:
        client = SDAClient()

    # Get survey area metadata

    columns = ["areasymbol", "areaname", "saversion", "saverest"]

    if areasymbols:
        # Build WHERE clause from areasymbols
        symbols_str = ", ".join(f"'{s}'" for s in areasymbols)
        where = f"areasymbol IN ({symbols_str})"
    else:
        where = where_clause  # type: ignore

    # Query sacatalog with custom WHERE
    from .query import Query
    query = Query().select(*columns).from_("sacatalog").where(where)
    response = await client.execute(query)
    metadata_df = response.to_pandas()

    if metadata_df.empty:
        logger.warning("No survey areas found matching criteria")
        return []

    logger.info(f"Found {len(metadata_df)} survey areas to download")

    # Build download URLs
    download_tasks = []
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for _, row in metadata_df.iterrows():
        areasymbol = row["areasymbol"]
        saverest = row["saverest"]

        if db == "SSURGO":
            url = build_ssurgo_url(areasymbol, saverest)
        else:  # STATSGO
            url = build_statsgo_url(areasymbol, saverest)

        # Create destination path
        # Convert saverest to string for filename
        if hasattr(saverest, 'strftime'):
            saverest_str = saverest.strftime('%Y%m%d')
        else:
            saverest_str = str(saverest).replace("-", "")
        zip_filename = f"{areasymbol}_{saverest_str}.zip"
        zip_path = dest_dir / zip_filename

        download_tasks.append((url, zip_path, areasymbol))

    # Download files concurrently
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(url: str, zip_path: Path, areasymbol: str) -> Path:
        async with semaphore:
            logger.info(f"Downloading {areasymbol}...")
            async with WSSClient() as wss_client:
                return await wss_client.download_zip(url, zip_path, progress_callback)

    # Execute downloads
    downloaded_paths = await asyncio.gather(
        *[download_with_semaphore(url, zip_path, areasymbol)
          for url, zip_path, areasymbol in download_tasks],
        return_exceptions=True
    )

    # Handle exceptions and collect successful downloads
    successful_paths: List[Path] = []
    for i, result in enumerate(downloaded_paths):
        if isinstance(result, Exception):
            areasymbol = download_tasks[i][2]
            logger.error(f"Failed to download {areasymbol}: {result}")
        else:
            successful_paths.append(result)  # type: ignore

    if not successful_paths:
        raise WSSDownloadError("No files were successfully downloaded")

    # Extract files if requested
    if extract:
        extracted_paths = []
        for zip_path in successful_paths:
            areasymbol = zip_path.stem.split('_')[0]  # Extract areasymbol from filename
            extract_dir = dest_dir / areasymbol

            try:
                # Extract ZIP preserving its internal structure
                WSSClient.extract_zip(zip_path, dest_dir, remove_zip)
                WSSClient.organize_extracted_files(extract_dir)
                extracted_paths.append(extract_dir)
                logger.info(f"Extracted {areasymbol} to {extract_dir}")
            except Exception as e:
                logger.error(f"Failed to extract {zip_path}: {e}")
                extracted_paths.append(zip_path)  # Return ZIP path if extraction failed

        return extracted_paths

    return successful_paths
