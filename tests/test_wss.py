"""
Tests for Web Soil Survey download functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soildb import WSSClient, WSSDownloadError, download_wss
from soildb.wss import build_ssurgo_url, build_statsgo_url


class TestWSSClient:
    """Test WSSClient functionality."""

    @pytest.mark.asyncio
    async def test_download_zip_success(self, tmp_path):
        """Test successful ZIP download."""
        url = "https://example.com/test.zip"
        dest_path = tmp_path / "test.zip"

        mock_content = b"fake zip content"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock the response
            mock_response = AsyncMock()
            mock_response.headers = {"content-length": str(len(mock_content))}
            mock_response.raise_for_status = MagicMock()

            # Create an async iterator for aiter_bytes
            async def mock_aiter_bytes():
                yield mock_content

            mock_response.aiter_bytes = mock_aiter_bytes

            # Create a proper async context manager mock
            class MockAsyncContextManager:
                def __init__(self, response):
                    self.response = response

                async def __aenter__(self):
                    return self.response

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))

            async with WSSClient() as client:
                result = await client.download_zip(url, dest_path)

                assert result == dest_path
                assert dest_path.exists()
                assert dest_path.read_bytes() == mock_content

    @pytest.mark.asyncio
    async def test_download_zip_404(self, tmp_path):
        """Test ZIP download with 404 error."""
        url = "https://example.com/missing.zip"
        dest_path = tmp_path / "missing.zip"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 404 response
            from httpx import HTTPStatusError
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_error = HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)

            mock_client.stream = MagicMock(side_effect=mock_error)

            async with WSSClient() as client:
                with pytest.raises(WSSDownloadError, match="Dataset not found"):
                    await client.download_zip(url, dest_path)

    def test_extract_zip(self, tmp_path):
        """Test ZIP extraction."""
        import zipfile

        # Create a test ZIP file
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "extracted"

        # Create some test content
        test_files = {
            "file1.txt": "content1",
            "file2.csv": "content2",
            "spatial.shp": "shapefile content"
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            for filename, content in test_files.items():
                zf.writestr(filename, content)

        # Extract
        result_dir = WSSClient.extract_zip(zip_path, extract_dir)

        assert result_dir == extract_dir
        assert extract_dir.exists()

        # Check files were extracted
        for filename in test_files:
            assert (extract_dir / filename).exists()

    def test_organize_extracted_files(self, tmp_path):
        """Test organization of extracted files."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Create test files
        test_files = [
            "table1.txt",
            "table2.csv",
            "spatial.shp",
            "spatial.shx",
            "spatial.dbf",
            "readme.md"
        ]

        for filename in test_files:
            (extract_dir / filename).write_text("test content")

        # Organize files
        WSSClient.organize_extracted_files(extract_dir)

        # Check organization
        tabular_dir = extract_dir / "tabular"
        spatial_dir = extract_dir / "spatial"

        assert tabular_dir.exists()
        assert spatial_dir.exists()

        # Check tabular files moved
        assert (tabular_dir / "table1.txt").exists()
        assert (tabular_dir / "table2.csv").exists()

        # Check spatial files moved
        assert (spatial_dir / "spatial.shp").exists()
        assert (spatial_dir / "spatial.shx").exists()
        assert (spatial_dir / "spatial.dbf").exists()

        # Check non-tabular/spatial files remain
        assert (extract_dir / "readme.md").exists()


class TestURLBuilding:
    """Test URL building functions."""

    def test_build_ssurgo_url(self):
        """Test SSURGO URL building."""
        url = build_ssurgo_url("IA109", "2023-10-01")
        expected = "https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache/SSA/wss_SSA_IA109_[2023-10-01].zip"
        assert url == expected

    def test_build_statsgo_url(self):
        """Test STATSGO URL building."""
        url = build_statsgo_url("IA", "2023-10-01")
        expected = "https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache/STATSGO2/wss_gsmsoil_IA_[2023-10-01].zip"
        assert url == expected


class TestDownloadWSS:
    """Test download_wss function."""

    @pytest.mark.asyncio
    async def test_download_wss_with_areasymbols(self, tmp_path):
        """Test downloading with specific areasymbols."""
        dest_dir = tmp_path / "downloads"

        # Mock SDA response
        mock_response = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iterrows.return_value = [
            ("IA109", MagicMock(areasymbol="IA109", saverest="2023-10-01")),
            ("IA113", MagicMock(areasymbol="IA113", saverest="2023-10-01"))
        ]
        mock_response.to_pandas.return_value = mock_df

        # Mock SDA client
        mock_sda_client = AsyncMock()
        mock_sda_client.execute.return_value = mock_response

        # Mock WSS download
        with patch("soildb.wss.WSSClient") as mock_wss_class:
            mock_wss_client = AsyncMock()
            mock_wss_class.return_value.__aenter__.return_value = mock_wss_client
            mock_wss_class.return_value.__aexit__.return_value = None

            # Mock successful downloads
            zip_path1 = dest_dir / "IA109_20231001.zip"
            zip_path2 = dest_dir / "IA113_20231001.zip"
            mock_wss_client.download_zip.side_effect = [zip_path1, zip_path2]

            # Mock extraction
            with patch.object(WSSClient, "extract_zip") as mock_extract:

                mock_extract.side_effect = [dest_dir / "IA109", dest_dir / "IA113"]

                result = await download_wss(
                    areasymbols=["IA109", "IA113"],
                    dest_dir=dest_dir,
                    client=mock_sda_client
                )

                assert len(result) == 2
                assert mock_sda_client.execute.called
                assert mock_wss_client.download_zip.call_count == 2

    @pytest.mark.asyncio
    async def test_download_wss_with_where_clause(self, tmp_path):
        """Test downloading with WHERE clause."""
        dest_dir = tmp_path / "downloads"

        # Mock SDA response
        mock_response = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iterrows.return_value = [
            ("IA109", MagicMock(areasymbol="IA109", saverest="2023-10-01"))
        ]
        mock_response.to_pandas.return_value = mock_df

        mock_sda_client = AsyncMock()
        mock_sda_client.execute.return_value = mock_response

        with patch("soildb.wss.WSSClient") as mock_wss_class:
            mock_wss_client = AsyncMock()
            mock_wss_class.return_value.__aenter__.return_value = mock_wss_client
            mock_wss_class.return_value.__aexit__.return_value = None

            zip_path = dest_dir / "IA109_20231001.zip"
            mock_wss_client.download_zip.return_value = zip_path

            with patch.object(WSSClient, "extract_zip") as mock_extract:
                mock_extract.return_value = dest_dir / "IA109"

                result = await download_wss(
                    where_clause="areasymbol LIKE 'IA%'",
                    dest_dir=dest_dir,
                    client=mock_sda_client
                )

                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_download_statsgo(self, tmp_path):
        """Test downloading STATSGO data."""
        dest_dir = tmp_path / "downloads"

        # Mock SDA response
        mock_response = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iterrows.return_value = [
            ("IA", MagicMock(areasymbol="IA", saverest="2023-10-01"))
        ]
        mock_response.to_pandas.return_value = mock_df

        mock_sda_client = AsyncMock()
        mock_sda_client.execute.return_value = mock_response

        with patch("soildb.wss.WSSClient") as mock_wss_class:
            mock_wss_client = AsyncMock()
            mock_wss_class.return_value.__aenter__.return_value = mock_wss_client
            mock_wss_class.return_value.__aexit__.return_value = None

            zip_path = dest_dir / "IA_20231001.zip"
            mock_wss_client.download_zip.return_value = zip_path

            with patch.object(WSSClient, "extract_zip") as mock_extract:
                mock_extract.return_value = dest_dir / "IA"

                result = await download_wss(
                    areasymbols=["IA"],
                    db="STATSGO",
                    dest_dir=dest_dir,
                    client=mock_sda_client
                )

                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_areas_found(self):
        """Test when no survey areas are found."""
        mock_response = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = True
        mock_response.to_pandas.return_value = mock_df

        mock_sda_client = AsyncMock()
        mock_sda_client.execute.return_value = mock_response

        result = await download_wss(
            areasymbols=["NONEXISTENT"],
            client=mock_sda_client
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_db_parameter(self):
        """Test invalid db parameter."""
        with pytest.raises(ValueError, match="db must be 'SSURGO' or 'STATSGO'"):
            await download_wss(areasymbols=["IA109"], db="INVALID")

    @pytest.mark.asyncio
    async def test_missing_parameters(self):
        """Test missing required parameters."""
        with pytest.raises(ValueError, match="Either areasymbols or where_clause must be provided"):
            await download_wss()
