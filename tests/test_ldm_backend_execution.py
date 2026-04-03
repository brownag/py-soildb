"""Integration tests for LDM backend execution."""

import pytest
import pytest_asyncio

from soildb import SDAClient
from soildb.ldm.backends import SDABackend
from soildb.ldm.exceptions import LDMSDAError
from soildb.response import SDAResponse


@pytest_asyncio.fixture
async def sda_backend():
    client = SDAClient(timeout=30.0)
    backend = SDABackend(client=client)
    yield backend
    await client.close()


class TestSDABackendExecution:
    def _skip_if_maintenance(self, e):
        if "maintenance" in str(e).lower():
            pytest.skip("SDA under maintenance")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_simple_query(self, sda_backend):
        try:
            sql = "SELECT TOP 1 areasymbol, areaname FROM sacatalog"
            response = await sda_backend.execute_query(sql)
            assert isinstance(response, SDAResponse)
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_empty_result(self, sda_backend):
        try:
            sql = "SELECT TOP 0 areasymbol FROM sacatalog"
            response = await sda_backend.execute_query(sql)
            assert response.is_empty()
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_invalid_column(self, sda_backend):
        try:
            with pytest.raises(LDMSDAError):
                await sda_backend.execute_query("SELECT invalid_col FROM sacatalog")
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_invalid_table(self, sda_backend):
        try:
            with pytest.raises(LDMSDAError):
                await sda_backend.execute_query("SELECT * FROM nonexistent_xyz")
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_multi_row(self, sda_backend):
        try:
            sql = "SELECT TOP 5 areasymbol, areaname FROM sacatalog"
            response = await sda_backend.execute_query(sql)
            assert len(response) <= 5
            if len(response) > 0:
                data = response.to_dict()
                assert all(isinstance(row, dict) for row in data)
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_to_dict(self, sda_backend):
        try:
            sql = "SELECT TOP 1 areasymbol, areaname FROM sacatalog"
            response = await sda_backend.execute_query(sql)
            data = response.to_dict()
            assert isinstance(data, list)
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_columns_present(self, sda_backend):
        try:
            sql = "SELECT TOP 1 areasymbol, areaname FROM sacatalog"
            response = await sda_backend.execute_query(sql)
            assert response.columns is not None or len(response) == 0
        except LDMSDAError as e:
            self._skip_if_maintenance(e)
            raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_sqlite_backend_skip(self):
        pytest.skip("KSSL SQLite database not configured")
