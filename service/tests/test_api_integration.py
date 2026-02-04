from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# ensure can import app
from app.main import app
from fastapi.testclient import TestClient

# client = TestClient(app)
# We use context manager in tests to ensure startup events run


@pytest.fixture
def mock_managers() -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    with (
        patch("app.main.BathyManager") as mock_bm,
        patch("topobathysim.lidar.LidarProvider") as mock_lp,
        patch("topobathysim.fusion.FusionEngine") as mock_fe,
    ):
        # Setup BathyManager instance
        bm_instance = MagicMock()
        mock_bm.return_value = bm_instance

        # Setup LidarProvider instance
        lp_instance = MagicMock()
        mock_lp.return_value = lp_instance

        yield bm_instance, lp_instance, mock_fe


def test_elevation_endpoint(mock_managers: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    bm, _, _ = mock_managers
    bm.get_elevation.return_value = -15.5

    with TestClient(app) as client:
        response = client.get("/elevation?lat=40.0&lon=-73.0")
        assert response.status_code == 200
        assert response.json()["elevation"] == -15.5


def test_source_info_endpoint(mock_managers: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    bm, _, _ = mock_managers
    bm.get_source_info.return_value = {"source": "BlueTopo"}

    with TestClient(app) as client:
        response = client.get("/source_info?lat=40.0&lon=-73.0")
        assert response.status_code == 200
        assert response.json()["source"] == "BlueTopo"


def test_fused_tile_ept(mock_managers: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    bm, lp, _ = mock_managers

    # Mock return DAs
    # Create simple DataArrays
    da_bathy = xr.DataArray(np.zeros((10, 10)) - 10, coords={"y": range(10), "x": range(10)})
    da_bathy.rio.write_crs("EPSG:4326", inplace=True)

    da_lidar = xr.DataArray(np.ones((10, 10)) * 5, coords={"y": range(10), "x": range(10)})
    da_lidar.rio.write_crs("EPSG:4326", inplace=True)

    bm.get_grid.return_value = da_bathy
    lp.fetch_lidar_from_ept.return_value = da_lidar

    url = "/fused_tile?north=41&south=40&west=-74&east=-73&ept_url=http://mock/ept.json"

    with TestClient(app) as client:
        response = client.get(url)
        assert response.status_code == 200
        # Check that fetch_lidar_from_ept was called with ept_url
        lp.fetch_lidar_from_ept.assert_called_once()
        args, kwargs = lp.fetch_lidar_from_ept.call_args
        assert args[0] == "http://mock/ept.json"
        assert kwargs["target_crs"] == "EPSG:4326"
        assert response.headers["content-type"] == "image/tiff"
