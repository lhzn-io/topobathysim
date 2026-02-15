from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

# ensure can import app
from topobathyserve.main import app

# client = TestClient(app)
# We use context manager in tests to ensure startup events run


@pytest.fixture
def mock_managers() -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    with (
        patch("topobathyserve.main.BathyManager") as mock_bm,
        patch("topobathysim.usgs_lidar.UsgsLidarProvider") as mock_lp,
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


@patch("pathlib.Path.exists", return_value=False)
def test_xyz_tile_endpoint(
    mock_exists: MagicMock, mock_managers: tuple[MagicMock, MagicMock, MagicMock]
) -> None:
    bm, _, _ = mock_managers

    # Mock return DAs
    # Create simple DataArrays roughly around New York (40N, 74W)
    xs = np.linspace(-74.5, -73.5, 10)
    ys = np.linspace(40.5, 41.5, 10)

    da_bathy = xr.DataArray(np.zeros((10, 10)) - 10, coords={"y": ys, "x": xs})
    da_bathy.rio.write_crs("EPSG:4326", inplace=True)

    bm.get_grid.return_value = da_bathy

    # Request a valid tile (Tile 10/301/388 covers part of this area)
    url = "/tiles/10/301/388.tif"

    with TestClient(app) as client:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

        # Verify manager called
        bm.get_grid.assert_called_once()
