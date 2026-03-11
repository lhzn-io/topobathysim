import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import xarray as xr
from fastapi.testclient import TestClient

# ensure can import app
try:
    from topobathyserve.main import app
except ImportError:
    # If package structure is weird during test
    import sys

    sys.path.append(str(Path.cwd() / "service"))
    from topobathyserve.main import app


def create_mock_ds(lat: float, lon: float, elev: float = -10.0, sid: int = 1) -> xr.Dataset:
    # Project lat/lon to Web Mercator (EPSG:3857) to match expected runtime output
    earth_r = 6378137.0
    x_m = earth_r * math.radians(lon)
    y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    # Create 1x1 dataset
    da_elev = xr.DataArray([[elev]], coords={"y": [y_m], "x": [x_m]}, dims=("y", "x"), name="elevation")
    # Ensure rioxarray accessor works if needed by endpoint
    da_elev.rio.write_crs("EPSG:3857", inplace=True)

    da_source = xr.DataArray(
        [[sid]], coords={"y": [y_m], "x": [x_m]}, dims=("y", "x"), name="source_elevation"
    )
    da_source.rio.write_crs("EPSG:3857", inplace=True)

    ds = xr.Dataset({"elevation": da_elev, "source_elevation": da_source})
    ds.attrs["provenance_dict"] = {sid: {"provider": "MockProvider"}}
    ds.rio.write_crs("EPSG:3857", inplace=True)
    return ds


@patch("topobathyserve.main.run")
def test_elevation_endpoint(mock_run: MagicMock) -> None:
    # Setup Mock
    lat, lon = 40.0, -73.0

    # Endpoint requires policy path to exist
    with patch("pathlib.Path.exists", return_value=True):
        mock_run.return_value = create_mock_ds(lat, lon, elev=-15.5)

        with TestClient(app) as client:
            response = client.get(f"/elevation?lat={lat}&lon={lon}")
            # If 500, print error
            if response.status_code != 200:
                print(response.content)
            assert response.status_code == 200
            data = response.json()
            assert "elevation" in data
            assert data["elevation"] == -15.5


@patch("topobathyserve.main.run")
def test_source_info_endpoint(mock_run: MagicMock) -> None:
    # Setup Mock
    lat, lon = 40.0, -73.0

    with patch("pathlib.Path.exists", return_value=True):
        mock_run.return_value = create_mock_ds(lat, lon, sid=42)

        with TestClient(app) as client:
            response = client.get(f"/source_info?lat={lat}&lon={lon}")
            assert response.status_code == 200
            data = response.json()
            assert "source" in data
            assert "42" in data["id"]  # Check ID is present


@patch("topobathyserve.main.run")
def test_fuse_endpoint(mock_run: MagicMock) -> None:
    # Test /fuse endpoint
    # Requires bbox
    with patch("pathlib.Path.exists", return_value=True):
        mock_run.return_value = create_mock_ds(40.0, -73.0, elev=100.0)

        bbox = "0,0,1,1"  # Dummy bbox
        # Provide west/south/east/north via query params or bbox string

        with TestClient(app) as client:
            # Request GeoTIFF
            response = client.get(f"/fuse?bbox={bbox}&format=geotiff")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/tiff"

            # Verify run called
            mock_run.assert_called()
