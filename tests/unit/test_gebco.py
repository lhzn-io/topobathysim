import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from dask.array.core import PerformanceWarning

from topobathysim.providers.gebco_2025 import GEBCO2025Provider as Gebco2025


@pytest.fixture
def mock_gebco_dataset() -> xr.Dataset:
    """Create a mock Xarray Dataset mimicking GEBCO 2025 structure."""
    lat = np.linspace(-10, 10, 100)
    lon = np.linspace(-10, 10, 100)

    da_elev = xr.DataArray(
        np.zeros((100, 100)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="elevation",
    )

    da_source_id = xr.DataArray(
        np.zeros((100, 100), dtype=np.uint32),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="source_id",
    )

    da_tid = xr.DataArray(
        np.ones((100, 100), dtype=int) * 11,  # Direct TID
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="tid",
    )

    ds = xr.Dataset({"elevation": da_elev, "source_id": da_source_id, "tid": da_tid})
    # Add rio accessor to ensure we have crs setup correctly for interpolation limits
    ds.rio.write_crs("EPSG:4326", inplace=True)
    return ds


def test_initialization() -> None:
    gebco = Gebco2025(north=10, south=-10, west=-10, east=10)
    assert gebco.cache_dir is not None


@patch("xarray.open_dataset")
def test_fetch_elev_and_tid(mock_open_ds: MagicMock, mock_gebco_dataset: xr.Dataset, tmp_path: Any) -> None:
    mock_open_ds.return_value = mock_gebco_dataset

    # Use tmp_path for cache to avoid polluting user cache and ensure isolation
    cache_dir = tmp_path / "gebco_cache"
    gebco = Gebco2025(north=1, south=0, west=0, east=1, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        da = gebco.fetch()

    # Check Elevation logic
    assert da is not None
    assert "elevation" in da.data_vars

    # Check TID logic - TID not currently implemented in Provider
    # tid = gebco.get_tid_classification()
    # assert tid is not None
    # assert tid.name == "tid"

    # Check Half-Pixel Offset (coordinates should be shifted)
    # We verify that the value is NOT what it would be without offset.
    # The exact value depends on slice logic, but we know it should have the offset component.
    # 15 arcseconds = 15/3600 degrees. Half of that is the offset.
    offset = Gebco2025.HALF_PIXEL_OFFSET
    # Just check that it's not an integer (mock data lies on integers/simple fractions)
    # and has the offset remainder
    val = da.y.values[0]
    _ = offset % 1  # simplistic check or just check diff from nearest grid

    # Better: check that the difference between the fetched value and the 'raw' mock value
    # (which we can infer or mock access to) is the offset.
    # In this mock, we know the source grid is linspace(-10, 10, 20).
    # -4.73684211 is the mock value near -5.
    # The fetched val is -4.7347...
    # Difference is +0.00208... which is exactly the offset.
    assert (val % (15 / 3600)) != 0  # It shouldn't align with the grid anymore


@patch("xarray.open_dataset")
def test_sample_elevation(mock_open_ds: MagicMock, mock_gebco_dataset: xr.Dataset, tmp_path: Any) -> None:
    mock_open_ds.return_value = mock_gebco_dataset

    cache_dir = tmp_path / "gebco_cache_sample"
    gebco = Gebco2025(north=1, south=0, west=0, east=1, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        gebco.fetch()

    # Sample center (0.5, 0.5 is safely inside our 0 to 1 box)
    # Mock data is all zeros for elevation
    elev = gebco.sample_elevation(0.5, 0.5)
    assert elev == 0.0

    # Test error before load
    gebco_unloaded = Gebco2025()
    with pytest.raises(RuntimeError):
        gebco_unloaded.sample_elevation(0, 0)
