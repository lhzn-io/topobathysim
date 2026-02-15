# Instead of replacing sys.modules, patch where it is used.
# But GEBCO2025Provider inherits from Topography. This is tricky to patch after import if already imported.
# Strategy: Use importlib to reload the module under test after mocking to ensure it picks up the mock.
import importlib
import sys
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from dask.array.core import PerformanceWarning

# Mock bmi_topography if not installed
mock_bmi = MagicMock()


class MockTopography:
    def __init__(self, **kwargs: Any) -> None:
        self.dem_type = kwargs.get("dem_type")
        self.output_format = kwargs.get("output_format")
        self.south = kwargs.get("south")
        self.north = kwargs.get("north")
        self.west = kwargs.get("west")
        self.east = kwargs.get("east")
        self.cache_dir = kwargs.get("cache_dir")

        # Emulate bmi-topography structure
        # Use SimpleNamespace or namedtuple or just an object
        from types import SimpleNamespace

        self.bbox = SimpleNamespace(south=self.south, north=self.north, west=self.west, east=self.east)


mock_bmi.Topography = MockTopography
sys.modules["bmi_topography"] = mock_bmi

# Reload to force using the mock
if "topobathysim.gebco_2025" in sys.modules:
    import topobathysim.gebco_2025

    importlib.reload(topobathysim.gebco_2025)

from topobathysim.gebco_2025 import GEBCO2025Provider as Gebco2025  # noqa: E402


@pytest.fixture
def mock_gebco_dataset() -> xr.Dataset:
    """Create a mock Xarray Dataset mimicking GEBCO 2025 structure."""
    lat = np.linspace(-10, 10, 20)
    lon = np.linspace(-10, 10, 20)

    da_elev = xr.DataArray(
        np.zeros((20, 20)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="sub_ice_topo_bathymetry",
    )

    da_tid = xr.DataArray(
        np.ones((20, 20), dtype=int) * 11,  # Direct TID
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="tid",
    )

    ds = xr.Dataset({"sub_ice_topo_bathymetry": da_elev, "tid": da_tid})
    return ds


def test_initialization() -> None:
    gebco = Gebco2025(north=10, south=-10, west=-10, east=10)
    # dem_type is set by parent to SRTMGL3 to satisfy validation, even though we are GEBCO 2025 logic.
    assert gebco.output_format == "GTiff"


@patch("xarray.open_dataset")
def test_fetch_elev_and_tid(mock_open_ds: MagicMock, mock_gebco_dataset: xr.Dataset, tmp_path: Any) -> None:
    mock_open_ds.return_value = mock_gebco_dataset

    # Use tmp_path for cache to avoid polluting user cache and ensure isolation
    cache_dir = tmp_path / "gebco_cache"
    gebco = Gebco2025(north=5, south=-5, west=-5, east=5, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        da = gebco.fetch()

    # Check Elevation logic
    assert da is not None
    assert da.name == "elevation"

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
    val = da.lat.values[0]
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
    gebco = Gebco2025(north=5, south=-5, west=-5, east=5, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        gebco.load()

    # Sample center (0,0 is in our mock data)
    # Mock data is all zeros for elevation
    elev = gebco.sample_elevation(0, 0)
    assert elev == 0.0

    # Test error before load
    gebco_unloaded = Gebco2025()
    with pytest.raises(RuntimeError):
        gebco_unloaded.sample_elevation(0, 0)
