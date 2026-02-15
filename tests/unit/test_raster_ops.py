import numpy as np
import pytest
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rioxarray.merge import merge_arrays


def test_reproject_manual_transform() -> None:
    """Test that manual affine transforms are respected during merge and reproject."""
    west = -70.1
    south = 40.0
    east = -70.0
    north = 40.1
    target_shape = (10, 10)

    height, width = target_shape
    lon_res = (east - west) / width
    lat_res = (north - south) / height

    # Manager Logic for Base Canvas
    xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=width)
    ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=height)

    target = xr.DataArray(np.nan, coords={"y": ys, "x": xs}, dims=("y", "x"))
    target.rio.write_crs("EPSG:4326", inplace=True)
    # Target has NO explicitly written transform, just coords

    # Source (Mock) Logic
    data = np.full(target_shape, 50.0)
    source = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))
    source.rio.write_crs("EPSG:4326", inplace=True)
    source.rio.write_nodata(np.nan, inplace=True)

    # Write Transform explicitly on Source
    transform = Affine(lon_res, 0.0, west, 0.0, -lat_res, north)
    source.rio.write_transform(transform, inplace=True)

    merged_source = merge_arrays([source])

    # Verify transform persisted after merge (approximate)
    m_trans = merged_source.rio.transform()
    assert m_trans.a == pytest.approx(transform.a)
    assert m_trans.e == pytest.approx(transform.e)
    assert m_trans.c == pytest.approx(transform.c)
    assert m_trans.f == pytest.approx(transform.f)

    # Reproject Match
    # Suppress Dask PerformanceWarning for tiny array operations where chunk overhead is negligible
    import warnings

    from dask.array.core import PerformanceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        reproj = merged_source.rio.reproject_match(target, resampling=Resampling.bilinear)

    # Validate results are not NaN (meaning alignment worked)
    assert not np.all(np.isnan(reproj.values))

    # Since grids align perfectly, should be close to 50.0
    assert np.allclose(reproj.values, 50.0, equal_nan=True)
