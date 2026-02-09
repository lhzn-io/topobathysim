import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine


def test_reproject():
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
    # Identical creation
    data = np.full(target_shape, 50.0)
    source = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))
    source.rio.write_crs("EPSG:4326", inplace=True)
    source.rio.write_nodata(np.nan, inplace=True)

    # Write Transform explicitly on Source (as I added to the test)
    transform = Affine(lon_res, 0.0, west, 0.0, -lat_res, north)
    source.rio.write_transform(transform, inplace=True)

    # Test merge_arrays
    from rioxarray.merge import merge_arrays

    merged_source = merge_arrays([source])
    print("Merged Source Transform:", merged_source.rio.transform())

    # Reproject Match
    reproj = merged_source.rio.reproject_match(target, resampling=Resampling.bilinear)

    print("Reproject Result Sample:")
    print(reproj.values[:3, :3])

    if np.all(np.isnan(reproj.values)):
        print("FAIL: All NaNs")
    else:
        print("SUCCESS")


if __name__ == "__main__":
    test_reproject()
