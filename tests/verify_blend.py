"""
Verification script for the metric_feather operator.
"""

import numpy as np
import xarray as xr

from topobathysim.operators.blend import metric_feather


def test_metric_feather_basic() -> None:
    """Test metric_feather with a synthetic base and overlay."""
    # Create Base DataArray (0-10 deg, 0.01 deg res)
    x = np.arange(0, 10, 0.01)
    y = np.arange(0, 10, 0.01)

    data_base = np.zeros((len(y), len(x)))
    base = xr.DataArray(data_base, coords={"y": y, "x": x}, dims=("y", "x"))
    base.rio.write_crs("EPSG:4326", inplace=True)

    # Create Overlay DataArray (subset 4-6 deg, offset value 100)
    x_sub = np.arange(4, 6, 0.01)
    y_sub = np.arange(4, 6, 0.01)

    data_ov = np.ones((len(y_sub), len(x_sub))) * 100
    overlay = xr.DataArray(data_ov, coords={"y": y_sub, "x": x_sub}, dims=("y", "x"))
    overlay.rio.write_crs("EPSG:4326", inplace=True)

    # Blend with 10km feather
    distance_m = 10000.0
    result = metric_feather(base, overlay, distance_m)

    # Verification Checks
    # 1. Center of overlay should be fully blended (100)
    center_val = result.sel(x=5, y=5, method="nearest").item()
    assert np.isclose(center_val, 100.0), f"Center value should be 100, got {center_val}"

    # 2. Far outside should be base value (0)
    outside_val = result.sel(x=2, y=2, method="nearest").item()
    assert np.isclose(outside_val, 0.0), f"Outside value should be 0, got {outside_val}"

    # 3. Transition zone (approx 4.02 deg should be within blend range)
    # The edge is at 4.0. 10km is roughly 0.1 degrees.
    edge_val = result.sel(x=4.02, y=5, method="nearest").item()

    assert 0 < edge_val < 100, f"Edge value {edge_val} not in blend range (0-100)"

    print("Metric feather verification passed.")


if __name__ == "__main__":
    test_metric_feather_basic()
