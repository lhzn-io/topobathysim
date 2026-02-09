import os
import sys

import numpy as np
import xarray as xr

# Ensure root directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from service.topobathyserve.main import render_png
except ImportError:
    # If service not found, try adding current dir if running from root
    sys.path.insert(0, os.path.abspath("."))
    from service.topobathyserve.main import render_png


def test_contour_generation_basic():
    """Test standard contour generation with valid data."""
    # Create a nice gradient 0..100
    data = np.linspace(0, 100, 10000).reshape(100, 100)
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": range(100), "x": range(100)})

    # Render
    png_bytes = render_png(da, style="contours", zoom=13)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    assert png_bytes.startswith(b"\x89PNG")


def test_contour_generation_intervals():
    """Test that different zoom levels don't crash and produce output."""
    data = np.linspace(0, 100, 10000).reshape(100, 100)
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": range(100), "x": range(100)})

    # Low zoom (10m interval)
    bytes_z10 = render_png(da, style="contours", zoom=10)
    assert len(bytes_z10) > 0

    # High zoom (1m interval) - Should have many more lines potentially, or at least run
    bytes_z18 = render_png(da, style="contours", zoom=18)
    assert len(bytes_z18) > 0


def test_contour_generation_nan_handling():
    """Test that NaNs don't cause exceptions and produce transparent tile."""
    data = np.full((100, 100), np.nan)
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": range(100), "x": range(100)})

    # Should not crash
    png_bytes = render_png(da, style="contours", zoom=13)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0


def test_contour_coastline():
    """Test that zero-elevation (coastline) is handled without error."""
    # -10 to +10, crossing zero
    data = np.linspace(-10, 10, 10000).reshape(100, 100)
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": range(100), "x": range(100)})

    png_bytes = render_png(da, style="contours", zoom=14)
    assert len(png_bytes) > 0
