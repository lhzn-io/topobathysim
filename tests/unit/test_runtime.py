"""
Unit tests for the Core Runtime.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

from topobathysim.providers.base import Provider
from topobathysim.providers.registry import registry
from topobathysim.runtime import run


class MockProvider(Provider):
    """A mock provider that returns synthetic data."""

    def __init__(self, value: float = 10.0):
        self.value = value

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Simulate fetching a layer by creating a synthetic grid.

        Args:
            bbox: Bounding box to cover.
            resolution: Grid resolution.
            crs: Coordinate Reference System.

        Returns:
            xr.DataArray: Synthetic elevation data.
        """
        # Create a tiny 10x10 grid covering the bbox
        min_lon, min_lat, max_lon, max_lat = bbox
        width, height = 10, 10

        # Simple transform
        res_x = (max_lon - min_lon) / width
        res_y = (max_lat - min_lat) / height
        transform = Affine.translation(min_lon, max_lat) * Affine.scale(res_x, -res_y)

        data = np.full((height, width), self.value, dtype=np.float32)

        coords = {
            "y": np.linspace(max_lat - res_y / 2, min_lat + res_y / 2, height),
            "x": np.linspace(min_lon + res_x / 2, max_lon - res_x / 2, width),
        }

        da = xr.DataArray(data, coords=coords, dims=("y", "x"))
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(transform, inplace=True)

        return da

    def get_metadata(self) -> dict[str, Any]:
        """Return mock metadata."""
        return {"name": "MockProvider"}


@pytest.fixture
def mock_policy_file(tmp_path: Any) -> str:
    """Create a temporary policy file."""
    policy_content = {
        "name": "TestPolicy",
        "crs": "EPSG:4326",
        "variables": [
            {
                "name": "elevation",
                "background": -999.0,
                "steps": [
                    {"provider": "mock_base", "operation": "overlay"},
                    {"provider": "mock_overlay", "operation": "overlay"},
                ],
            }
        ],
    }

    policy_path = tmp_path / "test_policy.yaml"
    with open(policy_path, "w") as f:
        yaml.dump(policy_content, f)

    return str(policy_path)


def test_runtime_flow(mock_policy_file: str) -> None:
    """Test the full runtime flow with mock providers."""

    # 1. Register Mock Providers
    # Register mock providers for testing
    mock_base_cls = type("MockBase", (MockProvider,), {})
    mock_overlay_cls = type("MockOverlay", (MockProvider,), {})

    registry._providers["mock_base"] = mock_base_cls
    registry._providers["mock_overlay"] = mock_overlay_cls

    # Define provider subclasses with deterministic return values
    class BaseProvider(MockProvider):
        def fetch_layer(self, *args: Any, **kwargs: Any) -> xr.DataArray:
            da = super().fetch_layer(*args, **kwargs)
            da[:] = 10.0  # Base value
            return da

    class OverlayProvider(MockProvider):
        def fetch_layer(self, *args: Any, **kwargs: Any) -> xr.DataArray:
            da = super().fetch_layer(*args, **kwargs)
            da[:] = 20.0  # Overlay value
            # Make it smaller or masked?
            # Let's make half of it NaN to test blending/masking
            da.values[:, :5] = np.nan
            return da

    registry.register("mock_base", BaseProvider)
    registry.register("mock_overlay", OverlayProvider)

    # 2. Run Runtime
    bbox = (-74.0, 40.0, -73.0, 41.0)  # 1x1 degree
    # Use coarse resolution (10km ~ 0.1 deg) to avoid memory issues
    ds = run(mock_policy_file, bbox, resolution=10000.0)

    # 3. Assertions
    assert isinstance(ds, xr.Dataset)
    assert "elevation" in ds
    assert "source_elevation" in ds

    print(ds)

    # Check values
    # Base is 10. Overlay is 20 (on right half).
    # Expected: Left half = 10 (Base), Right half = 20 (Overlay)
    # Source: Left half = ID(mock_base), Right half = ID(mock_overlay)

    # Get IDs
    legend_str = ds.attrs.get("policy_legend", "{}")
    legend = eval(legend_str)  # Returns dictionary string rep
    id_base = next(k for k, v in legend.items() if v == "mock_base")
    id_overlay = next(k for k, v in legend.items() if v == "mock_overlay")

    # Sampling
    # Left side (index 0)
    val_left = ds["elevation"].isel(x=0).mean().item()
    src_left = ds["source_elevation"].isel(x=0)[0].item()

    # Right side (index -1)
    val_right = ds["elevation"].isel(x=-1).mean().item()

    assert val_left == 10.0
    assert val_right == 20.0

    # Source verification
    # Note: Mock provider returns precise grid so no interpolation artifacts expected

    assert src_left == id_base
    assert ds["source_elevation"].isel(x=-1).max().item() == id_overlay
