import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr
from rasterio.transform import Affine

from topobathysim.manager import BathyManager


class TestCUDEMFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = BathyManager(use_blue_topo=True, use_cudem=True, use_lidar=False, use_land=False)

    def create_mock_da(
        self,
        val: float,
        shape: tuple[int, int] = (10, 10),
        bounds: tuple[float, float, float, float] = (-70.1, 40.0, -70.0, 40.1),
    ) -> xr.DataArray:
        west, south, east, north = bounds
        height, width = shape
        lon_res = (east - west) / width
        lat_res = (north - south) / height

        # Pixel Centers
        xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=width)
        ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=height)

        data = np.full(shape, val, dtype=float)
        coords = {"y": ys, "x": xs}
        da = xr.DataArray(data, coords=coords, dims=("y", "x"))
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.write_nodata(np.nan, inplace=True)

        # Explicit Transform is REQUIRED for reproject_match on generated data
        # Top Left (West, North) = (west, north)
        # Scale (lon_res, -lat_res)
        transform = Affine.translation(west, north) * Affine.scale(lon_res, -lat_res)
        da.rio.write_transform(transform, inplace=True)

        return da

    @patch("topobathysim.manager.GEBCO2025Provider")
    def test_bluetopo_overrides_cudem(self, mock_gebco: MagicMock) -> None:
        """
        Verify that Tier 2 BlueTopo overrides Tier 3 CUDEM in the fusion stack.
        """
        # Mock GEBCO (Tier 4)
        mock_gebco_da = self.create_mock_da(100.0)  # Deep background
        mock_gebco_instance = MagicMock()
        mock_gebco_instance.fetch.return_value = mock_gebco_da
        mock_gebco.return_value = mock_gebco_instance

        # Mock BlueTopo (Tier 2) - Value 50
        mock_bt_da = self.create_mock_da(50.0)
        self.manager.blue_topo = MagicMock()
        self.manager.blue_topo.resolve_tiles_in_bbox.return_value = ["tile1"]
        self.manager.blue_topo.load_tile_as_da.return_value = mock_bt_da

        # Mock CUDEM (Tier 3) - Value 10 (Intertidal/Shallow)
        mock_cudem_da = self.create_mock_da(10.0)

        # Create a partial CUDEM (NaNs in one half) to test blending
        # CUDEM is valid on Left, NaN on Right
        vals = mock_cudem_da.values.copy()
        vals[:, 5:] = np.nan
        mock_cudem_da.values = vals

        self.manager.cudem = MagicMock()
        self.manager.cudem.get_grid.return_value = mock_cudem_da

        # Mock BAG to None
        self.manager.bag = None

        # Mock Topobathy to None
        self.manager.topobathy = None

        # Execute with correct coords: South, North, West, East, Shape
        # Intended bounds: S=40.0, N=40.1, W=-70.1, E=-70.0
        result = cast(xr.DataArray, self.manager.get_grid(40.0, 40.1, -70.1, -70.0, target_shape=(10, 10)))

        self.assertIsNotNone(result)

        # Verify Left Half (CUDEM Valid, BlueTopo Valid) -> Should be ~50.0 (BlueTopo overrides CUDEM)
        # Verify Right Half (CUDEM NaN, BlueTopo Valid) -> Should be ~50.0 (BlueTopo)

        left_val = result.isel(x=2, y=5).item()
        right_val = result.isel(x=8, y=5).item()

        print(f"Left Val (Overlap): {left_val}")
        print(f"Right Val (BlueTopo Only): {right_val}")

        # Strict Override Check
        # BlueTopo (50) should suppress CUDEM (10)
        self.assertGreater(left_val, 40.0, "BlueTopo (Tier 2) should override CUDEM (Tier 3)")

        # Fallback Check
        self.assertGreater(right_val, 40.0, "BlueTopo value should be preserved where CUDEM is missing")

        # Check Source Metadata
        self.assertIn("BlueTopo", result.attrs.get("source", ""))


if __name__ == "__main__":
    unittest.main()
