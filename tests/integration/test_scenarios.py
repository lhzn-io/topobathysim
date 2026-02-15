import logging
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from topobathysim.manager import BathyManager

# Create a custom logger for this test module to ensure we see our own debugs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(autouse=True, scope="module")
def silence_chatty_libraries() -> None:
    """
    Configures logging to silence noisy third-party libraries while
    keeping 'topobathysim' and local tests verbose.
    """
    # List of libraries to silence (set to WARNING or higher)
    chatty_loggers = [
        "rasterio",
        "botocore",
        "s3fs",
        "fsspec",
        "urllib3",
        "fiona",
        "shapely",
        "matplotlib",
        "PIL",
        "asyncio",
        "azure",
        "geopandas",
        "pyproj",
    ]

    for log_name in chatty_loggers:
        logging.getLogger(log_name).setLevel(logging.WARNING)

    # Ensure topobathysim is loud and clear
    logging.getLogger("topobathysim").setLevel(logging.DEBUG)


def xyz_to_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert XYZ tile coordinates to (south, north, west, east) bounding box."""
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)

    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)
    return west, south, east, north


@pytest.mark.integration
@pytest.mark.parametrize(
    "z, x, y, expected_source_substr",
    [
        (13, 2418, 3075, "BAG"),  # Scenario from verify_tile.py
        (
            15,
            9672,
            12300,
            ["BAG", "LIDAR", "TOPO", "USGS"],
        ),  # Scenario from verify_user_tile.py (matches any of these)
    ],
)
def test_real_world_tile_scenarios(
    z: int, x: int, y: int, expected_source_substr: str | list[str], persistent_cache_dir: Path
) -> None:
    """
    Verify specific real-world tiles known to contain specific data types.
    This replaces the ad-hoc scripts in tests/scripts/.
    """
    west, south, east, north = xyz_to_bounds(z, x, y)
    logger.info(f"Testing Tile Z={z} X={x} Y={y} -> BBox: {west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}")

    manager = BathyManager(
        use_blue_topo=True, use_cudem=True, use_lidar=True, cache_dir=str(persistent_cache_dir)
    )

    # Use a smaller shape for speed, but large enough to catch features
    result = manager.get_grid(south, north, west, east, target_shape=(256, 256))
    da = cast(xr.DataArray, result)

    assert da is not None, "Grid generation failed (returned None)"
    assert da.shape == (256, 256)

    sources = da.attrs.get("source", "Unknown").upper()
    logger.info(f"Sources found: {sources}")

    # Verification Logic
    if isinstance(expected_source_substr, list):
        # Match ANY of the expected substrings
        found = any(sub in sources for sub in expected_source_substr)
        assert found, f"Expected one of {expected_source_substr} in sources, got: {sources}"
    else:
        assert (
            expected_source_substr in sources
        ), f"Expected {expected_source_substr} in sources, got: {sources}"

    # Data content validation
    valid_data = da.values[~np.isnan(da.values)]
    assert len(valid_data) > 0, "Grid contains only NaNs"

    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    logger.info(f"Elevation Range: {min_val:.2f} to {max_val:.2f}")

    # Heuristic checks based on known tile locations
    if z == 15 and x == 9672:
        # Known detailed tile with land and water
        assert max_val > 0, "Should have land (elevation > 0)"
        assert min_val < -2.0, "Should have deep water (elevation < -2.0)"
