from pathlib import Path

import pytest

from topobathysim.bluetopo import BlueTopoProvider

# Coordinates for Execution Rocks, Long Island Sound (near Rye, NY)
# This should fall into one of the user's identified tiles (e.g. BH4XH5FN or similar)
EXECUTION_ROCKS_LAT = 40.8835
EXECUTION_ROCKS_LON = -73.7369


@pytest.mark.integration
def test_bluetopo_real_integration(tmp_path: Path) -> None:
    """ """
    import os
    import warnings

    from dotenv import load_dotenv

    # Load env vars
    load_dotenv()
    # Map OPEN_TOPOGRAPHY_API_KEY (from .env) to OPENTOPOGRAPHY_API_KEY (expected by lib)
    if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
        os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

    # Suppress spurious warnings from third-party libs
    # warnings.filterwarnings("ignore", category=UserWarning, module="bmi_topography")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydap")

    # Use a temp directory for cache to verify download occurs
    cache_dir = tmp_path / "bathy_cache"
    provider = BlueTopoProvider(cache_dir=str(cache_dir))

    print(f"\n[Integration] Resolving tile for Lat: {EXECUTION_ROCKS_LAT}, Lon: {EXECUTION_ROCKS_LON}...")

    # 1. Test Tile Resolution (Requires network for GPKG download)
    tile_id = provider.resolve_tile_id(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)
    print(f"[Integration] Resolved Tile ID: {tile_id}")

    # Assert we found *something*
    assert tile_id is not None, "Failed to resolve any tile for Execution Rocks location."

    # The user expected tiles like BH4XJ5FP, BH4XH5FN etc.
    # We can check if the ID looks like that pattern (BlueTopo scheme often uses these IDs)
    # Note: The 'tile_id' from resolve_tile_id might be the filename or ID column.
    # provider.resolve_tile_id returns the first available of ['file_name', 'tile_name', ...].

    # 2. Test Fetch Elevation (Requires network for COG download)
    print("[Integration] Fetching elevation (downloading COG)...")
    depth = provider.fetch_elevation(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)
    print(f"[Integration] Elevation: {depth} meters")

    assert depth is not None, "Failed to fetch elevation value."
    assert isinstance(depth, float)
    # Depth in Long Island Sound is roughly 20-30m deep at max, Execution rocks is shallow.
    # Value should be reasonable (e.g. -50 to +10).
    assert -100.0 < depth < 100.0

    # 3. Verify Cache
    # We expect a .tiff file in the cache directory
    downloaded_files = list(cache_dir.glob("*.tiff"))
    print(f"[Integration] Cached files: {[f.name for f in downloaded_files]}")
    assert len(downloaded_files) > 0, "No TIFF files found in cache after fetch."

    # 4. Compare with GEBCO
    from topobathysim.gebco import GEBCO2025

    print("\n[Integration] Fetching GEBCO 2025 value for comparison...")
    gebco = GEBCO2025()
    gebco.load()
    gebco_depth = gebco.sample_elevation(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)
    print(f"[Integration] GEBCO Elevation: {gebco_depth} meters")

    delta = abs(depth - gebco_depth)
    print(f"[Integration] Difference (BlueTopo - GEBCO): {delta:.4f} meters")

    # We don't necessarily assert they are close, as BlueTopo is high res and GEBCO is low res.
    # But it's good for the user to see.
