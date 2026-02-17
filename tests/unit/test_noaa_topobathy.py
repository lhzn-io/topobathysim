from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from topobathysim.providers.noaa_topobathy import NoaaTopobathyProvider


@pytest.fixture
def mock_provider(tmp_path: Path) -> NoaaTopobathyProvider:
    # Setup directories
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Initialize provider with temp cache
    provider = NoaaTopobathyProvider(cache_dir=str(cache_dir))

    # Mock project list to avoid network calls
    provider._projects = {"9999": "Test Project", "10274": "LIS Project"}
    provider._projects_metadata_urls = {}

    return provider


def test_find_project_by_box_with_index(mock_provider: NoaaTopobathyProvider) -> None:
    """
    Test that find_project_by_box uses the spatial index correctly.
    """
    # Create a dummy spatial index
    index_path = mock_provider.metadata_dir / "noaa_project_extents.geojson"

    # Create two projects:
    # 1. Old Project (2010) covering unit square
    # 2. New Project (2020) covering same area
    # We expect the New Project to win based on date.

    data = [
        {
            "project_id": "8888",
            "project_name": "Old Project",
            "geometry": box(0, 0, 1, 1),
            "start_date": "2010-01-01",
            "end_date": "2010-12-31",
            "vertical_datum": "Ellipsoid",
        },
        {
            "project_id": "9999",
            "project_name": "New Project",
            "geometry": box(0, 0, 1, 1),
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "vertical_datum": "NAVD88",
        },
    ]

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # Save to mock location
    mock_provider.metadata_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(index_path, driver="GeoJSON")

    # Query inside the box
    # Using 0.5, 0.5 which is inside 0,0,1,1
    pid = mock_provider.find_project_by_box(0.4, 0.4, 0.6, 0.6)

    # Should pick "9999" (Newer + Better Datum)
    assert pid == "9999", f"Expected 9999, got {pid}"


def test_find_project_by_box_failure(mock_provider: NoaaTopobathyProvider) -> None:
    """
    Test failure behavior when index is missing/empty.
    """
    # Ensure no index file exists
    index_path = mock_provider.metadata_dir / "noaa_project_extents.geojson"
    if index_path.exists():
        index_path.unlink()

    # Mock the build script to do nothing (prevent actual build attempting to run)
    with (
        patch("topobathysim.scripts.build_noaa_index.main"),
        pytest.raises(RuntimeError, match="NOAA Spatial Index is unavailable"),
    ):
        mock_provider.find_project_by_box(-73.0, 41.0, -72.0, 41.1)
