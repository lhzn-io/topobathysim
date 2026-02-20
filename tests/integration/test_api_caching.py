from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from topobathyserve.main import app, get_policy_path

from topobathysim.policy.schema import CompositionStep, FusionPolicy, VariableStrategy


@pytest.fixture
def clean_cache(tmp_path: Path) -> Path:
    """Creates a temporary cache directory for the duration of the test."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def test_policy_path(clean_cache: Path) -> Path:
    """Creates a basic policy pointing its cache to the clean tmp_path."""
    policy = FusionPolicy(
        name="Test Caching Policy",
        version="1.0.0",
        variables=[
            VariableStrategy(
                name="elevation",
                steps=[
                    CompositionStep(provider="UsgsLidarProvider"),
                    CompositionStep(provider="Usgs3DepProvider"),
                ],
            )
        ],
    )

    policy_path = clean_cache / "policy.yaml"
    with open(policy_path, "w") as f:
        # Patch the environment cache to point here during test explicitly
        import os

        os.environ["TOPOBATHYSIM_CACHE_DIR"] = str(clean_cache)
        f.write(policy.model_dump_json())
    return policy_path


@pytest.fixture
def client(test_policy_path: Path) -> TestClient:
    """Returns a TestClient with the policy path dependency overridden."""
    app.dependency_overrides[get_policy_path] = lambda: test_policy_path
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_tile_request_caches_data(client: TestClient, clean_cache: Path) -> None:
    """
    Verifies that requesting a tile triggers data downloads and caching
    in the expected directories.
    """
    # NYC (Battery Park area) - Known to have Lidar and 3DEP coverage
    # Zoom 14 specific tile coordinates
    z, x, y = 14, 4825, 6156

    url = f"/tiles/{z}/{x}/{y}.png"

    # 1. Initial State: Cache should be empty of data
    lidar_dir = clean_cache / "usgs_lidar"
    land_dir = clean_cache / "usgs_3dep"

    assert not any(lidar_dir.rglob("*.laz")), "Lidar cache should be empty initially"
    assert not any(land_dir.rglob("*.tif")), "Land cache should be empty initially"

    # 2. Fire Request
    # We expect this to be slow on first run as it downloads data
    print(f"Requesting {url}...")
    response = client.get(url)

    # 3. Validation
    assert response.status_code == 200, f"Request failed: {response.text}"
    assert response.headers["content-type"] == "image/png"

    # 4. Side Effect Check: Did we cache files?
    # Note: Depending on the fusion logic and data availability,
    # specific providers might be skipped or hit.
    # For NYC Battery Park:
    # - USGS Lidar is usually available.
    # - USGS 3DEP should be fallback or complementary.

    lidar_files = list(lidar_dir.rglob("*.laz")) + list(lidar_dir.rglob("*.copc.laz"))
    land_files = list(land_dir.rglob("*.tif"))

    print(f"Cached Lidar Files: {len(lidar_files)}")
    print(f"Cached Land Files: {len(land_files)}")

    # We expect *at least* one of them to have data given the location
    has_data = len(lidar_files) > 0 or len(land_files) > 0
    assert has_data, "No data files were cached! Provider fetch may have failed."

    # 5. Check Output Cache (The generated PNG)
    output_cache = clean_cache / "tiles" / "visual" / "default"
    assert any(output_cache.glob("*.png")), "Output PNG was not cached."
