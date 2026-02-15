from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def persistent_cache_dir() -> Path:
    """
    Returns a persistent cache directory for integration tests.
    This avoids re-downloading large datasets (Lidar/BlueTopo) across test runs,
    significantly reducing test suite runtime.
    """
    # Use a subdirectory in the standard user cache to separate test data from production data
    path = Path("~/.cache/topobathysim/tests").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
