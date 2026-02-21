import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_sys_modules() -> Generator[None, None, None]:
    """Ensure we don't leak mocks into other tests."""
    original_modules = sys.modules.copy()
    yield
    # Restore
    sys.modules.clear()
    sys.modules.update(original_modules)


from topobathysim.manager import BathyManager  # noqa: E402
from topobathysim.providers.noaa_bluetopo import NoaaBlueTopoProvider  # noqa: E402
from topobathysim.vdatum import VDatumResolver  # noqa: E402


def test_vdatum_caching() -> None:
    with patch("requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": 1.5}
        mock_session.get.return_value = mock_response

        resolver = VDatumResolver()
        offset1 = resolver.get_navd88_to_lmsl_offset(42.0, -70.0)
        offset2 = resolver.get_navd88_to_lmsl_offset(42.0, -70.0)

        assert offset1 == 1.5
        assert offset2 == 1.5
        # Verify cache: request called only once
        # Since cache is on the static method, the second call returns cached result immediately
        # and doesn't even enter the function body to create a session?
        # Actually lru_cache is on the method.
        # But wait, static method with lru_cache? Yes.
        # If it's cached, method body isn't executed.
        assert mock_session.get.call_count == 1


def test_bluetopo_coverage() -> None:
    provider = NoaaBlueTopoProvider()

    with patch.object(provider, "resolve_tile_id") as mock_resolve:
        # Case: Covered
        mock_resolve.return_value = "BlueTopo_Tile_Example"
        assert provider.is_covered(42.0, -70.0) is True

        # Case: Not Covered
        mock_resolve.return_value = None
        assert provider.is_covered(0.0, 0.0) is False


@patch("topobathysim.providers.noaa_bluetopo.fsspec.filesystem")
@patch("topobathysim.providers.noaa_bluetopo.rioxarray.open_rasterio")
def test_bluetopo_caching_and_fetch(mock_open_rasterio: MagicMock, mock_filesystem: MagicMock) -> None:
    provider = NoaaBlueTopoProvider()

    # Mock resolve_tile_id to return a valid ID
    with patch.object(provider, "resolve_tile_id", return_value="BlueTopo_Tile_Test"):
        # Setup FSSpec Mock
        mock_fs = MagicMock()
        # Mock glob so it finds a file to stream
        mock_fs.glob.return_value = [
            "noaa-ocs-nationalbathymetry-pds/BlueTopo/BlueTopo_Tile_Test/BlueTopo_Tile_Test_2025.tiff"
        ]
        mock_filesystem.return_value = mock_fs

        with patch("pathlib.Path.exists", return_value=False):
            # Mock rioxarray dataset
            mock_da = MagicMock()
            mock_da.rio.nodata = -9999
            mock_da.rio.crs = "EPSG:4326"
            mock_val = MagicMock()
            mock_val.values.item.return_value = -15.0
            mock_val.dims = []  # scalar
            mock_val.size = 1
            mock_da.sel.return_value = mock_val

            # Since rioxarray.open_rasterio is used as a context manager:
            mock_op = MagicMock()
            mock_op.__enter__.return_value = mock_da
            mock_open_rasterio.return_value = mock_op

            # mock VDatum
            provider.vdatum = MagicMock()
            provider.vdatum.get_navd88_to_lmsl_offset.return_value = 1.0

            depth = provider.fetch_elevation(42.0, -70.0)

            # Verify stream called
            assert mock_open_rasterio.called
            assert depth == -16.0


@patch("topobathysim.manager.GEBCO2025Provider")
def test_manager_smart_selection(mock_gebco_cls: MagicMock) -> None:
    # Setup GEBCO Mock
    mock_gebco_instance = MagicMock()
    mock_gebco_instance.sample_elevation.return_value = -100.0
    mock_gebco_cls.return_value = mock_gebco_instance

    manager = BathyManager()

    # Setup BlueTopo Mock
    manager.blue_topo = MagicMock()
    manager.blue_topo.is_covered.return_value = True
    manager.blue_topo.fetch_elevation.return_value = -5.0

    depth = manager.get_elevation(42.0, -70.0)
    assert depth == -5.0
    assert manager.blue_topo.is_covered.called

    # Case 2: BlueTopo Not Covered -> GEBCO Fallback
    manager.blue_topo.is_covered.return_value = False

    depth = manager.get_elevation(0.0, 0.0)
    assert depth == -100.0


def test_source_info() -> None:
    manager = BathyManager(use_cudem=False)  # Disable CUDEM to ensure BlueTopo priority in this test
    manager.blue_topo = MagicMock()
    manager.blue_topo.is_covered.return_value = True
    info = manager.get_source_info(42.0, -70.0)
    assert "BlueTopo" in info["source"]

    manager.blue_topo.is_covered.return_value = False
    info = manager.get_source_info(0.0, 0.0)
    assert "GEBCO" in info["source"]
