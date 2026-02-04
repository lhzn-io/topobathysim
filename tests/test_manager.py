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


from topobathysim.bluetopo import BlueTopoProvider  # noqa: E402
from topobathysim.manager import BathyManager  # noqa: E402
from topobathysim.vdatum import VDatumResolver  # noqa: E402


def test_vdatum_caching() -> None:
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": 1.5}
        mock_get.return_value = mock_response

        resolver = VDatumResolver()
        offset1 = resolver.get_navd88_to_lmsl_offset(42.0, -70.0)
        offset2 = resolver.get_navd88_to_lmsl_offset(42.0, -70.0)

        assert offset1 == 1.5
        assert offset2 == 1.5
        # Verify cache: request called only once
        assert mock_get.call_count == 1


def test_bluetopo_coverage() -> None:
    provider = BlueTopoProvider()

    with patch.object(provider, "resolve_tile_id") as mock_resolve:
        # Case: Covered
        mock_resolve.return_value = "BlueTopo_Tile_Example"
        assert provider.is_covered(42.0, -70.0) is True

        # Case: Not Covered
        mock_resolve.return_value = None
        assert provider.is_covered(0.0, 0.0) is False


@patch("topobathysim.bluetopo.fsspec.filesystem")
@patch("topobathysim.bluetopo.rioxarray.open_rasterio")
def test_bluetopo_caching_and_fetch(mock_open_rasterio: MagicMock, mock_filesystem: MagicMock) -> None:
    provider = BlueTopoProvider()

    # Mock resolve_tile_id to return a valid ID
    with patch.object(provider, "resolve_tile_id", return_value="BlueTopo_Tile_Test"):
        # Setup FSSpec Mock
        mock_fs = MagicMock()
        # Case 1: File not local, exists on S3
        mock_fs.exists.side_effect = lambda path: "BlueTopo_Tile_Test" in path
        mock_filesystem.return_value = mock_fs

        # Mock local file check (pathlib.Path.exists)
        # We need to patch Path.exists globally or on the specific object
        # simpler: patch cache path check inside provider if possible, or mock Path
        # BUT Path is hard to mock.
        # Strategy: Mock internal path check logic?
        # Or rely on the fact that the code checks `local_path.exists()`.
        # I can mock `pathlib.Path.exists` but that's risky.
        # Alternative: We trust the logic flows if we mock `fs.get`.

        # Wait, the code does `if not local_path.exists():`.
        # To test download, we need this to be False.
        # To test read, we need it to eventually be "openable".

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
            mock_open_rasterio.return_value = mock_da

            depth = provider.fetch_elevation(42.0, -70.0)

            # Verify download called
            assert mock_fs.get.called

            # Verify value returned (minus VDatum offset 0.0 implied)
            # VDatum mock in provider.__init__ -> global VDatumResolver ...
            # We should verify VDatum interaction too or mock it.
            # VDatumResolver uses requests.get. We might hit network if not mocked.
            # Provider init creates real VDatumResolver.
            # Let's mock provider.vdatum.get_navd88_to_lmsl_offset
            provider.vdatum = MagicMock()
            provider.vdatum.get_navd88_to_lmsl_offset.return_value = 1.0

            # Recalculate with mock setup
            depth = provider.fetch_elevation(42.0, -70.0)
            assert depth == -16.0


@patch("topobathysim.manager.GEBCO2025")
@patch("topobathysim.bluetopo.BlueTopoProvider.fetch_elevation")
def test_manager_smart_selection(mock_fetch_elev: MagicMock, mock_gebco_cls: MagicMock) -> None:
    # Setup GEBCO Mock
    mock_gebco_instance = MagicMock()
    mock_gebco_instance.sample_elevation.return_value = -100.0
    mock_gebco_cls.return_value = mock_gebco_instance

    # Case 1: BlueTopo Available
    mock_fetch_elev.return_value = -5.0

    manager = BathyManager()
    # Mock cover check to True
    manager.blue_topo = MagicMock()
    manager.blue_topo.is_covered.return_value = True

    depth = manager.get_elevation(42.0, -70.0)
    assert depth == -5.0
    assert manager.blue_topo.is_covered.called

    # Case 2: BlueTopo Not Covered -> GEBCO Fallback
    manager.blue_topo.is_covered.return_value = False

    depth = manager.get_elevation(0.0, 0.0)
    assert depth == -100.0


def test_source_info() -> None:
    manager = BathyManager()
    manager.blue_topo = MagicMock()
    manager.blue_topo.is_covered.return_value = True
    info = manager.get_source_info(42.0, -70.0)
    assert "BlueTopo" in info["source"]

    manager.blue_topo.is_covered.return_value = False
    info = manager.get_source_info(0.0, 0.0)
    assert "GEBCO" in info["source"]
