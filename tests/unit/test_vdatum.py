from unittest.mock import MagicMock, patch

import pytest

from topobathysim.vdatum import VDatumResolver


def test_vdatum_valid_response() -> None:
    """Test that a valid t_z response is returned correctly."""
    with patch("topobathysim.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": "1.5"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        val = VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)
        assert val == 1.5


def test_vdatum_nodata_response_raises_value_error() -> None:
    """Test that extreme NoData values (-999999.0) raise a ValueError."""
    with patch("topobathysim.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": "-999999.0"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(ValueError, match="VDatum API returned NoData value"):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)


def test_vdatum_missing_tz_raises_value_error() -> None:
    """Test that a missing t_z field raises a ValueError."""
    with patch("topobathysim.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Out of bounds"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(ValueError, match="VDatum API returned no elevation data"):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)
