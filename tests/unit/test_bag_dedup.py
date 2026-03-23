"""Unit tests for BAG survey deduplication logic."""

from topobathysim.providers.ncei_bag import (
    _dedup_survey_resolutions,
    _parse_bag_resolution,
    _parse_survey_id,
)


class TestParseBagResolution:
    def test_centimeter(self) -> None:
        assert _parse_bag_resolution("H12696_MB_50cm_MLLW_1of3") == 0.5

    def test_meter_integer(self) -> None:
        assert _parse_bag_resolution("H12696_MB_4m_MLLW_combined") == 4.0

    def test_meter_decimal(self) -> None:
        assert _parse_bag_resolution("H12696_MB_0.5m_MLLW_1of1") == 0.5

    def test_vr_unknown(self) -> None:
        assert _parse_bag_resolution("F00810_MB_VR_MLLW_1of1") == 999.0

    def test_no_resolution(self) -> None:
        assert _parse_bag_resolution("some_random_file") == 999.0


class TestParseSurveyId:
    def test_h_prefix(self) -> None:
        assert _parse_survey_id("H12696_MB_50cm_MLLW_1of3") == "H12696"

    def test_w_prefix(self) -> None:
        assert _parse_survey_id("W00181_MB_4m_MLLW_combined") == "W00181"

    def test_e_prefix(self) -> None:
        assert _parse_survey_id("E01096_MB_7m_MLLW_6of7") == "E01096"

    def test_f_prefix(self) -> None:
        assert _parse_survey_id("F00810_MB_VR_MLLW_1of1") == "F00810"

    def test_d_prefix(self) -> None:
        assert _parse_survey_id("D00185_MB_16m_MLLW_Combined") == "D00185"

    def test_no_match(self) -> None:
        assert _parse_survey_id("unknown_file") == "unknown_file"


class TestDedupSurveyResolutions:
    """Test multi-resolution survey deduplication."""

    def _url(self, filename: str) -> str:
        return f"https://data.ngdc.noaa.gov/BAG/{filename}.bag"

    def test_no_target_resolution_returns_all(self) -> None:
        urls = [self._url("H12696_MB_50cm_MLLW_1of3"), self._url("H12696_MB_4m_MLLW_1of3")]
        assert _dedup_survey_resolutions(urls, target_resolution=None) == urls

    def test_single_survey_picks_closest_to_target(self) -> None:
        urls = [
            self._url("H12696_MB_50cm_MLLW_1of3"),  # 0.5m
            self._url("H12696_MB_2m_MLLW_2of3"),  # 2m
            self._url("H12696_MB_4m_MLLW_3of3"),  # 4m
        ]
        result = _dedup_survey_resolutions(urls, target_resolution=15.0)
        assert len(result) == 1
        assert "4m" in result[0]  # 4m is closest to 15m without being coarser

    def test_preserves_spatial_parts_at_chosen_resolution(self) -> None:
        urls = [
            self._url("H12696_MB_50cm_MLLW_1of3"),
            self._url("H12696_MB_4m_MLLW_3of3"),
            self._url("H12696_MB_4m_MLLW_combined"),
        ]
        result = _dedup_survey_resolutions(urls, target_resolution=15.0)
        # Both 4m parts should be kept
        assert len(result) == 2
        assert all("4m" in u for u in result)

    def test_distinct_surveys_all_kept(self) -> None:
        urls = [
            self._url("H12696_MB_4m_MLLW_1of3"),
            self._url("W00181_MB_4m_MLLW_1of2"),
            self._url("E01096_MB_7m_MLLW_4of7"),
        ]
        result = _dedup_survey_resolutions(urls, target_resolution=15.0)
        assert len(result) == 3  # all distinct surveys

    def test_all_coarser_than_target_keeps_finest(self) -> None:
        urls = [
            self._url("H12696_MB_16m_MLLW_1of1"),  # 16m > 15m
            self._url("H12696_MB_8m_MLLW_1of1"),  # 8m > 5m target
        ]
        # Target 5m, both are coarser — keep the 8m (finest available)
        result = _dedup_survey_resolutions(urls, target_resolution=5.0)
        assert len(result) == 1
        assert "8m" in result[0]

    def test_vr_bags_kept_as_fallback(self) -> None:
        urls = [
            self._url("F00810_MB_VR_MLLW_1of1"),  # VR = 999m (unknown)
        ]
        result = _dedup_survey_resolutions(urls, target_resolution=15.0)
        assert len(result) == 1

    def test_mixed_surveys_realistic(self) -> None:
        """Realistic scenario from Great Bay Estuary cell 58."""
        urls = [
            self._url("H12696_MB_50cm_MLLW_1of3"),
            self._url("H12614_MB_50cm_MLLW_1of3"),
            self._url("H12696_MB_2m_MLLW_2of3"),
            self._url("H12614_MB_2m_MLLW_2of3"),
            self._url("W00181_MB_2m_MLLW_1of2"),
            self._url("H12696_MB_4m_MLLW_3of3"),
            self._url("H12696_MB_4m_MLLW_combined"),
            self._url("H12614_MB_4m_MLLW_3of3"),
            self._url("H12614_MB_4m_MLLW_combined"),
            self._url("W00181_MB_4m_MLLW_2of2"),
            self._url("W00181_MB_4m_MLLW_combined"),
            self._url("H11296_LI_5m_MLLW_1of1"),
            self._url("E01096_MB_7m_MLLW_4of7"),
            self._url("E01096_MB_7m_MLLW_6of7"),
            self._url("F00810_MB_VR_MLLW_1of1"),
        ]
        result = _dedup_survey_resolutions(urls, target_resolution=15.0)

        filenames = [u.split("/")[-1] for u in result]
        # H12696: should keep 4m parts only (2 urls)
        h12696 = [f for f in filenames if f.startswith("H12696")]
        assert all("4m" in f for f in h12696)
        assert len(h12696) == 2

        # H12614: should keep 4m parts only (2 urls)
        h12614 = [f for f in filenames if f.startswith("H12614")]
        assert all("4m" in f for f in h12614)
        assert len(h12614) == 2

        # W00181: should keep 4m parts only (2 urls)
        w00181 = [f for f in filenames if f.startswith("W00181")]
        assert all("4m" in f for f in w00181)
        assert len(w00181) == 2

        # Single-resolution surveys kept as-is
        assert any("H11296" in f for f in filenames)
        assert any("F00810" in f for f in filenames)

        # E01096: both 7m parts kept
        e01096 = [f for f in filenames if f.startswith("E01096")]
        assert len(e01096) == 2

        # Total: 2+2+2+1+2+1 = 10 (down from 15)
        assert len(result) == 10
