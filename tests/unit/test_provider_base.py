from typing import Any

from topobathysim.providers.base import Provider


class DummyProvider(Provider):
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> Any:
        """Use our new helper."""
        return self._normalize_bbox(bbox, crs)

    def get_metadata(self) -> dict[str, Any]:
        """Return dummy metadata."""
        return {"name": "dummy"}


def test_normalize_bbox_4326() -> None:
    """Test normalization with EPSG:4326."""
    provider = DummyProvider()
    bbox = (-74.0, 40.0, -73.0, 41.0)
    # Should return unchanged
    result = provider.fetch_layer(bbox, crs="EPSG:4326")
    assert result == bbox


def test_normalize_bbox_3857() -> None:
    """Test normalization from EPSG:3857."""
    provider = DummyProvider()
    # NYC-ish in 3857 meters
    bbox_3857 = (-8238000.0, 4970000.0, -8226000.0, 4983000.0)
    result = provider.fetch_layer(bbox_3857, crs="EPSG:3857")

    # Verify result is in degrees (-180 to 180, -90 to 90)
    for coord in result:
        assert -180 <= coord <= 180

    # Specific check: -8238000 is approx -74 lon
    assert -74.5 < result[0] < -73.5
    assert 40.0 < result[1] < 41.0
