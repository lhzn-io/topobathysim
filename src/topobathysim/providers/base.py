import logging
from abc import ABC, abstractmethod
from typing import Any

import xarray as xr
from pyproj import Transformer

logger = logging.getLogger(__name__)


class ProviderNoDataError(LookupError):
    """Raised when a provider has no data for the requested bounding box.

    This is a normal operating condition — it means the provider simply does not
    cover the requested area.  The runtime handles it by skipping the provider and
    continuing with the next step in the fusion policy.  It is intentionally *not*
    logged as an error.
    """


class Provider(ABC):
    """
    Abstract Base Class for TopoBathySim data providers.

    All data sources (GEBCO, BlueTopo, BAG, etc.) must implement this interface
    to ensure a unified data access layer for the fusion runtime.
    """

    def _normalize_bbox(
        self,
        bbox: tuple[float, float, float, float],
        crs: str = "EPSG:4326",
    ) -> tuple[float, float, float, float]:
        """
        Ensures the bounding box is in WGS84 (EPSG:4326) degrees.
        If input is already in 4326, returns it unchanged.
        If in any other CRS (e.g. 3857 meters), transforms it.

        Args:
            bbox: (west, south, east, north)
            crs: The CRS of the input bbox.

        Returns:
            Tuple[float, float, float, float]: (west, south, east, north) in WGS84.
        """
        if crs.upper() == "EPSG:4326":
            return bbox

        try:
            west, south, east, north = bbox
            # We use always_xy=True to ensure (lon, lat) ordering
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            w, s = transformer.transform(west, south)
            e, n = transformer.transform(east, north)
            logger.debug(f"Normalized bbox from {crs} to EPSG:4326: {bbox} -> ({w}, {s}, {e}, {n})")
            return (w, s, e, n)
        except Exception as exc:
            logger.warning(f"Failed to normalize bbox from {crs} to EPSG:4326: {exc}")
            return bbox

    @abstractmethod
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetch a data layer for the given bounding box.

        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat).
            resolution: Desired resolution in meters (approximate).
            crs: Coordinate Reference System string (default: EPSG:4326).
            **kwargs: Additional provider-specific arguments (e.g., filters).

        Returns:
            xr.DataArray or xr.Dataset:
                - Legacy: Returns an `xr.DataArray` of elevation.
                - Detailed Provenance: Returns an `xr.Dataset` containing 'elevation' (float32) and
                  'source_id' (uint32) DataArrays, with a dictionary in `attrs['provenance_dict']`
                  mapping `source_id` integers to source asset names.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata about the provider.

        Returns:
            Dict[str, Any]: Dictionary containing provider name, version,
                            citation, and any other relevant metadata.
        """
        pass
