from abc import ABC, abstractmethod
from typing import Any

import xarray as xr


class Provider(ABC):
    """
    Abstract Base Class for TopoBathySim data providers.

    All data sources (GEBCO, BlueTopo, BAG, etc.) must implement this interface
    to ensure a unified data access layer for the fusion runtime.
    """

    @abstractmethod
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Fetch a data layer for the given bounding box.

        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat).
            resolution: Desired resolution in meters (approximate).
            crs: Coordinate Reference System string (default: EPSG:4326).
            **kwargs: Additional provider-specific arguments (e.g., filters).

        Returns:
            xr.DataArray: The fetched elevation data, strictly typed.
                          Must contain spatial coordinates (x/y or lon/lat).
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
