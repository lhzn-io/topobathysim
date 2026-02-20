"""
GEBCO 2025 Provider module.

This module implements the provider for the GEBCO 2025 Grid, a global terrain model for ocean and land.
It uses 'bmi-topography' or OPeNDAP access to fetch data, managing caching locally.
"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any

import xarray as xr
from bmi_topography import Topography

from .base import Provider
from .registry import registry

logger = logging.getLogger(__name__)

BBox = namedtuple("BBox", ["west", "south", "east", "north"])


class GEBCO2025Provider(Topography, Provider):
    """
    BMI-compliant interface for GEBCO 2025 data.
    Wraps bmi-topography to fetch from OPeNDAP with specific corrections.
    """

    # Official GEBCO 2025 OpenDAP URL
    # Note: 'sub_ice_topo_bathymetry' is the elevation variable
    OPENDAP_URL = "dap2://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2025/sub_ice_topography_bathymetry/netcdf/gebco_2025_sub_ice.nc"

    RESOLUTION_ARCSEC = 15
    HALF_PIXEL_OFFSET = (15 / 3600) * 0.5  # Degrees

    def __init__(
        self,
        dem_type: str = "GEBCO_2025",
        south: float = -90,
        north: float = 90,
        west: float = -180,
        east: float = 180,
        output_format: str = "GTiff",
        cache_dir: str = "~/.cache/topobathysim",
    ):
        """
        Initialize the GEBCO provider.

        Args:
            dem_type: Type of DEM (default: GEBCO_2025).
            south, north, west, east: Global bounds.
            output_format: Output format (default: GTiff).
            cache_dir: Directory to store cached data files.
        """
        p = Path(cache_dir).expanduser() / "gebco_2025"
        p.mkdir(parents=True, exist_ok=True)
        (p / "zarr").mkdir(exist_ok=True)  # Zarr subdir
        super().__init__(
            dem_type="SRTMGL3",  # Pass valid type to satisfy validation; we override fetch() anyway
            south=south,
            north=north,
            west=west,
            east=east,
            output_format=output_format,
            cache_dir=str(p),
        )
        self.tid_data: xr.DataArray | None = None
        self._da: xr.DataArray | None = None

    def extract_subset(self, bbox: BBox) -> xr.DataArray:
        """
        Implementation of Topography.extract_subset.
        For now just raises NotImplementedError as we use fetch_layer.
        """
        raise NotImplementedError("Use fetch_layer instead")

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Implements Provider.fetch_layer
        """
        west, south, east, north = bbox
        bbox_override = BBox(west, south, east, north)
        return self.fetch(bbox_override=bbox_override)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the GEBCO provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "GEBCO 2025",
            "citation": "GEBCO Compilation Group (2025) GEBCO 2025 Grid.",
            "resolution": "15 arc-second",
            "url": self.OPENDAP_URL,
        }

    def fetch(self, bbox_override: BBox | None = None) -> xr.DataArray:
        """
        Fetch data from GEBCO 2025 OPeNDAP server, utilizing a local Zarr cache
        tiled by 1x1 degree chunks to minimize repeat OPeNDAP hits.
        """
        # 1. Determine which 1x1 degree tiles we need
        import math
        import warnings

        current_bbox = bbox_override if bbox_override else self.bbox

        # Round bounds to nearest degree to find covering tiles
        s, w, n, e = current_bbox.south, current_bbox.west, current_bbox.north, current_bbox.east

        min_lat = math.floor(s)
        max_lat = math.ceil(n)  # Use ceil for north/east to ensure coverage
        min_lon = math.floor(w)
        max_lon = math.ceil(e)

        das_to_merge = []

        ds_remote = None  # Lazy init

        for lat_idx in range(min_lat, max_lat):
            for lon_idx in range(min_lon, max_lon):
                # Define 1x1 degree tile key and path
                # e.g. gebco_2025_n40_w74
                lat_str = f"n{lat_idx}" if lat_idx >= 0 else f"s{abs(lat_idx)}"
                lon_str = f"e{lon_idx}" if lon_idx >= 0 else f"w{abs(lon_idx)}"
                tile_key = f"gebco_2025_{lat_str}_{lon_str}"

                # Store in zarr subdirectory
                zarr_dir = Path(self.cache_dir) / "zarr"
                cache_path = zarr_dir / f"{tile_key}.zarr"
                da_tile = None

                # A. Try loading from Zarr Cache
                if cache_path.exists():
                    try:
                        # Use chunks="auto" for Dask
                        # Suppress "grid_mapping not found" warning for existing caches derived from netCDF

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message="Variable.s. referenced in grid_mapping not in variables"
                            )
                            da_tile = xr.open_dataarray(
                                cache_path, engine="zarr", chunks="auto", decode_coords="all"
                            )
                        # Ensure CRS
                        if da_tile.rio.crs is None:
                            logger.warning(f"GEBCO Zarr Cache {tile_key} missing CRS. Setting EPSG:4326")
                            da_tile.rio.write_crs("EPSG:4326", inplace=True)

                        # Basic check
                        if da_tile.size == 0:
                            da_tile = None
                        else:
                            logger.info(f"GEBCO Zarr Cache Hit: {tile_key}")
                    except Exception as err:
                        logger.warning(f"Corrupt GEBCO Zarr cache {cache_path}: {err}")
                        import shutil

                        if cache_path.exists():
                            shutil.rmtree(cache_path)
                        da_tile = None

                # B. Fetch from OPeNDAP if missing
                if da_tile is None:
                    import threading

                    from filelock import FileLock

                    # Class level thread lock for deduplication
                    if not hasattr(self.__class__, "_locks"):
                        self.__class__._locks = {}
                        self.__class__._locks_lock = threading.Lock()

                    with self.__class__._locks_lock:
                        if tile_key not in self.__class__._locks:
                            self.__class__._locks[tile_key] = threading.Lock()
                        thread_lock = self.__class__._locks[tile_key]

                    lock_path = cache_path.with_suffix(".zarr.lock")
                    with thread_lock, FileLock(lock_path):
                        # Double check
                        if cache_path.exists():
                            logger.info(f"GEBCO Zarr Cache Hit (after lock): {tile_key}")
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="Variable.s. referenced in grid_mapping not in variables",
                                )
                                da_tile = xr.open_dataarray(
                                    cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                )
                        else:
                            logger.info(f"GEBCO Zarr Cache Miss: {tile_key}")
                            try:
                                if (
                                    not hasattr(self.__class__, "_ds_remote")
                                    or self.__class__._ds_remote is None
                                ):
                                    try:
                                        logger.info("Initializing global GEBCO OPeNDAP Connection...")
                                        self.__class__._ds_remote = xr.open_dataset(
                                            self.OPENDAP_URL, engine="pydap"
                                        )
                                    except Exception as connection_err:
                                        logger.error(f"Failed to connect to GEBCO OPeNDAP: {connection_err}")
                                        raise

                                ds_remote = self.__class__._ds_remote

                                logger.info(f"Downloading GEBCO 1x1 Tile to Cache: {tile_key}")

                                # Slice exactly this 1x1 degree chunk
                                subset = ds_remote.sel(
                                    lat=slice(lat_idx, lat_idx + 1.0),
                                    lon=slice(lon_idx, lon_idx + 1.0),
                                )

                                if "sub_ice_topo_bathymetry" in subset:
                                    da_source = subset["sub_ice_topo_bathymetry"]
                                elif "elevation" in subset:
                                    da_source = subset["elevation"]
                                else:
                                    logger.warning(f"Variable missing in GEBCO tile {tile_key}")
                                    continue

                                da_source = da_source.load()
                                if da_source.rio.crs is None:
                                    da_source.rio.write_crs("EPSG:4326", inplace=True)

                                if "grid_mapping" in da_source.attrs:
                                    del da_source.attrs["grid_mapping"]
                                if "grid_mapping" in da_source.encoding:
                                    del da_source.encoding["grid_mapping"]

                                if da_source.size > 0:
                                    da_source = da_source.chunk({"lat": 240, "lon": 240})
                                    da_source.name = "elevation"
                                    da_source.to_zarr(cache_path, mode="w", consolidated=True)
                                    logger.info(f"GEBCO Zarr Cache Created: {tile_key}")

                                    with warnings.catch_warnings():
                                        warnings.filterwarnings(
                                            "ignore",
                                            message="Variable.s. referenced in grid_mapping not in variables",
                                        )
                                        da_tile = xr.open_dataarray(
                                            cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                        )
                                        if da_tile.rio.crs is None:
                                            logger.warning(
                                                f"GEBCO Zarr (Newly Created) {tile_key} missing CRS. "
                                                "Setting EPSG:4326"
                                            )
                                            da_tile.rio.write_crs("EPSG:4326", inplace=True)

                                else:
                                    logger.warning(f"GEBCO Tile {tile_key} returned empty data.")
                            except Exception as tile_err:
                                logger.error(f"Failed to fetch GEBCO tile {tile_key}: {tile_err}")
                                continue

                    # Attempt to clean up lock file if cache was successfully created
                    try:
                        if cache_path.exists() and lock_path.exists():
                            lock_path.unlink()
                    except Exception:
                        pass

                if da_tile is not None:
                    das_to_merge.append(da_tile)

        if not das_to_merge:
            raise KeyError(f"Failed to load any GEBCO data for bounds {current_bbox}")

        # 2. Merge tiles if multiple
        try:
            if len(das_to_merge) == 1:
                self._da = das_to_merge[0]
            else:
                normalized = []
                for d in das_to_merge:
                    d.name = "elevation"
                    normalized.append(d)

                self._da = xr.merge(normalized)["elevation"]

        except Exception as merge_err:
            logger.error(f"Merge failed: {merge_err}")
            self._da = das_to_merge[0]  # Fallback to first

        # 3. Slice to request bounds with padding
        # Pad bounds by 0.01 degrees to ensure data is returned for high zoom levels
        # where the requested extent is smaller than the native (15 arcsec) resolution.
        if self._da is not None:
            pad = 0.01
            self._da = self._da.sel(
                lat=slice(max(s - pad, -90), min(n + pad, 90)),
                lon=slice(max(w - pad, -180), min(e + pad, 180)),
            )

        if self._da is None or self._da.size == 0:
            raise KeyError(f"Failed to fetch GEBCO data for {self.layer_name}")

        logger.debug("Found GEBCO 2025 Coverage")
        # Ensure consistent naming
        self._da.name = "elevation"

        return self._da

    def load(self) -> xr.DataArray:
        """Standard BMI load method calling fetch"""
        return self.fetch()

    def get_tid_classification(self) -> xr.DataArray | None:
        """Returns the cached TID DataArray"""
        return self.tid_data

    def sample_elevation(self, lat: float, lon: float) -> float:
        """
        Returns bilinearly interpolated elevation at the given coordinate.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Depth/Elevation in meters (negative for depth).
        """
        if self._da is None:
            raise RuntimeError("Data not loaded. Call fetch() or load() first.")

        # xarray's interp handles bilinear interpolation by default (linear)
        val = self._da.interp(lat=lat, lon=lon, method="linear")
        return float(val.values)


# Register the provider
registry.register(Path(__file__).stem, GEBCO2025Provider)
