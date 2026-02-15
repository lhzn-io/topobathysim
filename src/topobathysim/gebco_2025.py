import logging
from pathlib import Path

import xarray as xr
from bmi_topography import Topography

logger = logging.getLogger(__name__)


class GEBCO2025Provider(Topography):
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

    def fetch(self) -> xr.DataArray:
        """
        Fetch data from GEBCO 2025 OPeNDAP server, utilizing a local Zarr cache
        tiled by 1x1 degree chunks to minimize repeat OPeNDAP hits.
        """
        # 1. Determine which 1x1 degree tiles we need
        import math
        import warnings

        # Round bounds to nearest degree to find covering tiles
        # bmi-topography (0.9+) exposes bounds via .bbox property (BoundingBox object)
        s, w, n, e = self.bbox.south, self.bbox.west, self.bbox.north, self.bbox.east

        min_lat = math.floor(s)
        max_lat = math.ceil(n)  # Use ceil for north/east to ensure coverage
        min_lon = math.floor(w)
        max_lon = math.ceil(e)

        das_to_merge = []

        # OPeNDAP specific handling for requests
        # We need a fresh dataset instance sometimes if timeouts occur
        try:
            ds_remote = xr.open_dataset(self.OPENDAP_URL, engine="pydap")
        except Exception as connection_err:
            logger.error(f"Failed to connect to GEBCO OPeNDAP: {connection_err}")
            raise

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
                    from filelock import FileLock

                    lock_path = cache_path.with_suffix(".zarr.lock")
                    with FileLock(lock_path):
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
                                else:
                                    logger.warning(f"GEBCO Tile {tile_key} returned empty data.")
                            except Exception as tile_err:
                                logger.error(f"Failed to fetch GEBCO tile {tile_key}: {tile_err}")
                                continue

                if da_tile is not None:
                    das_to_merge.append(da_tile)

        if not das_to_merge:
            raise KeyError(f"Failed to load any GEBCO data for bounds {self.bbox}")

        # 2. Merge tiles if multiple
        try:
            if len(das_to_merge) == 1:
                self._da = das_to_merge[0]
            else:
                # rename to ensure consistency strictly
                normalized = []
                for d in das_to_merge:
                    d.name = "elevation"
                    normalized.append(d)

                self._da = xr.merge(normalized)["elevation"]

        except Exception as merge_err:
            logger.error(f"Merge failed: {merge_err}")
            self._da = das_to_merge[0]  # Fallback to first

        # 3. Slice to exact request bounds (Crop)
        # This crop is fast against the lazy Zarr backends
        if self._da is not None:
            self._da = self._da.sel(lat=slice(s, n), lon=slice(w, e))

        if self._da is None:
            raise KeyError("Failed to load elevation data")

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
