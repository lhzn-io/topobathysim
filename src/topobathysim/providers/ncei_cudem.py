"""
NOAA NCEI CUDEM Provider module.

This module implements the provider for the Continuously Updated Digital Elevation Model (CUDEM).
It handles tile resolution, downloading, caching, and processing of CUDEM data.
"""

import contextlib
import fcntl
import logging
import zipfile
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from ..quality import QualityClass
from ..vdatum import VDatumResolver
from .base import Provider
from .registry import registry

logger = logging.getLogger(__name__)


class CUDEMProvider(Provider):
    """
    Provider for NOAA/NCEI Continuously Updated Digital Elevation Model (CUDEM).
    Ninth-Arc-Second (~3m) Resolution.
    Acts as Tier 1 Data (Gap Filler for Intertidal/Coastal).
    """

    # Bucket: noaa-nos-coastal-lidar-pds
    # Prefix: dem/NCEI_ninth_Topobathy_2014_8483/
    BASE_S3_URL = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483"
    TILE_INDEX_ZIP = "tileindex_NCEI_ninth_Topobathy_2014.zip"

    def __init__(self, cache_dir: str = "~/.cache/topobathysim"):
        """
        Initialize the CUDEM provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        self.cache_dir = Path(cache_dir).expanduser() / "ncei_cudem"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir

        self.index_path = self.cache_dir / self.TILE_INDEX_ZIP
        self.index_shp_dir = self.cache_dir / "index_shp"
        self._gdf: gpd.GeoDataFrame | None = None
        self.vdatum = VDatumResolver()

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
    ) -> xr.DataArray:
        """
        Fetch CUDEM data for the given bounding box.
        """
        west, south, east, north = bbox
        matches = self.resolve_tiles(west, south, east, north)

        if matches.empty:
            raise KeyError(f"No CUDEM tiles found for bbox {bbox}")

        logger.info(f"Found {len(matches)} CUDEM tiles for bbox.")

        das: list[xr.DataArray] = []
        for _, row in matches.iterrows():
            url = self._get_tile_url(row)
            if url:
                tile_result = self.load_tile(row, (west, south, east, north))

                if isinstance(tile_result, xr.DataArray):
                    das.append(tile_result)
                elif tile_result is not None:
                    with contextlib.suppress(Exception):
                        das.append(cast(xr.DataArray, tile_result))

        if not das:
            raise KeyError(f"Failed to load valid CUDEM data for bbox {bbox}")

        if len(das) == 1:
            merged = das[0]
        else:
            try:
                merged = merge_arrays(das)
            except Exception as e:
                logger.error(f"Failed to merge CUDEM tiles: {e}")
                merged = das[0]

        # Reproject if needed
        if crs and merged.rio.crs and merged.rio.crs != crs:
            try:
                logger.info(f"Reprojecting CUDEM from {merged.rio.crs} to {crs}")
                merged = merged.rio.reproject(crs)
            except Exception as e:
                logger.warning(f"Reprojection failed: {e}")

        # Final Exact Clip
        try:
            merged = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs=crs)
        except Exception as e:
            logger.warning(f"Final clip failed: {e}")

        merged.name = "elevation"
        return cast(xr.DataArray, merged)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the CUDEM provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA CUDEM (Continuously Updated DEM)",
            "citation": "NOAA National Centers for Environmental Information.",
            "resolution": "~3m (1/9th arc-second)",
            "url": "https://coast.noaa.gov/digitalcoast/data/cudem.html",
        }

    def _ensure_index_loaded(self) -> None:
        """Download and load the spatial tile index."""
        if self._gdf is not None:
            return

        # 1. Download Index Zip
        if not self.index_path.exists():
            url = f"{self.BASE_S3_URL}/{self.TILE_INDEX_ZIP}"
            logger.info(f"Downloading CUDEM Tile Index from {url}...")
            try:
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(self.index_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.error(f"Failed to download CUDEM Index: {e}")
                return

        # 2. Extract Shapefile
        if not self.index_shp_dir.exists():
            try:
                with zipfile.ZipFile(self.index_path, "r") as z:
                    z.extractall(self.index_shp_dir)
            except Exception as e:
                logger.error(f"Failed to unzip CUDEM Index: {e}")
                return

        # 3. Load GDF
        try:
            shps = list(self.index_shp_dir.glob("*.shp"))
            if not shps:
                shps = list(self.index_shp_dir.rglob("*.shp"))

            if not shps:
                logger.warning("No shapefile found in CUDEM index zip.")
                return

            self._gdf = gpd.read_file(shps[0])

            if self._gdf is None:
                logger.warning("CUDEM Index Shapefile read returned None.")
                return

            # Normalize CRS to 4326
            if self._gdf.crs is None:
                self._gdf.set_crs("EPSG:4326", inplace=True)
            elif self._gdf.crs.to_string() != "EPSG:4326":
                self._gdf = self._gdf.to_crs("EPSG:4326")

            # Normalize Columns to lowercase
            if self._gdf is not None:
                self._gdf.columns = [c.lower() for c in self._gdf.columns]

        except Exception as e:
            logger.error(f"Failed to load CUDEM Index GDF: {e}")

    def resolve_tiles(self, west: float, south: float, east: float, north: float) -> gpd.GeoDataFrame:
        """Find tiles intersecting the bbox."""
        self._ensure_index_loaded()
        if self._gdf is None:
            return gpd.GeoDataFrame()

        bbox_geom = box(west, south, east, north)
        matches = self._gdf[self._gdf.intersects(bbox_geom)]
        return matches

    def _get_tile_url(self, row: Any) -> str | None:
        """Determine URL from row metadata."""
        for col in ["url", "path", "location", "fileurl"]:
            if row.get(col):
                val = str(row[col])
                if val.startswith("http"):
                    return val
                return f"{self.BASE_S3_URL}/{val}"
        return None

    def _ensure_tile_cached(self, url: str) -> Path | None:
        """Download remote tile to cache."""
        filename = Path(url).name
        local_path = self.cache_dir / filename
        lock_path = self.cache_dir / f"{filename}.lock"

        if local_path.exists():
            return local_path

        try:
            with open(lock_path, "w") as lock_file:
                # Blocking lock
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                if local_path.exists():
                    return local_path

                logger.info(f"Downloading CUDEM Tile: {filename}")
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code != 200:
                    logger.warning(f"CUDEM Tile 404/Error: {url}")
                    return None

                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)

                Path(lock_path).unlink(missing_ok=True)
                return local_path

        except Exception as e:
            logger.error(f"Failed to download CUDEM tile {url}: {e}")
            if local_path.exists():
                local_path.unlink()
            return None

    def load_tile(self, row: Any, bbox: tuple[float, float, float, float]) -> xr.DataArray | None:
        """Load single tile."""
        url = self._get_tile_url(row)
        if not url:
            return None

        path = self._ensure_tile_cached(url)
        if not path:
            return None

        # Zarr Cache
        zarr_path = path.parent / "zarr" / path.with_suffix(".zarr").name

        if zarr_path.exists():
            try:
                da_cached = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.info(f"CUDEM Zarr Cache Hit: {zarr_path.name}")
                return da_cached
            except Exception:
                pass

        # Load Raw
        try:
            da = rioxarray.open_rasterio(path, chunks={"x": 2048, "y": 2048})
            if isinstance(da, list):
                da = da[0]
            elif isinstance(da, xr.Dataset):
                da = da.to_array().isel(variable=0)

            if "band" in da.dims:
                da = da.isel(band=0).drop_vars("band")
            if da.rio.nodata is not None:
                da = da.where(da != da.rio.nodata)

            # Robust CRS
            if da.rio.crs is None:
                da.rio.write_crs("EPSG:4269", inplace=True)

            # Cache to Zarr
            try:
                if da.size > 0:
                    da = da.chunk({"y": 1024, "x": 1024})
                    da.name = "elevation"
                    da.to_zarr(zarr_path, mode="w", consolidated=True)
                    return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
            except Exception as e:
                logger.warning(f"Zarr cache write failed: {e}")

            return cast(xr.DataArray, da)

        except Exception as e:
            logger.error(f"Error reading CUDEM tile {path}: {e}")
            return None

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        """
        Return quality tier for the given coordinate.
        """
        matches = self.resolve_tiles(lon, lat, lon, lat)
        if not matches.empty:
            return QualityClass.INDIRECT
        return QualityClass.UNKNOWN


# Register
registry.register("cudem", CUDEMProvider)
