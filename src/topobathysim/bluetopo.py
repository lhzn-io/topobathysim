import logging
from pathlib import Path

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from shapely.geometry import Point

from .quality import QualityClass
from .vdatum import VDatumResolver

logger = logging.getLogger(__name__)


class BlueTopoProvider:
    """
    Provider for NOAA BlueTopo High-Resolution Bathymetry.
    Accesses Cloud Optimized GeoTIFFs (COGs) from AWS S3.
    Resolves tile IDs using the official BlueTopo Tile Scheme GPKG.
    Implements 'Download-and-Cache' strategy for full tiles.
    """

    BUCKET_BASE = "noaa-ocs-nationalbathymetry-pds/BlueTopo"  # Bucket path for s3fs
    S3_URI_BASE = "s3://noaa-ocs-nationalbathymetry-pds/BlueTopo"

    # Official Tile Scheme URL
    TILE_SCHEME_URL = (
        "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/"
        "BlueTopo/_BlueTopo_Tile_Scheme/BlueTopo_Tile_Scheme_20260130_200842.gpkg"
    )

    def __init__(self, cache_dir: str = "~/.cache/topobathysim"):
        self.vdatum = VDatumResolver()
        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "bluetopo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Tile Scheme stays in root or moves? Let's move to bluetopo dir too to be clean.
        self.scheme_path = self.cache_dir / "BlueTopo_Tile_Scheme.gpkg"
        self._gdf = None

    def _ensure_scheme_loaded(self) -> None:
        """
        Downloads and loads the Tile Scheme GPKG if not already loaded.
        """
        if self._gdf is not None:
            return

        if not self.scheme_path.exists():
            try:
                # Ensure directory exists (it might have been cleared runtime)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                response = requests.get(self.TILE_SCHEME_URL, stream=True)
                response.raise_for_status()
                with open(self.scheme_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.warning(f"Failed to download BlueTopo Tile Scheme: {e}")
                return

        try:
            self._gdf = gpd.read_file(self.scheme_path)
            if self._gdf is not None and hasattr(self._gdf, "sindex"):
                _ = self._gdf.sindex
        except Exception as e:
            logger.error(f"Failed to load Tile Scheme GPKG: {e}")
            self._gdf = None

    def resolve_tile_id(self, lat: float, lon: float) -> str | None:
        """
        Queries the Tile Scheme to find the tile covering the coordinate.
        Returns the tile identifier (e.g. 'BlueTopo_Tile_X_Y_...') or None.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        point = Point(lon, lat)

        # Proj check (assuming GDF might not be 4326)
        query_point = point
        if self._gdf.crs and self._gdf.crs.to_string() != "EPSG:4326":
            pt_gdf = gpd.GeoSeries([point], crs="EPSG:4326")
            try:
                pt_gdf_proj = pt_gdf.to_crs(self._gdf.crs)
                query_point = pt_gdf_proj[0]
            except Exception:
                pass

        try:
            matches = self._gdf[self._gdf.contains(query_point)]
        except Exception:
            return None

        if matches.empty:
            return None

        row = matches.iloc[0]
        # 'tile' is the primary ID column in BlueTopo Scheme 2025+
        for col in ["tile", "file_name", "tile_name", "name", "tile_id", "standard_name"]:
            if col in row:
                return str(row[col])
        return None

    def resolve_tiles_in_bbox(self, west: float, south: float, east: float, north: float) -> list[str]:
        """
        Finds all tiles intersecting the bounding box.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return []

        from shapely.geometry import box

        search_box = box(west, south, east, north)

        # CRS check
        query_geom = search_box
        if self._gdf.crs and self._gdf.crs.to_string() != "EPSG:4326":
            gdf_box = gpd.GeoSeries([search_box], crs="EPSG:4326")
            try:
                gdf_proj = gdf_box.to_crs(self._gdf.crs)
                query_geom = gdf_proj[0]
            except Exception:
                pass

        try:
            matches = self._gdf[self._gdf.intersects(query_geom)]
        except Exception:
            return []

        if matches.empty:
            return []

        results = []
        for _, row in matches.iterrows():
            for col in ["tile", "file_name", "tile_name", "name", "tile_id", "standard_name"]:
                if col in row:
                    results.append(str(row[col]))
                    break

        # logger.debug(f"Resolved BlueTopo Tiles: {results}")
        return list(set(results))

    def is_covered(self, lat: float, lon: float) -> bool:
        return self.resolve_tile_id(lat, lon) is not None

    def _ensure_tile_cached(self, tile_id: str) -> Path | None:
        """
        Ensures the specified tile is present in the cache.
        Downloads from S3 if necessary.
        Returns the local path or None if not found/failed.
        """
        import fcntl

        try:
            # Check if we already have a file matching this tile ID in cache
            # This is tricky with globs, so we must rely on specific filename logic if possible.
            # However, BlueTopo filenames are inconsistent in hash/date.
            # We will search glob first (Fast Path).
            candidates = list(self.cache_dir.glob(f"*{tile_id}*.tiff"))
            if candidates:
                logger.debug(f"BlueTopo Cache Hit: {candidates[0]}")
                return candidates[0]

            # Not found locally, attempt download
            # We need to know the S3 Path to construct a local filename and lock it.
            fs = fsspec.filesystem("s3", anon=True)
            search_pattern = f"{self.BUCKET_BASE}/{tile_id}/*.tiff"
            files = fs.glob(search_pattern)

            if not files:
                logger.warning(f"BlueTopo: No files found in S3 for pattern: {search_pattern}")
                return None

            source_path = files[0]
            real_filename = Path(source_path).name
            local_path = self.cache_dir / real_filename
            lock_path = self.cache_dir / f"{real_filename}.lock"
            temp_path = self.cache_dir / f".tmp_{real_filename}"

            # 2. Acquire Lock
            with open(lock_path, "w") as lock_file:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)

                    # 3. Double Check
                    if local_path.exists():
                        logger.info(f"BlueTopo Cache Hit (After Lock): {local_path}")
                        return local_path

                    logger.info(f"Downloading {source_path} to {local_path}...")
                    fs.get(source_path, str(temp_path))

                    # Atomic Rename
                    Path(temp_path).rename(local_path)
                    return local_path

                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    # lock_file is closed automatically by 'with'

        except Exception as e:
            logger.error(f"BlueTopo Cache Error: {e}")
            if "temp_path" in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
            return None

    def fetch_elevation(self, lat: float, lon: float) -> float | None:
        """
        Fetches elevation from BlueTopo.
        """
        tile_id = self.resolve_tile_id(lat, lon)
        if not tile_id:
            return None

        try:
            local_path = self._ensure_tile_cached(tile_id)
            if not local_path:
                return None

            # Open with rioxarray
            da = rioxarray.open_rasterio(local_path)

            # Sample
            sample_x, sample_y = lon, lat
            if da.rio.crs and da.rio.crs != "EPSG:4326":
                # Project point to raster CRS
                from pyproj import Transformer

                transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
                sample_x, sample_y = transformer.transform(lon, lat)

            val = da.sel(x=sample_x, y=sample_y, method="nearest")
            if "band" in val.dims or val.size > 1:
                val = val.isel(band=0)
            val = val.values.item()

            if val == da.rio.nodata:
                return None

            # VDatum
            offset = self.vdatum.get_navd88_to_lmsl_offset(lat, lon)
            return float(val) - offset

        except Exception as e:
            logger.error(f"BlueTopo Fetch Error: {e}", exc_info=True)
            return None

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        if self.is_covered(lat, lon):
            return QualityClass.DIRECT
        return QualityClass.UNKNOWN

    def load_tile_as_da(self, tile_id: str, bbox: tuple[float, float, float, float]) -> "xr.DataArray | None":
        """
        Loads the cached tile and clips to bbox (west, south, east, north).
        Downloads if missing.
        """
        try:
            path = self._ensure_tile_cached(tile_id)
            if not path:
                return None

            da = rioxarray.open_rasterio(path)

            if "band" in da.dims:
                da = da.isel(band=0).drop_vars("band")

            # Clip
            # WARNING: Clipping here can cause edge artifacts (transparency triangles)
            # if the reprojection of the bbox doesn't fully cover the rotated raster corner.
            # We return the full tile and let main.py's reproject_match handle the cropping.
            return da

            # try:
            #     # Pass crs="EPSG:4326" so rioxarray handles projection of bounds to raster CRS
            #     return da.rio.clip_box(
            #         minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
            #     )
            # except Exception as e:
            #     logger.warning(f"Clip failed, returning full tile: {e}")
            #     return da

        except Exception as e:
            logger.error(f"Load Tile Error: {e}")
            return None
