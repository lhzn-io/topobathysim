"""
NOAA BlueTopo Provider module.

This module implements the provider for NOAA's BlueTopo data set, providing high-resolution
bathymetry from modern surveys. It handles S3 access, tile resolution via RAT, and Zarr caching.
"""

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from filelock import FileLock
from rioxarray.merge import merge_arrays
from shapely.geometry import Point, box

from ..quality import QualityClass
from ..runtime import should_consolidate
from ..vdatum import VDatumResolver
from .base import Provider, ProviderNoDataError
from .registry import registry

logger = logging.getLogger(__name__)


class NoaaBlueTopoProvider(Provider):
    """
    Provider for NOAA BlueTopo High-Resolution Bathymetry.
    Accesses Cloud Optimized GeoTIFFs (COGs) from AWS S3.
    Resolves tile IDs using the official BlueTopo Tile Scheme GPKG.
    Implements 'Download-and-Cache' strategy for full tiles.
    """

    BUCKET_BASE = "noaa-ocs-nationalbathymetry-pds/BlueTopo"  # Bucket path for s3fs
    S3_URI_BASE = "s3://noaa-ocs-nationalbathymetry-pds/BlueTopo"

    # Official Tile Scheme URL (Fallback)
    TILE_SCHEME_URL = (
        "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/"
        "BlueTopo/_BlueTopo_Tile_Scheme/BlueTopo_Tile_Scheme_20260206_112953.gpkg"
    )

    # Persistent cache mapping tile_id -> resolved HTTPS URL.
    # Format: {"BA04XA": "https://noaa-ocs-nationalbathymetry-pds.s3.../BA04XA_20240101.tiff", ...}
    # Stored in: ~/.cache/topobathysim/noaa_bluetopo/tile_url_cache.json
    #
    # BlueTopo tile filenames include a date stamp so their S3 keys change when the NOAA
    # team republishes a tile with updated bathymetry.  Invalidate this cache to pick up
    # renamed/republished tile assets:
    #   manage_discovery_cache.py --invalidate noaa_bluetopo
    TILE_URL_CACHE_PATH = Path("~/.cache/topobathysim/noaa_bluetopo/tile_url_cache.json").expanduser()

    _singleton: ClassVar["NoaaBlueTopoProvider | None"] = None
    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> "NoaaBlueTopoProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(NoaaBlueTopoProvider, cls._singleton)

    def __init__(self, cache_dir: str = "~/.cache/topobathysim") -> None:
        """
        Initialize the BlueTopo provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if getattr(self, "_initialized", False):
            return

        self.vdatum = VDatumResolver()
        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "noaa_bluetopo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir
        self.scheme_path = self.cache_dir / "BlueTopo_Tile_Scheme.gpkg"
        self._gdf = None
        self._initialized = True

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetch BlueTopo layer for the given bounding box.
        Resolves, fetches, and merges all intersecting tiles.
        """
        # Ensure the bbox is in EPSG:4326 for tile resolution and metadata checks
        west, south, east, north = self._normalize_bbox(bbox, crs)

        # 1. Resolve Tiles
        tile_ids = self.resolve_tiles_in_bbox(west, south, east, north)
        if not tile_ids:
            raise ProviderNoDataError(f"No BlueTopo tiles found for bbox {bbox}")

        logger.info(f"BlueTopo fetch: Resolved {len(tile_ids)} tiles for bbox {bbox}")

        # 2. Fetch/Load Tiles
        das = []
        provenance_dict = {}
        for tid in tile_ids:
            # Pass the query bbox to maximize efficiency if underlying method supports it
            # defaulting to full tile load via existing method
            ds = self.load_tile_as_da(tid, bbox)
            if ds is not None:
                import hashlib

                import numpy as np

                da_elev = ds["elevation"]

                if "source_id" in ds:
                    da_src = ds["source_id"]
                    unique_vals = np.unique(da_src.values)

                    translated_src = xr.zeros_like(da_src, dtype=np.uint32)
                    for pval in unique_vals:
                        if pval == 0 or np.isnan(pval):
                            continue

                        pval_int = int(pval)
                        survey_id = self._resolve_from_sidecar_rat(tid, pval_int)
                        # fallback
                        if not survey_id:
                            mid_lat = (south + north) / 2
                            mid_lon = (west + east) / 2
                            survey_id = self.get_source_survey_id(mid_lat, mid_lon)

                        if survey_id:
                            # Keep BlueTopo data available as a resilient fallback.
                            # Runtime fusion order resolves final precedence.
                            survey_id = survey_id.strip()

                        name = f"BlueTopo: {survey_id}" if survey_id else f"BlueTopo: {tid}_{pval_int}"
                        digest = hashlib.md5(name.encode()).hexdigest()
                        project_uid = int(digest, 16) % 100000 + 50000

                        provenance_dict[project_uid] = {
                            "name": name,
                            "provider": "noaa_bluetopo",
                        }

                        translated_src = xr.where(da_src == pval, project_uid, translated_src)

                    p_source = translated_src.astype(np.uint32)
                else:
                    digest = hashlib.md5(tid.encode()).hexdigest()
                    project_uid = int(digest, 16) % 100000 + 50000
                    mid_lat = (south + north) / 2
                    mid_lon = (west + east) / 2
                    survey_id = self.get_source_survey_id(mid_lat, mid_lon)

                    if survey_id:
                        # Keep BlueTopo data available as a resilient fallback.
                        # Runtime fusion order resolves final precedence.
                        survey_id = survey_id.strip()

                    name = f"BlueTopo: {survey_id}" if survey_id else f"BlueTopo: {tid}"

                    # Parse local resolution from survey_id (e.g. H13385_MB_50cm_MLLW -> 50cm)
                    res_str = "4m (Default)"
                    if survey_id:
                        import re

                        # Common patterns: _50cm_, _1m_, _2m_, _4m_
                        match = re.search(r"_(\d+(?:\.\d+)?(?:m|cm))_", survey_id, re.IGNORECASE)
                        if match:
                            res_str = match.group(1)

                    provenance_dict[project_uid] = {
                        "name": name,
                        "provider": "noaa_bluetopo",
                        "resolution": res_str,
                    }
                    p_source = xr.where(da_elev.notnull(), project_uid, 0).astype(np.uint32)

                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                # Mask elevation where source was filtered out (0)
                da_elev = da_elev.where(p_source != 0)

                p_ds = xr.Dataset({"elevation": da_elev, "source_id": p_source})
                if da_elev.rio.crs:
                    p_ds.rio.write_crs(da_elev.rio.crs, inplace=True)
                p_ds.rio.write_transform(da_elev.rio.transform(), inplace=True)

                das.append(p_ds)

        if not das:
            raise ProviderNoDataError(f"Failed to load any BlueTopo data for bbox {bbox}")

        # 3. Handle Mixed CRSs (e.g. crossing UTM zones)
        # If tiles are in different CRSs, merge_arrays will fail or produce garbage.
        # We must reproject to a common CRS (the requested one, or 4326) before merging.
        unique_crss = set()
        for ds in das:
            if ds.rio.crs:
                unique_crss.add(ds.rio.crs.to_string())

        if len(unique_crss) > 1:
            logger.info(f"BlueTopo tiles have mixed CRSs {unique_crss}. Reprojecting to {crs} before merge.")
            reprojected_das = []
            for ds in das:
                try:
                    # Reproject to target and clip loosely to avoid huge memory usage
                    # Note: We can't clip easily before reproject if we are in mixed zones.
                    # Just reprojecting handles it.
                    reprojected_das.append(ds.rio.reproject(crs))
                except Exception as e:
                    logger.warning(f"Failed to reproject BlueTopo tile during merge: {e}")
                    reprojected_das.append(ds)
            das = reprojected_das

        # 4. Merge
        if len(das) == 1:
            merged = das[0]
        else:
            try:
                # Merge arrays natively puts last element on top
                elevs = [ds["elevation"] for ds in das]
                sources = [ds["source_id"] for ds in das]

                final_elev = merge_arrays(elevs)
                final_src = merge_arrays(sources)
                merged = xr.Dataset({"elevation": final_elev, "source_id": final_src})
            except Exception as e:
                logger.error(f"Failed to merge BlueTopo tiles: {e}")
                merged = das[0]  # Fallback

        # OPTIMIZATION: Clip in Source CRS first (using 4326 bounds)
        # We add a 10% buffer to the clip bounds. This prevents "rotated swatch" gaps
        # where the source-CRS clip (axis-aligned in source) becomes a rotated rectangle
        # in the target CRS that doesn't fully cover the corners of the target bbox.
        try:
            pad_x = (east - west) * 0.1
            pad_y = (north - south) * 0.1

            merged = merged.rio.clip_box(
                minx=west - pad_x,
                miny=south - pad_y,
                maxx=east + pad_x,
                maxy=north + pad_y,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
        except Exception as e:
            logger.warning(f"BlueTopo Clip failed (no overlap?): {e}")
            pass

        # Reproject to Requested CRS if needed
        if crs and merged.rio.crs and merged.rio.crs != crs:
            try:
                merged = merged.rio.reproject(crs)
            except Exception as e:
                logger.warning(f"Reprojection failed: {e}")

        # Final Exact Clip to exact bbox (cleanup)
        with contextlib.suppress(Exception):
            merged_elev = merged["elevation"].rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
            merged_src = merged["source_id"].rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
            merged = xr.Dataset({"elevation": merged_elev, "source_id": merged_src})

        merged["elevation"].name = "elevation"
        merged.attrs["provenance_dict"] = provenance_dict
        logger.debug("Found BlueTopo Coverage")

        from typing import cast

        return cast(xr.Dataset, merged)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the BlueTopo provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA BlueTopo",
            "citation": "NOAA Office of Coast Survey (2025). BlueTopo™.",
            "resolution": "Variable (approx 4-8m)",
            "url": "https://nauticalcharts.noaa.gov/data/bluetopo.html",
        }

    def _resolve_scheme_url(self) -> str:
        """
        Resolves the latest BlueTopo Tile Scheme GPKG URL from S3.
        """
        try:
            fs = fsspec.filesystem("s3", anon=True)
            pattern = f"{self.BUCKET_BASE}/_BlueTopo_Tile_Scheme/*.gpkg"
            files = fs.glob(pattern)

            if not files:
                logger.warning("No BlueTopo scheme files found via S3 glob.")
                return self.TILE_SCHEME_URL

            # Sort by name (timestamps are in filename)
            latest = sorted(files)[-1]
            filename = Path(latest).name

            logger.info(f"Resolved latest BlueTopo Scheme: {filename}")

            # Construct HTTPS URL
            return (
                "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/"
                f"BlueTopo/_BlueTopo_Tile_Scheme/{filename}"
            )
        except Exception as e:
            logger.warning(f"Failed to resolve dynamic scheme URL: {e}")
            return self.TILE_SCHEME_URL

    def _ensure_scheme_loaded(self) -> None:
        """
        Downloads and loads the Tile Scheme GPKG if not already loaded.
        """
        if self._gdf is not None:
            return

        from filelock import FileLock

        lock_path = self.scheme_path.with_suffix(".lock")

        with FileLock(lock_path):
            if not self.scheme_path.exists():
                try:
                    # Ensure directory exists (it might have been cleared runtime)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

                    # Dynamic Resolution
                    download_url = self._resolve_scheme_url()

                    logger.info(f"Downloading BlueTopo Tile Scheme to {self.scheme_path}...")
                    response = requests.get(download_url, stream=True)
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

        return list(set(results))

    def _get_tile_url_from_cache(self, tile_id: str) -> str | None:
        """Read tile_id→url from disk cache. No lock needed on read."""
        if not self.TILE_URL_CACHE_PATH.exists():
            return None
        try:
            with open(self.TILE_URL_CACHE_PATH) as f:
                data = json.load(f)
            result = data.get(tile_id)
            return str(result) if isinstance(result, str) else None
        except Exception:
            return None

    def _update_tile_url_cache(self, tile_id: str, url: str) -> None:
        """Write tile_id→url to disk cache atomically with FileLock."""
        try:
            self.TILE_URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.TILE_URL_CACHE_PATH.with_suffix(".json.lock")
            with FileLock(lock_path):
                data: dict[str, str] = {}
                if self.TILE_URL_CACHE_PATH.exists():
                    try:
                        with open(self.TILE_URL_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
                if data.get(tile_id) == url:
                    return
                data[tile_id] = url
                with open(self.TILE_URL_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to update BlueTopo tile URL cache: {e}")

    def _resolve_tile_url(self, tile_id: str) -> str | None:
        """
        Resolves the HTTPS URL for a given tile ID by checking S3.
        Results are persisted to disk so subsequent calls avoid S3 glob lookups.
        """
        # 1. Check disk cache (tile filenames include a date stamp, so S3 glob is needed
        #    on first access; thereafter the URL is stable until the tile scheme is refreshed)
        cached_url = self._get_tile_url_from_cache(tile_id)
        if cached_url:
            logger.debug(f"BlueTopo tile URL cache hit: {tile_id}")
            return cached_url

        from filelock import FileLock

        lock_path = self.cache_dir / f"url_resolve_{tile_id}.lock"

        with FileLock(str(lock_path)):
            # Double-check cache inside lock
            cached_url = self._get_tile_url_from_cache(tile_id)
            if cached_url:
                logger.debug(f"BlueTopo tile URL cache hit (post-lock): {tile_id}")
                return cached_url

            logger.debug(f"BlueTopo tile URL cache miss: {tile_id} — resolving via S3 glob")
            try:
                fs = fsspec.filesystem("s3", anon=True)
                search_pattern = f"{self.BUCKET_BASE}/{tile_id}/*.tiff"
                files = fs.glob(search_pattern)

                if not files:
                    logger.warning(f"BlueTopo: No files found in S3 for pattern: {search_pattern}")
                    return None

                source_path = files[0]
                # source_path example:
                # "noaa-ocs-nationalbathymetry-pds/BlueTopo/TileX/BlueTopo_TileX_2024.tiff"

                # Construct HTTPS URL
                # BUCKET_BASE = "noaa-ocs-nationalbathymetry-pds/BlueTopo"
                # We want: https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/BlueTopo/TileX/BlueTopo_TileX_2024.tiff

                # Split bucket from key
                parts = source_path.split("/", 1)
                if len(parts) < 2:
                    return None
                key = parts[1]

                url = f"https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/{key}"

                # 2. Persist to disk for future calls
                self._update_tile_url_cache(tile_id, url)
                logger.debug(f"BlueTopo tile URL cache created: {tile_id} -> {url}")
                return url

            except Exception as e:
                logger.error(f"BlueTopo URL Resolve Error: {e}")
                return None

    def fetch_elevation(self, lat: float, lon: float) -> float | None:
        """
        Fetches elevation from BlueTopo.
        """
        tile_id = self.resolve_tile_id(lat, lon)
        if not tile_id:
            return None

        try:
            url = self._resolve_tile_url(tile_id)
            if not url:
                return None

            # Stream with rioxarray
            # Use chunks=True (or dict) to enable dask streaming, but for single point we want
            # direct window read if possible.
            # rioxarray.open_rasterio loads lazily.
            # .sel() on a lazy array might trigger a download of the chunk.

            # Note: For single point access, standard requests/vsicurl is best.
            # rioxarray might read metadata footer first.

            with rioxarray.open_rasterio(url) as da_raw:  # type: ignore
                from typing import cast

                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = cast(xr.DataArray, da_raw.to_array().isel(variable=0))
                else:
                    da = cast(xr.DataArray, da_raw)

                # Reproject point to raster CRS
                sample_x, sample_y = lon, lat
                if da.rio.crs and da.rio.crs != "EPSG:4326":
                    from pyproj import Transformer

                    transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
                    sample_x, sample_y = transformer.transform(lon, lat)

                # Sample nearest
                val = da.sel(x=sample_x, y=sample_y, method="nearest")
                if "band" in val.dims or val.size > 1:
                    val = val.isel(band=0)

                val_item = val.values.item()

                if val_item == da.rio.nodata:
                    return None

                # VDatum
                offset = self.vdatum.get_navd88_to_lmsl_offset(lat, lon)
                return float(val_item) - offset

        except Exception as e:
            logger.error(f"BlueTopo Fetch Error: {e}", exc_info=True)
            return None

    def is_covered(self, lat: float, lon: float) -> bool:
        """
        Check if the coordinate is covered by a BlueTopo tile.
        """
        return self.resolve_tile_id(lat, lon) is not None

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        """
        Return the quality tier for the given coordinate.
        Returns QualityClass.DIRECT if covered, else UNKNOWN.
        """
        if self.is_covered(lat, lon):
            return QualityClass.DIRECT
        return QualityClass.UNKNOWN

    def load_tile_as_da(self, tile_id: str, bbox: tuple[float, float, float, float]) -> "xr.Dataset | None":
        """
        Loads the cached tile and clips to bbox (west, south, east, north).
        Uses Streaming Access (VSICURL) to avoid full download.
        Applies caching to Zarr format for faster subsequent reads.
        """
        # 1. Check Zarr Cache First (Fastest)
        zarr_name = f"{tile_id}.zarr"
        zarr_dir = self.cache_dir / "zarr"
        zarr_path = zarr_dir / zarr_name

        if zarr_path.exists():
            try:
                ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.debug(f"BlueTopo Zarr Cache Hit: {zarr_name}")
                if "elevation" in ds:
                    return ds
                return ds
            except Exception as e:
                logger.warning(f"Corrupt BlueTopo Zarr cache {zarr_path}: {e}")
                import shutil

                if zarr_path.exists():
                    shutil.rmtree(zarr_path)

        # 2. Resolve Remote URL (Cache Miss)
        http_url = self._resolve_tile_url(tile_id)
        if not http_url:
            return None

        logger.info(f"Streaming BlueTopo Asset: {http_url}")

        # 3. Stream & Cache to Zarr
        da_raw = None
        try:
            # Open Streaming
            da_raw = cast(xr.DataArray, rioxarray.open_rasterio(http_url, chunks={"x": 2048, "y": 2048}))

            # BlueTopo: Band 1=Elevation, Band 2=Uncertainty, Band 3=Contributor
            if "band" in da_raw.dims and da_raw.sizes["band"] >= 3:
                da_elev = da_raw.isel(band=0).drop_vars("band")
                da_elev.name = "elevation"
                da_src = da_raw.isel(band=2).drop_vars("band")
                da_src.name = "source_id"
                ds_to_cache = xr.Dataset({"elevation": da_elev, "source_id": da_src})
            else:
                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = da_raw.to_array().isel(variable=0)
                else:
                    da = cast(xr.DataArray, da_raw)

                if "band" in da.dims:
                    da = da.isel(band=0).drop_vars("band")
                da = cast(xr.DataArray, da)
                da.name = "elevation"
                ds_to_cache = xr.Dataset({"elevation": da})

            # Cache to Zarr
            # Lock to prevent race conditions
            from filelock import FileLock

            lock_path = zarr_path.with_suffix(".zarr.lock")

            with FileLock(lock_path):
                if zarr_path.exists():
                    return xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

                if ds_to_cache["elevation"].size > 0:
                    if "y" in ds_to_cache.dims and "x" in ds_to_cache.dims:
                        ds_to_cache = ds_to_cache.chunk({"y": 1024, "x": 1024})

                    logger.info(f"Caching BlueTopo tile to Zarr: {zarr_name}")
                    ds_to_cache.to_zarr(zarr_path, mode="w", consolidated=should_consolidate())

                    return xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

        except Exception as e:
            logger.error(f"Failed to stream/cache BlueTopo tile {tile_id}: {e}")
            return None
        finally:
            if da_raw is not None:
                with contextlib.suppress(Exception):
                    da_raw.close()

        return None

        return None

    def get_tile_id(self, lat: float, lon: float) -> str | None:
        """
        Returns the BlueTopo tile ID covering the given coordinate.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        from shapely.geometry import Point

        p = Point(lon, lat)
        # Assuming _gdf is in 4326
        matches = self._gdf[self._gdf.contains(p)]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return str(row.get("tile_id", row.get("tile")))

    def get_source_survey_id(self, lat: float, lon: float) -> str | None:
        """
        Identifies the Source Survey ID (e.g., 'H13385') at the given coordinate.
        Strategy Cascade: Embedded RAT -> Sidecar RAT -> HSMDB API.
        """
        # 1. Tile Resolution
        tile_id = self.get_tile_id(lat, lon)
        if not tile_id:
            logger.debug("No BlueTopo tile found. Fallback to API.")
            return self._resolve_from_hsmdb_api(lat, lon)

        # 2. Local File Inspection (Embedded RAT + Pixel Value)
        url = self._resolve_tile_url(tile_id)
        pixel_val: int | None = None

        if url:
            try:
                # Use rioxarray/rasterio as osgeo.gdal is unreliable in this env
                with rioxarray.open_rasterio(url, masked=True) as da:  # type: ignore
                    # Reproject point to tile CRS
                    import numpy as np
                    from pyproj import Transformer

                    # da.rio.crs must be present
                    if da.rio.crs:
                        logger.debug(f"CRS: {da.rio.crs}")

                        # Handle potential Compound CRS (Horiz + Vert) which causes failure without geoids
                        from pyproj import CRS

                        target_crs = da.rio.crs
                        try:
                            crs_obj = CRS.from_user_input(da.rio.crs)
                            if crs_obj.is_compound:
                                logger.debug("Compound CRS detected, extracting horizontal component.")
                                # Assuming first sub-crs is horizontal (standard)
                                target_crs = crs_obj.sub_crs_list[0]
                                if target_crs.to_epsg():
                                    target_crs = f"EPSG:{target_crs.to_epsg()}"
                                    logger.debug(f"Using EPSG Code: {target_crs}")
                        except Exception as e:
                            logger.warning(f"CRS parsing warning: {e}")

                        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

                        xx, yy = transformer.transform(lon, lat)
                        if np.isinf(xx) or np.isinf(yy):
                            logger.warning("Transformation returned INF! Retrying with hardcoded EPSG:26918")
                            t2 = Transformer.from_crs("EPSG:4326", "EPSG:26918", always_xy=True)
                            xx, yy = t2.transform(lon, lat)

                        logger.debug(
                            f"Rio Pixel Check: {lat},{lon} -> {xx:.2f}, {yy:.2f} in bounds {da.rio.bounds()}"
                        )

                        # Select from Band 3 (Contributor)
                        # Use nearest neighbor lookup
                        try:
                            # band=3 (1-based index in rioxarray usually)
                            val = da.sel(x=xx, y=yy, method="nearest").sel(band=3).item()
                            # Check masking
                            if val is not None and not np.isnan(val):
                                pixel_val = int(val)
                                logger.debug(f"Resolved Pixel Value: {pixel_val}")
                        except Exception:
                            # Point might be out of bounds
                            pass
            except Exception as e:
                logger.warning(f"Error inspecting local tile (rioxarray) {tile_id}: {e}")

        # 3. Sidecar Fallback
        if pixel_val is not None:
            survey_id = self._resolve_from_sidecar_rat(tile_id, pixel_val)
            if survey_id:
                logger.debug(f"Resolved from Sidecar RAT: {survey_id}")
                return survey_id

        # 4. API Fallback
        logger.info(f"Fallback to HSMDB API for {lat}, {lon}")
        return self._resolve_from_hsmdb_api(lat, lon)

    def _geo_to_pixel(self, ds: Any, lat: float, lon: float) -> tuple[int | None, int | None]:
        """
        Helper to transform lat/lon to pixel coordinates.
        """
        try:
            from osgeo import gdal, osr

            gt = ds.GetGeoTransform()
            logger.debug(f"GeoTransform: {gt}")
            proj = ds.GetProjection()
            x_geo, y_geo = lon, lat

            if proj:
                src_srs = osr.SpatialReference()
                src_srs.ImportFromEPSG(4326)
                if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs = osr.SpatialReference()
                dst_srs.ImportFromWkt(proj)
                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                # Traditional: Lon, Lat
                point = transform.TransformPoint(x_geo, y_geo)
                x_geo, y_geo = point[0], point[1]

            inv_gt = gdal.InvGeoTransform(gt)
            if inv_gt:
                x_pix = int(inv_gt[0] + x_geo * inv_gt[1] + y_geo * inv_gt[2])
                y_pix = int(inv_gt[3] + x_geo * inv_gt[4] + y_geo * inv_gt[5])

                logger.debug(f"GeoToPixel: {lat},{lon} -> {x_geo},{y_geo} (proj) -> {x_pix},{y_pix} (pix)")

                if 0 <= x_pix < ds.RasterXSize and 0 <= y_pix < ds.RasterYSize:
                    return x_pix, y_pix
        except Exception as e:
            logger.warning(f"GeoToPixel Error: {e}")
            pass
        return None, None

    def _lookup_rat(self, rat: Any, pixel_val: int) -> str | None:
        """
        Helper to query a GDAL RAT.
        """
        for i in range(rat.GetColumnCount()):
            col_name = rat.GetNameOfCol(i)
            if (
                col_name.lower() in ["survey_id", "source_survey_id", "source_id"]
                and 0 <= pixel_val < rat.GetRowCount()
            ):
                val = rat.GetValueAsString(pixel_val, i)
                if val:
                    survey_id = str(val)
                    return survey_id
        return None

    def _resolve_from_sidecar_rat(self, tile_id: str, pixel_val: int) -> str | None:
        """Look up the survey ID for a pixel value via the tile's sidecar RAT file.

        Downloads the `.aux.xml` or `.dbf` RAT linked in the BlueTopo Tile Scheme GPKG and
        caches it locally under ``sidecars/``.  A ``.failed`` sentinel is written on HTTP
        errors (typically 404 when the tile scheme's RAT_Link is stale) so the download is
        not retried on subsequent requests for the same tile.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        try:
            # Use 'tile' column (common in geopackage)
            # Check columns first
            tile_col = "tile" if "tile" in self._gdf.columns else "Tile_Name"
            matches = self._gdf[self._gdf[tile_col] == tile_id]
            if matches.empty:
                logger.warning(f"Tile {tile_id} not found in scheme index.")
                return None

            tile_row = matches.iloc[0]
            rat_link = tile_row.get("RAT_Link") or tile_row.get("rat_link")
            if not rat_link:
                return None
            # Download Sidecar
            filename = Path(rat_link).name
            sidecar_dir = self.cache_dir / "sidecars"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_file = sidecar_dir / filename
            failed_sentinel = sidecar_file.with_suffix(sidecar_file.suffix + ".failed")

            if failed_sentinel.exists():
                logger.debug(f"Skipping previously failed sidecar: {filename}")
                return None

            if not sidecar_file.exists():
                logger.info(f"Downloading Sidecar RAT: {rat_link}")
                r = requests.get(rat_link, timeout=30)
                if r.status_code == 200:
                    with open(sidecar_file, "wb") as f:
                        f.write(r.content)
                else:
                    logger.warning(
                        f"Failed to download sidecar {filename}: HTTP {r.status_code} — "
                        f"tile scheme RAT_Link may be stale. Will not retry."
                    )
                    failed_sentinel.touch()
                    return None

            # Parse based on extension
            if sidecar_file.suffix.lower() == ".xml":
                return self._parse_aux_xml_rat(sidecar_file, pixel_val)

            # Default .dbf parser (geopandas)
            df = gpd.read_file(sidecar_file)
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]

            val_col = next((c for c in df.columns if c in ["value", "oid", "id"]), None)
            survey_col = next(
                (c for c in df.columns if c in ["survey_id", "source_survey_id", "source_id"]), None
            )

            if val_col and survey_col:
                match = df[df[val_col] == pixel_val]
                if not match.empty:
                    return str(match.iloc[0][survey_col])

        except Exception as e:
            logger.warning(f"Sidecar parsing failed: {e}")

    def _parse_aux_xml_rat(self, xml_path: Path, pixel_val: int) -> str | None:
        """
        Parses GDAL PAM XML to find Survey ID.
        """
        import xml.etree.ElementTree as Et

        try:
            tree = Et.parse(xml_path)
            root = tree.getroot()

            # Look for Band 3 RAT
            rat_node = None
            for band in root.findall("PAMRasterBand"):
                if band.get("band") == "3":
                    rat_node = band.find("GDALRasterAttributeTable")
                    break

            # Fallback
            if rat_node is None:
                rat_node = root.find("GDALRasterAttributeTable")

            if rat_node is None:
                return None

            # Map Columns
            field_map = {}
            for fd in rat_node.findall("FieldDefn"):
                idx_str = fd.get("index")
                if not idx_str:
                    continue
                idx = int(idx_str)
                name_tag = fd.find("Name")
                if name_tag is not None and name_tag.text:
                    field_map[idx] = name_tag.text.lower()

            val_idx = next((i for i, n in field_map.items() if n in ["value", "id"]), None)
            survey_idx = next(
                (i for i, n in field_map.items() if n in ["survey_id", "source_survey_id", "source_id"]), None
            )

            if val_idx is None or survey_idx is None:
                return None

            # Search Rows
            for row in rat_node.findall("Row"):
                fs = row.findall("F")
                if len(fs) <= max(val_idx, survey_idx):
                    continue

                v_txt = fs[val_idx].text
                if v_txt:
                    try:
                        v = int(float(str(v_txt)))
                        if v == pixel_val:
                            s_txt = fs[survey_idx].text
                            return str(s_txt) if s_txt else None
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"XML Parsing Error: {e}")

        return None

    def _resolve_from_hsmdb_api(self, lat: float, lon: float) -> str | None:
        """
        Tertiary fallback: Query NCEI HSMDB API.
        """
        try:
            from pyproj import Transformer

            # Project to EPSG:3857
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer.transform(lon, lat)

            url = "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/nos_hydro_dynamic/MapServer/0/query"
            # Use strictly formatted JSON geometry with SR
            geo_json = f'{{"x":{x},"y":{y},"spatialReference":{{"wkid":3857}}}}'

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryPoint",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SURVEY_ID",
                "returnGeometry": "false",
                "f": "json",
            }

            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                if features:
                    # Return first match
                    val = features[0]["attributes"].get("SURVEY_ID")
                    return str(val) if val else None
        except Exception as e:
            logger.warning(f"HSMDB API Query failed: {e}")
        return None


# Register the provider
registry.register(Path(__file__).stem, NoaaBlueTopoProvider)
