"""
NOAA NCEI BAG Provider module.

This module implements the provider for accessing Bathymetric Attributed Grid (BAG) files
from NOAA's National Centers for Environmental Information (NCEI). It handles discovery,
downloading, caching, and reading of BAG files.
"""

import contextlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np  # Added numpy import
import requests  # type: ignore
import xarray as xr
from filelock import FileLock
from rioxarray.merge import merge_arrays

from ..utils.cache import concurrent_lru_cache
from ..vdatum import VDatumNoDataError, VDatumResolver
from .base import Provider, ProviderNoDataError
from .registry import registry

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@concurrent_lru_cache()
def _read_bag_cached(local_path: Path) -> xr.DataArray | None:
    """
    Cached, standalone function to read a single BAG file.
    This avoids keeping large dataset references inside class definitions.
    Cached to avoid re-opening/parsing headers for every tile.
    """
    # Define a pathway for the cached Zarr version (directory)
    # Zarr is more parallel-friendly and I/O efficient than monolithic HDF5/NetCDF
    # Store Zarr in "zarr" subdirectory
    zarr_dir = local_path.parent / "zarr"
    zarr_path = zarr_dir / local_path.name.replace(".bag", "_navd88.zarr")

    # If we have a pre-computed NAVD88 Zarr store, load strictly that.
    if zarr_path.exists():
        try:
            # chunks="auto" enables Dask lazy loading efficiently
            # consolidated=False is safer for local directory stores unless explicitly consolidated
            da_cached = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
            logger.info(f"BAG Zarr Cache Hit: {zarr_path.name}")
            return da_cached
        except Exception as e:
            logger.warning(f"Failed to load cached Zarr {zarr_path}, falling back to raw BAG: {e}")
            # If load fails, delete and regenerate
            try:
                import shutil

                if zarr_path.is_dir():
                    shutil.rmtree(zarr_path)
            except Exception:
                pass

    # Cache Miss - Acquire Lock
    from filelock import FileLock

    lock_path = zarr_path.with_suffix(".zarr.lock")
    try:
        with FileLock(lock_path):
            # Double check
            if zarr_path.exists():
                logger.info(f"BAG Zarr Cache Hit (after lock): {zarr_path.name}")
                return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

            logger.info(f"BAG Zarr Cache Miss: {zarr_path.name}")

            import rioxarray as rxr

            # Use chunks to initiate Delayed/Lazy loading via Dask.
            # Prevents OOM crashes when reading large BAG files.
            with ignore_specific_gdal_warnings("cornerPoints not consistent with resolution"):
                da_raw = rxr.open_rasterio(local_path, chunks={"x": 2048, "y": 2048}, masked=True)
                da: xr.DataArray
                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = cast(xr.DataArray, da_raw.to_array().isel(variable=0))
                else:
                    da = cast(xr.DataArray, da_raw)

            # BAGs usually have 'elevation' and 'uncertainty'.
            elev = da.isel(band=0).drop_vars("band")

            # MASK SUSPICIOUS VALUES
            # Mask standard BAG NoData values (1,000,000 or -1,000,000).
            # These can leak through if GDAL doesn't parse the BAG XML correctly.
            elev = elev.where((elev > -100000.0) & (elev < 100000.0))

            # Note: Additional dynamic filtering (e.g., configurable spike removal or NoData masking)
            # is deferred to `fetch_layer` where policy configuration (`kwargs`) is available.

            # Check for Ellipsoid vs MLLW
            filename = local_path.name
            is_ellipsoid = "_Ellipsoid_" in filename or "Ellipsoid" in filename

            # Vertical Datum Correction (to NAVD88)
            crs = elev.rio.crs
            bounds = elev.rio.bounds()  # (minx, miny, maxx, maxy)

            # Candidate points: Center, then corners, then mid-edges
            candidate_points = [
                ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2),  # Center
                (bounds[0], bounds[1]),  # MinX, MinY
                (bounds[2], bounds[3]),  # MaxX, MaxY
                (bounds[0], bounds[3]),  # MinX, MaxY
                (bounds[2], bounds[1]),  # MaxX, MinY
            ]

            offset = 0.0
            try:
                from pyproj import Transformer

                if crs:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

                    found_offset = False
                    last_error: Exception | None = None

                    for cx, cy in candidate_points:
                        try:
                            lon, lat = transformer.transform(cx, cy)

                            if is_ellipsoid:
                                # Ellipsoid -> NAVD88
                                offset = VDatumResolver.get_ellipsoid_to_navd88_offset(  # type: ignore
                                    lat, lon
                                )
                                logger.info(
                                    f"Applying Ellipsoid->NAVD88 offset of {offset:.3f}m for {filename} "
                                    f"(at {lat:.4f}, {lon:.4f})"
                                )
                            else:
                                # MLLW -> NAVD88
                                offset = VDatumResolver.get_mllw_to_navd88_offset(  # type: ignore
                                    lat, lon
                                )
                                logger.info(
                                    f"Applying MLLW->NAVD88 offset of {offset:.3f}m for {filename} "
                                    f"(at {lat:.4f}, {lon:.4f})"
                                )

                            found_offset = True
                            break  # Success!

                        except VDatumNoDataError as e:
                            # This point failed, try next
                            last_error = e
                            continue
                        except Exception as e:
                            # Other errors should probably abort or warn
                            logger.warning(f"VDatum check failed at ({lat:.4f}, {lon:.4f}): {e}")
                            last_error = e

                    if not found_offset:
                        msg = (
                            f"Could not determine VDatum offset for {filename} "
                            f"(tried {len(candidate_points)} points). defaulting to 0.0."
                        )
                        if last_error:
                            msg += f" Last error: {last_error}"
                        logger.warning(msg)

                    # Applying offset to Dask array adds a task to the graph (Lazy).
                    if found_offset:
                        elev = elev + offset

                # --- OPTIMIZATION: Persist the adjusted data to Zarr ---
                # Zarr allows chunked parallel writes and reads, ideal for this use case.
                try:
                    logger.info(f"Caching NAVD88 adjusted BAG to {zarr_path}...")

                    # Ensure we have good chunks for Zarr (e.g. 1024x1024)
                    # This balances number of files vs I/O size
                    if "x" in elev.dims and "y" in elev.dims:
                        elev = elev.chunk({"y": 1024, "x": 1024})

                    # compute() happens during the write.
                    # mode='w' overwrites if exists (safe here as we checked existence above)
                    elev.to_zarr(zarr_path, mode="w", consolidated=True)
                    logger.info(f"BAG Zarr Cache Created: {zarr_path.name}")

                    # Re-open from the fresh Zarr to return a consistent Dask-backed object
                    elev = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                except Exception as e:
                    logger.error(f"Failed to write cached Zarr: {e}")
                # ---------------------------------------------------------

            except Exception as e:
                logger.warning(f"Failed to apply VDatum correction: {e}")

            elev.attrs["survey_source"] = filename
            return cast(xr.DataArray, elev)

    except Exception as e:
        logger.error(f"Error reading BAG {local_path}: {e}")
        return None


def clean_data_deviation(da: xr.DataArray, threshold: float = 50.0) -> xr.DataArray:
    """
    Apply a 'Deviation from Median' filter to remove spikes.
    Calculates median of 3x3 neighborhood. If abs(pixel - median) > threshold, mask as NaN.

    Args:
        da: Input DataArray (elevation)
        threshold: Max allowed deviation from local median (meters)

    Returns:
        xr.DataArray: Filtered data
    """
    try:
        from scipy.ndimage import median_filter

        def _filter_block(block: Any) -> Any:
            if block.size == 0:
                return block

            # Apply standard median filter (fast, but NaN propagates)
            med = median_filter(block, size=3)
            diff = np.abs(block - med)

            if not np.issubdtype(block.dtype, np.floating):
                block = block.astype(np.float32)

            # Mask spikes exceeding threshold with NaN
            return np.where(diff > threshold, np.nan, block)

        if da.chunks is not None:
            # Apply via dask map_overlap for chunked processing
            cleaned_dask = da.data.map_overlap(
                _filter_block,
                depth=1,
                boundary="reflect",
                dtype=da.dtype,
            )

            # Return new DataArray with same coords
            return xr.DataArray(cleaned_dask, coords=da.coords, dims=da.dims, attrs=da.attrs)
        else:
            # Numpy array - Direct application
            cleaned_np = _filter_block(da.values)
            return xr.DataArray(cleaned_np, coords=da.coords, dims=da.dims, attrs=da.attrs)

    except Exception as e:
        logger.warning(f"Failed to apply cleaning filter: {e}")
        return da


@contextmanager
def ignore_specific_gdal_warnings(message_substring: str) -> Iterator[None]:
    """
    Context manager to suppress specific GDAL/rasterio warnings that are known to be benign.
    e.g. "CPLE_AppDefined in cornerPoints not consistent with resolution"
    """
    gdal_logger = logging.getLogger("rasterio._env")

    class Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return message_substring not in record.getMessage()

    f = Filter()
    gdal_logger.addFilter(f)
    try:
        yield
    finally:
        gdal_logger.removeFilter(f)


class BAGDiscovery:
    """
    Discovers BAG (Bathymetric Attributed Grid) files via NOAA NCEI API so we can
    bypass generalized BlueTopo tiles and fetch raw 50cm sonar data.
    """

    # NCEI Hydrodynamic MapServer
    QUERY_URL = (
        "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/nos_hydro_dynamic/MapServer/0/query"
    )

    # Persistent Cache for Redirects (HTML Landing Page -> .bag URL)
    # Stored in ~/.cache/topobathysim/metadata/ncei_bag_redirects.json
    REDIRECT_CACHE_PATH = Path("~/.cache/topobathysim/metadata/ncei_bag_redirects.json").expanduser()

    # Persistent Cache for BBox -> BAG URL list discovery results
    # Stored in ~/.cache/topobathysim/ncei_bag/discovery_cache.json
    # Format: {"west_south_east_north": ["https://...bag", ...], ...}
    # Invalidate to discover newly published multibeam surveys from NCEI.
    # See: manage_discovery_cache.py --invalidate ncei_bag
    DISCOVERY_CACHE_PATH = Path("~/.cache/topobathysim/ncei_bag/discovery_cache.json").expanduser()

    @classmethod
    def _read_bag_cached(cls, local_path: Path) -> xr.DataArray | None:
        """
        Reads BAG using h5py (or rasterio) and converts to xarray with NAVD88 correction.
        Cached to avoid re-opening/parsing headers for every tile.
        """
        return cast("xr.DataArray | None", _read_bag_cached(local_path))

    @classmethod
    def _get_redirect_from_cache(cls, download_url: str) -> list[str] | None:
        """Read from the JSON cache if available."""
        if not cls.REDIRECT_CACHE_PATH.exists():
            return None

        try:
            # Simple read without locking (locking on write is critical)
            with open(cls.REDIRECT_CACHE_PATH) as f:
                data = json.load(f)
                result = data.get(download_url)
                if isinstance(result, str):
                    return [result]
                return cast(list[str] | None, result)
        except Exception:
            return None

    @classmethod
    def _update_redirect_cache(cls, download_url: str, bag_urls: list[str]) -> None:
        """Update the JSON cache with a new mapping."""
        try:
            cls.REDIRECT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Use lock file alongside the json
            lock_path = cls.REDIRECT_CACHE_PATH.parent / (cls.REDIRECT_CACHE_PATH.name + ".lock")

            with FileLock(lock_path):
                data = {}
                if cls.REDIRECT_CACHE_PATH.exists():
                    try:
                        with open(cls.REDIRECT_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}

                # Check if already added
                if download_url in data and data[download_url] == bag_urls:
                    return

                data[download_url] = bag_urls

                # Write back sorted
                with open(cls.REDIRECT_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")  # EOF newline

        except Exception as e:
            logger.warning(f"Failed to update redirect cache: {e}")

    @classmethod
    def _get_from_discovery_cache(cls, bbox_key: str) -> list[str] | None:
        """
        Read bbox→BAG URL list from the disk discovery cache.

        Lock-free on read; the write side uses ``FileLock`` for safety.
        Returns ``None`` on a cache miss so the caller can fall back to the
        network query and then populate the cache via
        :meth:`_update_discovery_cache`.

        To force a fresh lookup (e.g. after NCEI publishes new surveys), delete
        ``DISCOVERY_CACHE_PATH`` or run
        ``manage_discovery_cache.py --invalidate ncei_bag``.
        """
        if not cls.DISCOVERY_CACHE_PATH.exists():
            return None
        try:
            with open(cls.DISCOVERY_CACHE_PATH) as f:
                data = json.load(f)
            result = data.get(bbox_key)
            return list(result) if isinstance(result, list) else None
        except Exception:
            return None

    @classmethod
    def _update_discovery_cache(cls, bbox_key: str, urls: list[str]) -> None:
        """
        Persist a bbox→BAG URL list to the disk discovery cache atomically.

        Uses ``FileLock`` to prevent concurrent writes from multiple worker
        processes from corrupting the JSON file.  Empty result lists are also
        stored so that known-empty bounding boxes are not re-queried.
        """
        try:
            cls.DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            lock_path = cls.DISCOVERY_CACHE_PATH.parent / (cls.DISCOVERY_CACHE_PATH.name + ".lock")
            with FileLock(lock_path):
                data: dict[str, list[str]] = {}
                if cls.DISCOVERY_CACHE_PATH.exists():
                    try:
                        with open(cls.DISCOVERY_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
                if bbox_key in data and data[bbox_key] == urls:
                    return
                data[bbox_key] = urls
                with open(cls.DISCOVERY_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to update BAG discovery cache: {e}")

    @classmethod
    def _scrape_landing_page(cls, download_url: str) -> list[str]:
        """Helper to scrape .bag URL zfrom HTML landing page."""
        # 1. Check Cache
        cached_urls = cls._get_redirect_from_cache(download_url)
        if cached_urls:
            logger.debug(f"Redirect Cache Hit: {download_url} -> {cached_urls}")
            return cached_urls

        try:
            logger.debug(f"Scraping landing page: {download_url}")
            headers = {"User-Agent": USER_AGENT}
            r_scrape = requests.get(download_url, headers=headers, timeout=10)
            r_scrape.raise_for_status()
            import re

            # Simple regex for .bag file, allowing for potential whitespace or newlines
            # Find ALL .bag links (Ellipsoid + MLLW)
            # href="..." or href='...'
            matches = re.findall(r'href=["\']?([^"\'>\s]+\.bag)["\']?', r_scrape.text, re.IGNORECASE)

            bag_links: list[str] = []
            for m in matches:
                # Cleanup logic
                link = m
                if link.startswith("//"):
                    link = "https:" + link
                elif not link.startswith("http"):
                    from urllib.parse import urljoin

                    link = urljoin(download_url, link)
                bag_links.append(link)

            final_urls: list[str] = []

            # Prioritize MLLW (Tidal Datum usually matches user expectation)
            # Deprioritize Ellipsoid because we lack robust Geoid separation grids.
            mllw_links = [L for L in bag_links if "_MLLW_" in L or "MLLW" in L]
            ellip_links = [L for L in bag_links if "_Ellipsoid_" in L or "Ellipsoid" in L]

            if mllw_links:
                logger.info(f"Preferred MLLW BAGs found: {len(mllw_links)} files")
                final_urls = sorted(list(set(mllw_links)))
            elif ellip_links:
                logger.info(
                    "Selecting Ellipsoid BAGs (Fallback, may require large Geoid offset): "
                    f"{len(ellip_links)} files"
                )
                final_urls = sorted(list(set(ellip_links)))
            elif bag_links:
                final_urls = sorted(list(set(bag_links)))

            if final_urls:
                # 2. Update Cache
                cls._update_redirect_cache(download_url, final_urls)
                return final_urls

        except Exception as e:
            logger.warning(f"Scraping failed for {download_url}: {e}")
        return []

    @classmethod
    @concurrent_lru_cache()
    def find_bag_by_survey_id(cls, survey_id: str) -> list[str]:
        """
        Query NCEI API for a specific Survey ID (e.g., 'H13385') to get the BAG download URL(s).
        """
        headers = {"User-Agent": USER_AGENT}
        # Try to clean Survey ID (e.g. H13385_MB_... -> H13385)
        clean_id = survey_id
        import re

        # Look for standard pattern: 1 letter + 5 digits
        match = re.search(r"([A-Z]\d{5})", survey_id)
        if match:
            clean_id = match.group(1)
            if clean_id != survey_id:
                logger.info(f"Cleaned Survey ID: {survey_id} -> {clean_id}")

        params = {
            "text": clean_id,
            "outFields": "SURVEY_ID,DOWNLOAD_URL",
            "returnGeometry": "false",
            "f": "json",
        }

        try:
            logger.info(f"Querying NCEI for BAG Survey: {clean_id}")
            resp = requests.get(cls.QUERY_URL, params=params, headers=headers, timeout=10)
            logger.info(f"NCEI API Request URL: {resp.url}")
            resp.raise_for_status()
            data = resp.json()

            if "features" in data and len(data["features"]) > 0:
                attr = data["features"][0]["attributes"]
                download_url = attr.get("DOWNLOAD_URL")

                if download_url and download_url.lower().endswith(".bag"):
                    return [str(download_url)]

                if download_url and download_url.lower().endswith(".html"):
                    return cls._scrape_landing_page(download_url)

                # Default return
                return [str(download_url)] if download_url else []
            else:
                logger.warning(f"No features in response: {data}")

            logger.warning(f"No NCEI record found for Survey ID: {survey_id}")
            return []

        except Exception as e:
            logger.error(f"BAG Discovery Field: {e}")
            return []

    @classmethod
    @concurrent_lru_cache()
    def find_bag_by_location(cls, lat: float, lon: float) -> list[str]:
        """
        Query NCEI API by location (spatial intersection).
        Useful when Survey ID resolving fails or matches a dataset not in the NCEI ID index.
        """
        try:
            from pyproj import Transformer

            # Project to Web Mercator (EPSG:3857) as the service uses it
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer.transform(lon, lat)

            headers = {"User-Agent": USER_AGENT}

            # Query Layer 0 (Surveys with BAGs) using geometry intersection
            # Use JSON geometry for robustness
            geo_json = f'{{"x":{x},"y":{y},"spatialReference":{{"wkid":3857}}}}'

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryPoint",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SURVEY_ID,DOWNLOAD_URL",
                "returnGeometry": "false",
                "f": "json",
            }

            logger.info(f"Querying NCEI by Location: {lat}, {lon} -> {x:.1f}, {y:.1f}")
            resp = requests.get(cls.QUERY_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "features" in data and len(data["features"]) > 0:
                attr = data["features"][0]["attributes"]
                survey_id = attr.get("SURVEY_ID")
                download_url = attr.get("DOWNLOAD_URL")

                logger.info(f"Spatial Query found Survey: {survey_id}")

                if download_url:
                    if download_url.lower().endswith(".bag"):
                        return [str(download_url)]
                    if download_url.lower().endswith(".html"):
                        return cls._scrape_landing_page(download_url)
            else:
                logger.info(f"Spatial Query - No features found (Standard Fallback). Response: {data}")
        except Exception as e:
            logger.warning(f"Spatial query failed: {e}")

        return []

    @classmethod
    def find_bags_by_bbox(
        cls, west: float, south: float, east: float, north: float, crs: str = "EPSG:4326"
    ) -> list[str]:
        """
        Queries NCEI for BAGs intersecting the bounding box.
        Returns list of download URLs.
        Results are persisted to disk so subsequent calls (including across server restarts)
        return immediately without a network round-trip.
        """
        # OPTIMIZATION: Snap to a 0.01 degree grid to ensure high cache hit rate
        # regardless of small floating point or halo variations.
        grid = 0.01
        swest = np.floor(west / grid) * grid
        ssouth = np.floor(south / grid) * grid
        seast = np.ceil(east / grid) * grid
        snorth = np.ceil(north / grid) * grid

        crs_tag = crs.replace(":", "_")
        bbox_key = f"{round(swest, 4)}_{round(ssouth, 4)}_{round(seast, 4)}_{round(snorth, 4)}_{crs_tag}"

        # 1. Check disk cache (survives server restarts and process recycling)
        cached = cls._get_from_discovery_cache(bbox_key)
        if cached is not None:
            logger.debug(f"BAG Discovery Cache Hit: {bbox_key} ({len(cached)} urls)")
            return cached

        logger.debug(f"BAG Discovery Cache Miss: {bbox_key} — querying NCEI ArcGIS API")
        found_urls: list[str] = []
        try:
            from pyproj import Transformer

            # MapServer expects EPSG:3857 (Web Mercator)
            transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
            minx, miny = transformer.transform(swest, ssouth)
            maxx, maxy = transformer.transform(seast, snorth)

            headers = {"User-Agent": USER_AGENT}

            # Envelope Query
            geo_json = (
                f'{{"xmin":{minx},"ymin":{miny},"xmax":{maxx},'
                f'"ymax":{maxy},"spatialReference":{{"wkid":3857}}}}'
            )

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SURVEY_ID,DOWNLOAD_URL,SURVEY_YEAR,DATE_SURVEY_END",
                "returnGeometry": "false",
                "f": "json",
            }

            logger.info(f"Querying NCEI by BBox: {swest},{ssouth},{seast},{snorth}")
            resp = requests.get(cls.QUERY_URL, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            if "features" in data:
                import re

                # Collect valid BAGs with their sort key (Resolution, Date)
                # Primary Sort: Resolution (Ascending, so 0.5 < 1.0)
                # Secondary Sort: Date (Descending, so Newest first)
                found_bags: list[tuple[str, float, int]] = []  # (URL, Resolution, DateTimestamp)

                for feature in data["features"]:
                    attr = feature.get("attributes", {})
                    download_url = attr.get("DOWNLOAD_URL")
                    # Prefer precise end date (timestamp), fallback to year (int), fallback to 0
                    d_end = attr.get("DATE_SURVEY_END")
                    d_year = attr.get("SURVEY_YEAR")
                    sort_date = 0

                    try:
                        if d_end is not None:
                            sort_date = int(d_end)
                        elif d_year is not None:
                            sort_date = (int(d_year) - 1970) * 31536000000
                    except Exception:
                        pass

                    # Extract Resolution from URL/Filename
                    # e.g. H13384_MB_50cm_MLLW_1of1.bag
                    resolution_val = 100.0  # Default low priority
                    if download_url:
                        fname = download_url.split("/")[-1].lower()
                        # Match _50cm_ or _0.5m_
                        m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
                        m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)

                        if m_cm:
                            resolution_val = float(m_cm.group(1)) / 100.0
                        elif m_m:
                            resolution_val = float(m_m.group(1))

                    # Tuple: (URL, Res, Date)
                    if download_url:
                        final_urls = []
                        if download_url.lower().endswith(".bag"):
                            final_urls = [str(download_url)]
                        elif download_url.lower().endswith(".html"):
                            scraped = cls._scrape_landing_page(download_url)
                            if scraped:
                                final_urls = scraped

                        for url in final_urls:
                            # Re-check resolution on final URLs (in case scraping found variations)
                            fname = url.split("/")[-1].lower()
                            res_final = resolution_val
                            # If scraped, re-calculate resolution from actual filename
                            m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
                            m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)
                            if m_cm:
                                res_final = float(m_cm.group(1)) / 100.0
                            elif m_m:
                                res_final = float(m_m.group(1))

                            found_bags.append((url, res_final, sort_date))
                            logger.debug(f"Discovered BAG: {fname} | Res: {res_final}m | Date: {sort_date}")

                # Sort:
                # 1. Resolution ASC (Smallest = Best)
                # 2. Date DESC (Largest = Newest = Best) -> Use negative
                found_bags.sort(key=lambda x: (x[1], -x[2]))

                # Extract just the URLs for compatibility
                # Deduplicate while preserving order
                seen = set()
                found_urls = []
                for url, _, _ in found_bags:
                    if url not in seen:
                        found_urls.append(url)
                        seen.add(url)

            logger.info(f"Found {len(found_urls)} BAGs in BBox (Sorted by Date).")

        except Exception as e:
            logger.warning(f"BBox query failed: {e}")

        # 2. Persist to disk (even empty results, to avoid re-querying known-empty bboxes)
        cls._update_discovery_cache(bbox_key, found_urls)
        logger.debug(f"BAG Discovery Cache Created: {bbox_key} ({len(found_urls)} urls persisted)")
        return found_urls


class BAGProvider(Provider):
    """
    Manages downloading, caching, and reading of NOAA BAG files.
    """

    # Singleton instance to avoid re-initializing discovery caches
    _singleton: ClassVar["BAGProvider | None"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "BAGProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(BAGProvider, cls._singleton)

    def __init__(self, cache_dir: str = "~/.cache/topobathysim") -> None:
        """
        Initialize the BAG provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if getattr(self, "_initialized", False):
            return

        self.cache_dir = Path(cache_dir).expanduser() / "ncei_bag"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir
        self.vdatum = VDatumResolver()
        self._initialized = True

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetches and merges BAG files intersecting the bounding box.
        """
        west, south, east, north = bbox

        # 1. Discover BAGs
        urls = BAGDiscovery.find_bags_by_bbox(west, south, east, north, crs=crs)
        if not urls:
            raise ProviderNoDataError(f"No BAG files found for bbox {bbox}")

        # Note: BAGDiscovery now returns URLs sorted Best -> Worst (Resolution, then Date).
        # rioxarray.merge_arrays prioritizes the FIRST array (Top-Down), so this order is correct.

        logger.info(f"BAG fetch: Found {len(urls)} files to process.")

        # 2. Fetch/Load Each
        project_layers = []
        provenance_dict: dict[int, dict[str, str]] = {}

        for url in urls:
            try:
                # Load (Lazy/Zarr if available)
                da = self.fetch_bag(survey_id="unknown_bbox_fetch", download_url=url)
                if da is None:
                    logger.warning(f"BAG fetch: Failed to load data from {url}")
                    continue

                # OPTIMIZATION: Clip EARLY (before reprojection/merge)
                # This drastically reduces memory usage for large surveys
                try:
                    # Clip using EPSG:4326 bounds. rioxarray handles transformations.
                    da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326")
                except Exception as e:
                    # rioxarray.exceptions.NoDataInBounds (or similar) - Skip this tile
                    logger.debug(f"BAG tile empty after clip (url={url}): {e}")
                    continue

                if da.size == 0:
                    continue

                # --- CLEANING / FILTERING ---
                filter_cfg = kwargs.get("filter", {})

                # 1. Deviation/Spike Removal
                max_dev = filter_cfg.get("max_depth_change") or filter_cfg.get("max_deviation")

                if max_dev:
                    logger.info(f"Applying BAG Deviation Filter (Threshold={max_dev}m) to {url}")
                    da = clean_data_deviation(da, threshold=float(max_dev))

                # Reproject individual chunk to target CRS
                if crs and da.rio.crs and da.rio.crs != crs:
                    try:
                        # Optional: Pass resolution if provided to enforce downsampling early
                        reproj_knn = {}
                        # If target is projected (meters) and we have input_res (meters)
                        if resolution and "EPSG:4326" not in crs:
                            reproj_knn["resolution"] = resolution

                        da = da.rio.reproject(crs, **reproj_knn)
                    except Exception as e:
                        logger.warning(f"Reprojection failed for BAG segment: {e}")
                        continue

                import hashlib

                filename = url.split("/")[-1].replace(".bag", "")
                project_uid = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 100000 + 40000
                provenance_dict[project_uid] = {
                    "name": filename,
                    "provider": "ncei_bag",
                }

                da.name = "elevation"
                p_source = xr.where(da.notnull(), project_uid, 0).astype(np.uint32)
                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                p_ds = xr.Dataset({"elevation": da, "source_id": p_source})
                if da.rio.crs:
                    p_ds.rio.write_crs(da.rio.crs, inplace=True)
                p_ds.rio.write_transform(da.rio.transform(), inplace=True)

                project_layers.append(p_ds)

            except Exception as e:
                logger.warning(f"Failed to process BAG {url}: {e}")

        if not project_layers:
            # It is possible all BAGs were clipped out or failed
            # This is not necessarily an error, just no coverage in this detailed window
            # Return empty or raise?
            # Runtime expects an array. Raise KeyError to trigger 'continue' in runtime loop.
            raise ProviderNoDataError(f"No BAG data intersects bbox {bbox} after clipping")

        # 3. Merge
        if len(project_layers) == 1:
            merged_ds = project_layers[0]
        else:
            try:
                # merge_arrays puts FIRST element on TOP (Priority).
                # URLs are sorted Best -> Worst by discovery.
                elevs = [ds["elevation"] for ds in project_layers]
                sources = [ds["source_id"] for ds in project_layers]

                final_elev = merge_arrays(elevs)
                final_src = merge_arrays(sources)

                # Defensive: Drop duplicate indices from merge
                if not final_elev.indexes["x"].is_unique:
                    final_elev = final_elev.drop_duplicates(dim="x")
                if not final_elev.indexes["y"].is_unique:
                    final_elev = final_elev.drop_duplicates(dim="y")

                if not final_src.indexes["x"].is_unique:
                    final_src = final_src.drop_duplicates(dim="x")
                if not final_src.indexes["y"].is_unique:
                    final_src = final_src.drop_duplicates(dim="y")

                merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
            except Exception as e:
                logger.error(f"Failed to merge BAGs: {e}")
                merged_ds = project_layers[0]

        # 4. Final Clip (Cleanup)
        # Ensure exact bounds (reprojection might have introduced slight over-run)
        with contextlib.suppress(Exception):
            final_elev = merged_ds["elevation"].rio.clip_box(
                minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326"
            )
            final_src = merged_ds["source_id"].rio.clip_box(
                minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326"
            )
            merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})

        merged_ds.attrs["provenance_dict"] = provenance_dict
        logger.debug(f"Found NCEI BAG Coverage with provenance {provenance_dict}")
        return cast(xr.Dataset, merged_ds)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the BAG provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA NCEI BAG (Bathymetric Attributed Grid)",
            "citation": "NOAA National Centers for Environmental Information.",
            "resolution": "High (Variable, typically 0.5m - 4m)",
            "url": "https://www.ncei.noaa.gov/products/bathymetry",
        }

    def fetch_bag(self, survey_id: str, download_url: str | list[str] | None = None) -> xr.DataArray | None:
        """
        Fetches and reads a BAG file for a given Survey ID.
        Auto-discovers URL if not provided. Supports multiple BAG files for a single survey.
        """
        if not download_url:
            download_url = BAGDiscovery.find_bag_by_survey_id(survey_id)

        if not download_url:
            return None

        # Normalize to list
        urls = [download_url] if isinstance(download_url, str) else download_url

        loaded_arrays = []

        for url in urls:
            local_path = self._ensure_downloaded(url)
            if local_path:
                da = self._read_bag(local_path)
                if da is not None:
                    loaded_arrays.append(da)

        if not loaded_arrays:
            return None

        if len(loaded_arrays) == 1:
            return loaded_arrays[0]

        # Merge if multiple
        try:
            from rioxarray.merge import merge_arrays

            # Sort by resolution (finest first) to preserve detail
            # abs(res[0]) is pixel width. Smallest pixel width = highest resolution.
            loaded_arrays.sort(key=lambda da: abs(da.rio.resolution()[0]))

            logger.info(f"Merging {len(loaded_arrays)} BAG segments for {survey_id}")
            # merge_arrays returns a DataArray by default if inputs are DataArrays
            merged = merge_arrays(loaded_arrays)
            merged.attrs["survey_source"] = f"{survey_id} (Merged {len(loaded_arrays)})"
            return cast(xr.DataArray, merged)
        except Exception as e:
            logger.error(f"Failed to merge BAGs for {survey_id}: {e}")
            # Return the first one as best effort
            return loaded_arrays[0]

    def _ensure_downloaded(self, url: str) -> Path | None:
        """Helper to download a single BAG file if missing."""
        filename = url.split("/")[-1]
        local_path = self.cache_dir / filename

        # 1. Download if missing
        if not local_path.exists():
            import fcntl

            lock_path = self.cache_dir / f"{filename}.lock"
            temp_path = self.cache_dir / f".tmp_{filename}"

            try:
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    if local_path.exists():
                        pass  # Double check inside lock
                    else:
                        logger.info(f"Downloading BAG: {url}")
                        # Use requests for progress? Or urllib/shutil for simplicity
                        # Since BAGs are large, streaming via requests is better
                        with requests.get(url, stream=True) as r:
                            r.raise_for_status()
                            total_size = int(r.headers.get("content-length", 0))
                            with open(temp_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=32768):
                                    f.write(chunk)

                            # Verify size if content-length was provided
                            if total_size > 0 and temp_path.stat().st_size != total_size:
                                raise OSError(
                                    f"Incomplete download: {temp_path.stat().st_size}/{total_size} bytes"
                                )

                        Path(temp_path).rename(local_path)

                    # Cleanup Lock if successful
                    lock_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to download BAG {url}: {e}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                return None

        return local_path

    def _read_bag(self, local_path: Path) -> xr.DataArray | None:
        """wrapper to call standalone cached function."""
        # Use standalone function to leverage python LRU cache safely outside class methods.
        return cast("xr.DataArray | None", _read_bag_cached(local_path))


# Register the provider
registry.register(Path(__file__).stem, BAGProvider)
