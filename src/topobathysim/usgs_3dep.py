import fcntl
import logging

# Caching setup
from functools import lru_cache
from pathlib import Path
from typing import cast

import planetary_computer
import requests  # type: ignore
import rioxarray
import xarray as xr
from pystac_client import Client

# Share cache directory with Lidar for STAC queries (Not used for query caching anymore)
# stac_cache_dir = Path.home() / ".cache" / "topobathysim" / "stac_queries"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _query_stac_cached(
    bbox: tuple[float, float, float, float],
    collection_id: str,
    cache_dir_str: str,
) -> list[dict] | None:
    """
    Cached STAC query for Land Collections (3DEP, NASADEM).
    Returns list of item dicts (href only) to avoid pickling.
    """
    import random
    import time
    from pathlib import Path

    cache_dir = Path(cache_dir_str)

    # Global Concurrency Lock (One STAC query at a time across all processes)
    # This prevents 4-8 workers from bursting the API simultaneously on cold start.
    lock_file_path = cache_dir / "stac_query.lock"

    logger.debug(f"Querying STAC collection {collection_id} for bbox={bbox}")
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

    max_retries = 4

    with open(lock_file_path, "w") as lock_file:
        # Blocking Exclusive Lock - Only one worker can query at a time
        # Downloads are not locked, just the metadata query/signing.
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(
                        f"Usgs3DepProvider querying (Cached) {collection_id} "
                        f"for {bbox} (Attempt {attempt + 1})"
                    )

                    # We MUST sign inplace to get SAS tokens for blob access
                    # Note: Client.open might make a network call
                    catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)

                    # Search
                    search = catalog.search(collections=[collection_id], bbox=bbox, limit=10)
                    items = list(search.items())

                    if not items:
                        return None

                    results = []
                    for item in items:
                        # Extract asset href
                        asset_key = "data"
                        if asset_key not in item.assets and "elevation" in item.assets:
                            asset_key = "elevation"

                        if asset_key in item.assets:
                            results.append(
                                {
                                    "href": item.assets[asset_key].href,
                                    "bbox": item.bbox,
                                    "properties": item.properties,
                                }
                            )

                    return results

                except Exception as e:
                    # Check if it's a timeout or throttling error
                    is_last_attempt = attempt == max_retries
                    log_level = logging.WARNING
                    if is_last_attempt:
                        log_level = logging.ERROR

                    logger.log(
                        log_level,
                        f"Error fetching {collection_id} (Attempt {attempt + 1}): {e}",
                    )

                    if is_last_attempt:
                        return None

                    # Exponential Backoff with Jitter: 1s, 2s, 4s, 8s...
                    sleep_time = (2**attempt) + random.uniform(0.1, 1.0)
                    logger.debug(f"Sleeping {sleep_time:.2f}s before retry...")
                    time.sleep(sleep_time)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return None


class Usgs3DepProvider:
    """
    Provider for Mid-Resolution Land Topography.
    Tier 2: USGS 3DEP (10m)
    Tier 3: NASADEM / Copernicus (30m)
    """

    def __init__(self, cache_dir: str = "~/.cache/topobathysim", offline_mode: bool = False):
        self.cache_dir = Path(cache_dir).expanduser() / "land"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode

        # Manifest for Offline Lookup
        from .manifest import OfflineManifest

        self.manifest = OfflineManifest(self.cache_dir)

    def _query_land_collection(
        self,
        bbox: tuple[float, float, float, float],
        collection_id: str,
        datetime_range: str | None = None,
    ) -> list[dict] | None:
        return _query_stac_cached(bbox, collection_id, str(self.cache_dir))

    def _download_and_cache(self, url: str) -> Path | None:
        import hashlib

        import requests

        # Create a safe filename from URL
        # Strip query parameters (SAS tokens) for stable hashing
        # URL: https://.../blob.core.windows.net/.../file.tif?sv=...
        url_base = url.split("?")[0]
        logger.debug(f"Hashing URL Base: {url_base} (Original: {url[:50]}...)")
        url_hash = hashlib.md5(url_base.encode()).hexdigest()
        ext = ".tif"
        if ".tiff" in url_base:
            ext = ".tiff"

        filename = f"{url_hash}{ext}"
        local_path = self.cache_dir / filename
        lock_path = self.cache_dir / f"{filename}.lock"
        temp_path = self.cache_dir / f".tmp_{filename}"

        # 1. Fast Path
        if local_path.exists():
            return local_path

        # 2. Acquire Lock
        # Using a separate lock file avoids opening the target file in write mode before we are ready
        try:
            with open(lock_path, "w") as lock_file:
                # Exclusive lock (blocking)
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # 3. Double-Check
                    if local_path.exists():
                        return local_path

                    logger.info(f"Downloading land asset to {local_path}...")

                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(temp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    # Atomic move
                    temp_path.rename(local_path)
                    logger.debug("Download complete.")
                    return local_path

                finally:
                    # Release lock
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        except Exception as e:
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 403:
                raise e  # Propagate 403 for upper layers to handle token refresh

            logger.warning(f"Download or Lock failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None
        finally:
            # Clean up lock file?
            # Deleting lock file allows race condition if another process is waiting on it.
            # Safe to leave lock files or delete only if we held it exclusively?
            # Standard practice: usually leave lock files or use /tmp.
            # We'll leave it to be safe and simple.
            pass

    def fetch_dem(self, bounds: tuple[float, float, float, float]) -> xr.DataArray | None:
        logger.debug(f"fetch_dem called with bounds={bounds}")
        """
        Fetches best available land DEM for the bbox.
        Search priority: 3DEP (US), then Copernicus (Global), then NASADEM (Global)
        """
        # 1. 3DEP Seamless (10m) - Best for US
        da = self._fetch_collection(bounds, "3dep-seamless")
        if da is not None:
            logger.debug("Found USGS 3DEP Coverage")
            return da

        # 2. Try Copernicus DEM (GLO-30)
        da = self._fetch_collection(bounds, "cop-dem-glo-30")
        if da is not None:
            logger.debug("Found Copernicus DEM Coverage")
            return da

        # 3. Try NASADEM
        da = self._fetch_collection(bounds, "nasadem")
        if da is not None:
            logger.debug("Found NASADEM Coverage")
            return da

        return None

    def _fetch_collection(self, bounds: tuple, collection_id: str) -> xr.DataArray | None:
        try:
            items = None

            # 1. Always Check Manifest First (Local STAC Cache)
            # This allows us to skip throttled API calls if we already know the assets for this bbox.
            # We trust our local cache of "resolution" (mapping from bbox -> assets)
            logger.debug(f"Checking Local Manifest for {collection_id} in {bounds}")
            manifest_items = self.manifest.find_items(collection_id, bounds)

            if manifest_items:
                # Convert manifest items to the list structure used below
                items = [
                    {"href": m["href"], "bbox": m["bbox"], "properties": m.get("properties")}
                    for m in manifest_items
                ]
                logger.info(
                    f"Manifest Cache Hit: Found {len(items)} items for {collection_id} (Skipping API)."
                )

            # 2. Online Mode - Only Query API if Manifest Miss AND Not Offline
            if not items and not self.offline_mode:
                # Use cached query
                bbox_tuple = tuple(bounds)
                items = self._query_land_collection(bbox_tuple, collection_id)

            if not items:
                logger.debug(f"Usgs3DepProvider found 0 items for {collection_id}")
                return None

            # Record found items to Manifest (if online)
            if not self.offline_mode:
                for item in items:
                    self.manifest.add_item(
                        collection_id=collection_id,
                        bbox=item.get("bbox", bounds),  # Fallback to query bounds if item bbox missing
                        asset_href=item["href"],
                        properties=item.get("properties"),
                    )

            logger.debug(f"Usgs3DepProvider found {len(items)} items for {collection_id}")

            das: list[xr.DataArray] = []
            for item in items:
                href = item["href"]
                logger.debug(f"Fetching Land Asset: {href}")

                # Retry loop for corruption handling & Token Expiry
                for attempt in range(2):
                    # Download to cache first
                    try:
                        local_path = self._download_and_cache(href)
                    except requests.HTTPError as e:
                        if e.response is not None and e.response.status_code == 403:
                            logger.warning(f"403 Forbidden for {href}. Clearing STAC cache and retrying...")
                            _query_stac_cached.cache_clear()
                            # Retry THIS item? We need to re-query everything actually.
                            # But we are inside a loop over OLD items.
                            # We should probably break out and restart the entire _fetch_collection logic?
                            # Or just re-query this specific call?
                            # If we clear cache, the next call to _query_land_collection will get new tokens.
                            # So we should recursively call _fetch_collection once?
                            return self._fetch_collection(bounds, collection_id)
                        raise e  # Re-raise other errors to be handled below?
                        # Actually previous logic swallowed errors in _download_and_cache.
                        # We need _download_and_cache to RAISE 403, but maybe return None for others?

                    if not local_path:
                        break  # Download failed (non-403)

                    # Open local file
                    try:
                        # Test open to check for corruption
                        # Test open to check for corruption
                        # Use chunks for lazy loading (Avoid OOM)
                        da_raw = rioxarray.open_rasterio(local_path, chunks={"x": 2048, "y": 2048})
                        if isinstance(da_raw, list):
                            da = cast(xr.DataArray, da_raw[0])
                        elif isinstance(da_raw, xr.Dataset):
                            da = da_raw.to_array().isel(variable=0)
                        else:
                            da = da_raw

                        da = cast(xr.DataArray, da)
                        if "band" in da.dims:
                            da = da.isel(band=0).drop_vars("band")

                        # Verify we can read metadata, but DO NOT load data eagerley
                        # da.load()  <-- REMOVED to prevent OOM
                        das.append(da)

                        break  # Success
                    except Exception as e:
                        logger.warning(
                            f"Failed to load cached land asset {local_path} (Attempt {attempt + 1}): {e}"
                        )
                        # Corruption likely? Delete and try again
                        if local_path.exists():
                            logger.warning(f"Deleting corrupted file: {local_path}")
                            local_path.unlink()
                        if attempt == 1:
                            logger.error(f"Permanent failure loading {local_path}")

            if not das:
                return None

            from rioxarray.merge import merge_arrays

            # Merge
            merged = merge_arrays(das)

            # Ensure CRS
            if not merged.rio.crs:
                # STAC Land collections (3DEP/COP-30) are EPSG:4326
                merged.rio.write_crs("EPSG:4326", inplace=True)

            # Mask Zeros and low-level noise as NaNs
            # (Water/NoData commonly 0, but can be noisy in 3DEP/COP-30)
            # We want BlueTopo (Bathy) to handle everything near sea level.
            merged = merged.where(merged > 0.5)

            return merged

        except Exception as e:
            logger.error(f"Error fetching {collection_id}: {e}", exc_info=True)
            return None

    def get_grid(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        target_shape: tuple[int, int] | None = None,
    ) -> xr.DataArray | None:
        """
        Unified access method for Manager compatibility.
        """
        return self.fetch_dem(bounds=(west, south, east, north))
