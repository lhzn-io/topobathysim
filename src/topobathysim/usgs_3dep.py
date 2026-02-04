import logging

# Caching setup
from functools import lru_cache
from pathlib import Path

import planetary_computer
import requests  # type: ignore
import rioxarray
import xarray as xr
from pystac_client import Client

# Share cache directory with Lidar for STAC queries (Not used for query caching anymore)
# stac_cache_dir = Path.home() / ".cache" / "topobathysim" / "stac_queries"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _query_land_collection(bbox: tuple[float, float, float, float], collection_id: str) -> list[dict] | None:
    """
    Cached STAC query for Land Collections (3DEP, NASADEM).
    Returns list of item dicts (href only) to avoid pickling.
    """
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

    try:
        logger.debug(f"Usgs3DepProvider querying (Cached) {collection_id} for {bbox}")
        catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
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
        logger.warning(f"Error fetching {collection_id}: {e}")
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

    def _download_and_cache(self, url: str) -> Path | None:
        import fcntl
        import hashlib

        import requests

        # Create a safe filename from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = ".tif"
        if ".tiff" in url:
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

    def fetch_dem(
        self, bounds: tuple[float, float, float, float], target_crs: str = "EPSG:4326"
    ) -> xr.DataArray | None:
        """
        Fetches best available land DEM for the bbox.
        1. 3DEP (US)
        2. Copernicus/NASADEM (Global)
        """
        # 1. Try USGS 3DEP
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

            # 1. Offline Mode / Manifest Lookup
            if self.offline_mode:
                logger.debug(f"Offline Mode: Checking Manifest for {collection_id} in {bounds}")
                manifest_items = self.manifest.find_items(collection_id, bounds)
                if manifest_items:
                    # Convert manifest dicts back to minimal item structure for loop
                    items = [{"href": m["href"]} for m in manifest_items]
                    logger.info(f"Offline Manifest found {len(items)} items.")

            # 2. Online Mode (if not found in manifest or not offline)
            if not items and not self.offline_mode:
                # Use cached query
                bbox_tuple = tuple(bounds)
                items = _query_land_collection(bbox_tuple, collection_id)

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

            das = []
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
                            _query_land_collection.cache_clear()
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
                        with rioxarray.open_rasterio(local_path) as da:
                            if "band" in da.dims:
                                da = da.isel(band=0).drop_vars("band")
                            # Load data into memory
                            da.load()
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
