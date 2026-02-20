"""
USGS 3DEP Provider module.

This module implements the provider for USGS 3D Elevation Program (3DEP) data.
It queries the Microsoft Planetary Computer STAC API for '3dep-seamless', 'cop-dem-glo-30',
and 'nasadem' collections.
"""

import contextlib
import fcntl
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, cast

import planetary_computer
import rioxarray
import xarray as xr
from pystac_client import Client
from rioxarray.merge import merge_arrays

from ..manifest import OfflineManifest
from .base import Provider
from .registry import registry

logger = logging.getLogger(__name__)


class Usgs3DepProvider(Provider):
    """
    Provider for Mid-Resolution Land Topography.
    Tier 2: USGS 3DEP (10m)
    Tier 3: NASADEM / Copernicus (30m)
    """

    def __init__(self, cache_dir: str = "~/.cache/topobathysim", offline_mode: bool = False):
        """
        Initialize the 3DEP provider.

        Args:
            cache_dir: Directory to store cached data files.
            offline_mode: If True, only use locally cached data/manifests.
        """
        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "usgs_3dep"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata Cache Directory
        self.metadata_dir = base_cache / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.offline_mode = offline_mode

        # Manifest for Offline Lookup
        self.manifest = OfflineManifest(self.metadata_dir, filename="stac_manifest.json")

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Fetch USGS 3DEP (or fallback) data for the given bounding box.
        """
        bounds = bbox

        # 1. 3DEP Seamless (10m) - Best for US
        da = self._fetch_collection(bounds, "3dep-seamless")
        if da is not None:
            logger.debug("Found USGS 3DEP Coverage")
            da.name = "elevation"
            return da

        # 2. Try Copernicus DEM (GLO-30)
        da = self._fetch_collection(bounds, "cop-dem-glo-30")
        if da is not None:
            logger.debug("Found Copernicus DEM Coverage")
            da.name = "elevation"
            return da

        # 3. Try NASADEM
        da = self._fetch_collection(bounds, "nasadem")
        if da is not None:
            logger.debug("Found NASADEM Coverage")
            da.name = "elevation"
            return da

        raise KeyError(f"No USGS 3DEP/Land coverage found for bbox {bbox}")

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
        try:
            return self.fetch_layer(bbox=(west, south, east, north))
        except KeyError:
            return None

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the 3DEP provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "USGS 3DEP (Seamless)",
            "citation": "U.S. Geological Survey.",
            "resolution": "10m (1/3 arc-second)",
            "url": "https://www.usgs.gov/core-science-systems/ngp/3dep",
        }

    def _query_stac_api(
        self,
        bbox: tuple[float, float, float, float],
        collection_id: str,
    ) -> list[dict] | None:
        """
        Query STAC API for Items. Handles Negative Caching and Locking.
        """
        if self.offline_mode:
            return None

        # 1. Check Negative Cache (No Coverage)
        if self.manifest.has_no_coverage(collection_id, bbox):
            logger.debug(f"Negative Cache Hit: {collection_id} is known empty for {bbox}")
            return None

        # Global Concurrency Lock
        lock_file_path = self.cache_dir / "stac_query.lock"
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        max_retries = 4

        try:
            with open(lock_file_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Double-Check Negative Cache (in case another worker updated it while we waited)
                    if self.manifest.has_no_coverage(collection_id, bbox):
                        return None

                    for attempt in range(max_retries + 1):
                        try:
                            logger.debug(
                                f"Usgs3DepProvider querying {collection_id} "
                                f"for {bbox} (Attempt {attempt + 1})"
                            )

                            catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
                            search = catalog.search(collections=[collection_id], bbox=bbox, limit=10)
                            items = list(search.items())

                            # Record Search Result (Positive or Negative)
                            self.manifest.record_search(collection_id, bbox, len(items))

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
                            # Retry logic
                            is_last_attempt = attempt == max_retries
                            log_level = logging.WARNING if not is_last_attempt else logging.ERROR
                            logger.log(
                                log_level,
                                f"Error fetching {collection_id} (Attempt {attempt + 1}): {e}",
                            )

                            if is_last_attempt:
                                return None

                            sleep_time = (2**attempt) + random.uniform(0.1, 1.0)
                            time.sleep(sleep_time)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        finally:
            # Best-effort cleanup of lock file
            with contextlib.suppress(OSError):
                os.remove(lock_file_path)

        return None

    def _fetch_collection(self, bounds: tuple, collection_id: str) -> xr.DataArray | None:
        try:
            items = None

            # 1. Check Manifest (Positive Cache)
            # 1. Check Manifest (Positive Cache) - Logic Update:
            # We trust the Zarr cache over the manifest for DATA.
            # But we need to know WHICH items to look for.
            # The issue is: If we have Zarrs, we don't need to query API.
            # But we don't know the Zarr filenames without the HREFs.
            # So:
            # A) Query API/Manifest to get Items/HREFs.
            # B) For each Item, Check Zarr.
            # C) If Zarr exists, load it. If not, stream & save.

            logger.debug(f"Checking Local Manifest for {collection_id} in {bounds}")
            manifest_items = self.manifest.find_items(collection_id, bounds)

            if manifest_items:
                items = [
                    {"href": m["href"], "bbox": m["bbox"], "properties": m.get("properties")}
                    for m in manifest_items
                ]
                logger.info(
                    f"Manifest Cache Hit: Found {len(items)} items for {collection_id} (Skipping API)."
                )

            # 2. Query API (if Miss)
            if not items and not self.offline_mode:
                items = self._query_stac_api(bounds, collection_id)

            # Logic continues... Checks Zarr inside the loop now.

            if not items:
                logger.debug(f"Usgs3DepProvider found 0 items for {collection_id}")
                return None

            items_found: list[dict[str, Any]] = []
            das: list[xr.DataArray] = []

            for item in items:
                href = item["href"]
                max_retries = 1

                for retry in range(max_retries + 1):
                    try:
                        logger.debug(f"Streaming Land Asset: {href}")
                        # Open streaming (Lazy)
                        da_raw = rioxarray.open_rasterio(href, chunks={"x": 2048, "y": 2048})

                        if isinstance(da_raw, list):
                            da = cast(xr.DataArray, da_raw[0])
                        elif isinstance(da_raw, xr.Dataset):
                            da = da_raw.to_array().isel(variable=0)
                        else:
                            da = da_raw

                        da = cast(xr.DataArray, da)
                        if "band" in da.dims:
                            da = da.isel(band=0).drop_vars("band")

                        # --- WRITE TO ZARR CACHE ---
                        # We stream the specific item to a local Zarr to avoid re-streaming
                        if True:  # Was try/except, but we handle errors internally now
                            # CRITICAL: Clip to requested bounds BEFORE caching
                            # Otherwise we download the entire 1x1 degree COG (GBs of data)
                            # for a tiny request.
                            try:
                                da = da.rio.clip_box(*bounds)
                            except Exception:
                                # If clip fails (no overlap?), this item is useless for the requested bbox.
                                # Skip it to avoid downloading huge irrelevant files.
                                logger.debug(f"Clip FAILED for {href} with bounds {bounds}. Skipping item.")
                                continue

                            # Check if the result is actually reduced
                            if da.size > 100_000_000:  # Arbitrary large size
                                logger.warning(
                                    f"3DEP Asset {href} is too large {da.size} after clip "
                                    "(or clip failed). Skipping Zarr cache to avoid stall."
                                )
                                # We treat this as a failure to cache.

                            # 1. Provide Feedback: Remove stale item
                            self.manifest.remove_item_by_href(href)

                            # 2. Refresh: Force Query API
                            # We query the specific item's bbox to get a fresh token for it
                            item_bbox = tuple(item.get("bbox", bounds))

                            # ... match context ...

                    except Exception as e:
                        err_msg = str(e)
                        is_403 = "403" in err_msg or "Forbidden" in err_msg or "Access Denied" in err_msg

                        if is_403 and retry < max_retries:
                            logger.warning(f"403 Forbidden for {href}. Refreshing token and retrying...")

                            # 1. Provide Feedback: Remove stale item
                            self.manifest.remove_item_by_href(href)

                            # 2. Refresh: Force Query API
                            # We query the specific item's bbox to get a fresh token for it
                            item_bbox = tuple(item.get("bbox", bounds))
                            fresh_items = self._query_stac_api(item_bbox, collection_id)

                            if fresh_items:
                                # Find the matching item in fresh results (by spatial overlap)
                                # Simple heuristic: replace 'href' with the first fresh item's href
                                # roughly matching location.
                                # Actually, just updating the manifest via _query_stac_api is enough?
                                # No, we need the *new URL* to try opening again.

                                # Let's try to match by bounds check
                                best_match = None
                                from shapely.geometry import box

                                target_box = box(*item_bbox)

                                for fresh in fresh_items:
                                    fresh_box = box(*fresh["bbox"])
                                    if fresh_box.intersects(target_box):
                                        best_match = fresh["href"]
                                        break

                                if best_match:
                                    logger.info(f"Refreshed URL: {best_match}")
                                    href = best_match  # Update loop variable for next try
                                    # Update 'item' dict too so we save valid one to manifest later
                                    item["href"] = best_match
                                    continue  # Retry loop

                            logger.error("Failed to refresh SAS token.")

                        logger.warning(f"Failed to stream land asset {href}: {e}")
                        break

            # Update Manifest only with successfully opened items
            if not self.offline_mode and items_found:
                for item in items_found:
                    self.manifest.add_item(
                        collection_id=collection_id,
                        bbox=item.get("bbox", bounds),
                        asset_href=item["href"],
                        properties=item.get("properties"),
                    )

            if not das:
                logger.debug(f"3DEP/Land items found but resulted in no valid data for {collection_id}.")
                # Negative Cache: Mark this bbox as having no coverage to prevent re-querying
                self.manifest.record_search(collection_id, bounds, 0)
                return None

            # Merge
            merged = merge_arrays(das)

            # Ensure CRS
            if not merged.rio.crs:
                # STAC Land collections (3DEP/COP-30) are EPSG:4326
                merged.rio.write_crs("EPSG:4326", inplace=True)

            # Check if effective data remains
            if merged.isnull().all():
                logger.debug(f"3DEP/Land data masked out (All Water?) for {collection_id}.")
                self.manifest.record_search(collection_id, bounds, 0)
                return None

            return cast(xr.DataArray | None, merged)

        except Exception as e:
            logger.error(f"Error fetching {collection_id}: {e}", exc_info=True)
            return None


# Register
registry.register(Path(__file__).stem, Usgs3DepProvider)
