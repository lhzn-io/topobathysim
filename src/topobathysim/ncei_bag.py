import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import requests
import xarray as xr

from .vdatum import VDatumResolver

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@lru_cache(maxsize=8)
def _read_bag_cached(local_path: Path) -> xr.DataArray | None:
    """
    Reads BAG using h5py (or rasterio) and converts to xarray with NAVD88 correction.
    Cached to avoid re-opening/parsing headers for every tile.
    """
    from .vdatum import VDatumResolver

    try:
        import rioxarray as rxr

        # CRITICAL STABILITY FIX: Use chunks to initiate Delayed/Lazy loading via Dask.
        # Reading entire 50cm BAGs (GBs) into memory causes OOM crashes.
        # Open with Dask chunks for lazy loading
        with ignore_specific_gdal_warnings("cornerPoints not consistent with resolution"):
            da_raw = rxr.open_rasterio(local_path, chunks={"x": 2048, "y": 2048}, masked=True)
            if isinstance(da_raw, list):
                da = da_raw[0]
            elif isinstance(da_raw, xr.Dataset):
                da = da_raw.to_array().isel(variable=0)
            else:
                da = da_raw

            from typing import cast

            da = cast(xr.DataArray, da)

        # BAGs usually have 'elevation' and 'uncertainty'.
        # Rasterio usually reads band 1 as elevation.
        elev = da.isel(band=0).drop_vars("band")

        # Check for Ellipsoid vs MLLW
        filename = local_path.name
        is_ellipsoid = "_Ellipsoid_" in filename or "Ellipsoid" in filename

        # Vertical Datum Correction (to NAVD88)
        crs = elev.rio.crs
        bounds = elev.rio.bounds()  # (minx, miny, maxx, maxy)
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        offset = 0.0
        try:
            from pyproj import Transformer

            if crs:
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(center_x, center_y)

                if is_ellipsoid:
                    # Ellipsoid -> NAVD88
                    offset = VDatumResolver.get_ellipsoid_to_navd88_offset(lat, lon)  # type: ignore
                    logger.info(f"Applying Ellipsoid->NAVD88 offset of {offset:.3f}m for {filename}")
                else:
                    # MLLW -> NAVD88
                    offset = VDatumResolver.get_mllw_to_navd88_offset(lat, lon)  # type: ignore
                    logger.info(f"Applying MLLW->NAVD88 offset of {offset:.3f}m for {filename}")

                # Applying offset to Dask array adds a task to the graph (Lazy).
                elev = elev + offset

        except Exception as e:
            logger.warning(f"Failed to apply VDatum correction: {e}")

        elev.attrs["survey_source"] = filename
        return elev

    except Exception as e:
        logger.error(f"Error reading BAG {local_path}: {e}")
        return None


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

    @classmethod
    def _scrape_landing_page(cls, download_url: str) -> str | None:
        """Helper to scrape .bag URL from HTML landing page."""
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

            # Prioritize MLLW (Tidal Datum usually matches user expectation)
            # Deprioritize Ellipsoid because we lack robust Geoid separation grids.
            mllw_links = [L for L in bag_links if "_MLLW_" in L or "MLLW" in L]
            ellip_links = [L for L in bag_links if "_Ellipsoid_" in L or "Ellipsoid" in L]

            if mllw_links:
                logger.info(f"Preferred MLLW BAG found: {mllw_links[0]}")
                return mllw_links[0]

            if ellip_links:
                logger.info(
                    f"Selecting Ellipsoid BAG (Fallback, may require large Geoid offset): {ellip_links[0]}"
                )
                return ellip_links[0]

            if mllw_links:
                logger.info(f"Selecting MLLW BAG: {mllw_links[0]}")
                return mllw_links[0]

            if bag_links:
                return bag_links[0]
        except Exception as e:
            logger.warning(f"Scraping failed for {download_url}: {e}")
        return None

    @classmethod
    @lru_cache(maxsize=128)
    def find_bag_by_survey_id(cls, survey_id: str) -> str | None:
        """
        Query NCEI API for a specific Survey ID (e.g., 'H13385') to get the BAG download URL.
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
                    return str(download_url)

                if download_url and download_url.lower().endswith(".html"):
                    return cls._scrape_landing_page(download_url)

                # Default return
                return str(download_url) if download_url else None
            else:
                logger.warning(f"No features in response: {data}")

            logger.warning(f"No NCEI record found for Survey ID: {survey_id}")
            return None

        except Exception as e:
            logger.error(f"BAG Discovery Field: {e}")
            return None

    @classmethod
    @lru_cache(maxsize=128)
    def find_bag_by_location(cls, lat: float, lon: float) -> str | None:
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
                        return str(download_url)
                    if download_url.lower().endswith(".html"):
                        return cls._scrape_landing_page(download_url)
            else:
                logger.info(f"Spatial Query - No features found (Standard Fallback). Response: {data}")
        except Exception as e:
            logger.warning(f"Spatial query failed: {e}")

        return None


class BAGProvider:
    """
    Manages downloading, caching, and reading of NOAA BAG files.
    """

    def __init__(self, cache_dir: str = "~/.cache/topobathysim"):
        self.cache_dir = Path(cache_dir).expanduser() / "bag"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vdatum = VDatumResolver()

    def fetch_bag(self, survey_id: str, download_url: str | None = None) -> xr.DataArray | None:
        """
        Fetches and reads a BAG file for a given Survey ID.
        Auto-discovers URL if not provided.
        """
        if not download_url:
            download_url = BAGDiscovery.find_bag_by_survey_id(survey_id)

        if not download_url:
            return None

        # Determine filename from URL
        filename = download_url.split("/")[-1]
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
                        logger.info(f"Downloading BAG: {download_url}")
                        # Use requests for progress? Or urllib/shutil for simplicity
                        # Since BAGs are large, streaming via requests is better
                        with requests.get(download_url, stream=True) as r:
                            r.raise_for_status()
                            with open(temp_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=32768):
                                    f.write(chunk)

                        Path(temp_path).rename(local_path)
            except Exception as e:
                logger.error(f"Failed to download BAG {survey_id}: {e}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                return None

        # 2. Read BAG
        return self._read_bag(local_path)

    def _read_bag(self, local_path: Path) -> xr.DataArray | None:
        """wrapper to call standalone cached function."""
        # Note: B019 warns about lru_cache on method.
        # We rely on the standalone function's cache.
        return _read_bag_cached(local_path)
