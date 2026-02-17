import contextlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from ..vdatum import VDatumResolver
from .base import Provider
from .registry import registry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fetch_index_cached(url: str) -> str:
    """Cached fetch of the index page (Standalone)."""
    logger.info("Fetching NOAA Coastal Lidar PDS Index...")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return str(response.text)


class NoaaTopobathyProvider(Provider):
    """
    Provider for NOAA Topobathymetric LiDAR DEMs (Tier 0).
    Distinguished from Terrestrial Lidar (USGS 3DEP) by its use of green-wavelength
    lasers that penetrate the water column to capture submerged topography.

    Bucket: s3://noaa-nos-coastal-lidar-pds/
    """

    INDEX_URL = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/index.html"
    BUCKET_BASE = "noaa-nos-coastal-lidar-pds"

    def __init__(self, cache_dir: str = "~/.cache/topobathysim") -> None:
        self.base_cache_dir = Path(cache_dir).expanduser()
        self.cache_dir = self.base_cache_dir / "noaa_topobathy"
        self.metadata_dir = self.base_cache_dir / "metadata"
        self.inport_cache_dir = self.metadata_dir / "inport"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.inport_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vdatum = VDatumResolver()

        # Internal state
        self._projects: dict[str, str] = {}  # ID -> FolderName
        self._projects_metadata_urls: dict[str, str] = {}  # ID -> InPort/Info URL

        self._active_project_id: str | None = None
        self._tile_index: gpd.GeoDataFrame | None = None
        self._spatial_index: gpd.GeoDataFrame | None = None
        self.fs = fsspec.filesystem("s3", anon=True)

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
    ) -> xr.DataArray:
        """
        Fetch NOAA Topobathy data for the given bounding box.
        Auto-selects the best project based on overlap and recency.
        """
        west, south, east, north = bbox

        # 1. Identify Project
        pid = self.find_project_by_box(west, south, east, north)
        if not pid:
            raise KeyError(f"No NOAA Topobathy project found for bbox {bbox}")

        self.set_active_project(pid)
        if not self._active_project_id:
            raise KeyError(f"Failed to activate project {pid}")

        # 2. Identify Tiles
        tiles = self.resolve_tiles_in_bbox(west, south, east, north)
        if not tiles:
            raise KeyError(f"No tiles found in project {pid} for bbox {bbox}")

        logger.info(f"Topobathy fetch: Found {len(tiles)} tiles in project {pid}")

        # 3. Fetch/Load Tiles
        das = []
        for t in tiles:
            da = self.fetch_tile(t)
            if da is not None:
                # Pre-clip to save memory?
                try:
                    clipped = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs=da.rio.crs)
                    if clipped.size > 0:
                        das.append(clipped)
                except Exception:
                    pass

        if not das:
            raise KeyError(f"Failed to load valid Topobathy data for bbox {bbox}")

        # 4. Merge
        if len(das) == 1:
            merged = das[0]
        else:
            try:
                merged = merge_arrays(das)
                raise KeyError("Failed to load or merge NOAA Topobathy data")
            except Exception as e:
                logger.error(f"Merge error: {e}")
                merged = das[0]

        # 5. Reproject/Clip
        if crs and merged.rio.crs and merged.rio.crs != crs:
            try:
                logger.info(f"Reprojecting Topobathy from {merged.rio.crs} to {crs}")
                merged = merged.rio.reproject(crs)
            except Exception as e:
                logger.warning(f"Reprojection failed: {e}")

        try:
            merged = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs=crs)
        except Exception as e:
            logger.warning(f"Final clip failed: {e}")

        merged.name = "elevation"
        merged.attrs["source_provider"] = "noaa_topobathy"
        merged.attrs["project_id"] = pid
        return cast(xr.DataArray, merged)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the NOAA Topobathy provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA Coastal Topobathy LiDAR",
            "citation": "NOAA National Geodetic Survey (NGS) / OCM.",
            "resolution": "High (Variable, typically 1m - 3m)",
            "url": "https://coast.noaa.gov/digitalcoast/data/coastallidar.html",
        }

    def _fetch_index(self) -> str:
        """
        Fetches the HTML index page from NOAA.
        """
        logger.info("Fetching NOAA Coastal Lidar PDS Index (HTML)...")
        return _fetch_index_cached(self.INDEX_URL)

    def _ensure_project_list(self) -> None:
        """
        Parses the map of ID -> Project Folder Name.
        Checks local JSON or fetches remote HTML index.
        """
        if self._projects:
            return

        json_path = self.metadata_dir / "noaa_coastal_lidar.json"
        import json
        import time

        # 1. Try Loading JSON
        if json_path.exists():
            age = time.time() - json_path.stat().st_mtime
            # 7 days expiration
            if age < 604800:
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                        for k, v in data.items():
                            if isinstance(v, dict):
                                self._projects[k] = v.get("name", "")
                                if "info" in v:
                                    self._projects_metadata_urls[k] = v["info"]
                            else:
                                self._projects[k] = str(v)

                        logger.debug(f"Loaded {len(self._projects)} projects from JSON metadata.")
                        return
                except Exception as e:
                    logger.warning(f"Corrupt metadata JSON {json_path}: {e}")

        # 2. Fetch and Parse HTML
        try:
            text = self._fetch_index()
            new_projects = {}
            new_metadata_urls = {}

            lines = text.split("\n")
            current_folder = None

            for line in lines:
                # 1. Find Folder / Bulk Link
                folder_match = re.search(r'href=".*dem/([^/]+)/index\.html"', line)
                if folder_match:
                    folder_name = folder_match.group(1)
                    parts = folder_name.split("_")
                    if parts and parts[-1].isdigit():
                        pid = parts[-1]
                        current_folder = (pid, folder_name)
                        new_projects[pid] = folder_name

                # 2. Find InPort Link
                if current_folder:
                    pid = current_folder[0]
                    # Check for metadata link in the SAME line
                    pattern = r'href="(https://www.fisheries.noaa.gov/inport/item/\d+)"'
                    inport_match = re.search(pattern, line)
                    if inport_match:
                        new_metadata_urls[pid] = inport_match.group(1)

            self._projects = new_projects
            self._projects_metadata_urls = new_metadata_urls

            p_count = len(self._projects)
            m_count = len(self._projects_metadata_urls)
            logger.info(f"Discovered {p_count} Projects & {m_count} Metadata Links.")

            # 3. Save to JSON
            save_data = {}
            for pid, folder in self._projects.items():
                entry = {"name": folder}
                if pid in self._projects_metadata_urls:
                    entry["info"] = self._projects_metadata_urls[pid]
                save_data[pid] = entry

            try:
                with open(json_path, "w") as f:
                    json.dump(save_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write metadata JSON: {e}")

        except Exception as e:
            logger.error(f"Failed to load project index: {e}")
            if json_path.exists():
                logger.warning("Using stale metadata JSON as fallback.")
                with open(json_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if isinstance(v, dict):
                            self._projects[k] = v.get("name", "")
                            if "info" in v:
                                self._projects_metadata_urls[k] = v["info"]
                        else:
                            self._projects[k] = str(v)

    def fetch_inport_metadata(self, project_id: str) -> dict | None:
        """
        Fetches and parses the InPort XML metadata for a project.
        """
        self._ensure_project_list()

        info_url = self._projects_metadata_urls.get(project_id)
        if not info_url:
            logger.debug(f"No InPort URL known for Project {project_id}")
            return None

        # Ensure it's the XML endpoint
        if not info_url.endswith("/xml"):
            info_url = info_url.rstrip("/")
            if not info_url.endswith("/xml"):
                info_url = f"{info_url}/xml"

        # Extract InPort ID for cache filename
        try:
            parts = info_url.split("/")
            inport_id = parts[parts.index("xml") - 1] if "xml" in parts else parts[-1]
        except Exception:
            inport_id = f"pid_{project_id}"

        xml_path = self.inport_cache_dir / f"{inport_id}.xml"

        # 1. Fetch
        if not xml_path.exists():
            try:
                logger.info(f"Fetching InPort Metadata: {info_url}")
                headers = {"User-Agent": "Mozilla/5.0 TopoBathySim/1.0"}
                r = requests.get(info_url, headers=headers, timeout=10)
                if r.status_code == 200 and len(r.content) > 100:
                    with open(xml_path, "wb") as f:
                        f.write(r.content)
                else:
                    logger.warning(f"InPort fetch failed {r.status_code} for {project_id}")
                    return None
            except Exception as e:
                logger.warning(f"InPort fetch error: {e}")
                return None

        # 2. Parse
        return self._parse_inport_xml(xml_path)

    def _parse_inport_xml(self, xml_path: Path) -> dict:
        """
        Parses FGDC/InPort XML to extract Vertical Datum and Temporal range.
        """
        try:
            import xml.etree.ElementTree

            tree = xml.etree.ElementTree.parse(xml_path)
            root = tree.getroot()

            meta = {"vertical_datum": "Unknown", "start_date": None, "end_date": None, "sensor_name": None}

            # 1. Vertical Datum
            for elem in root.iter():
                if ("altdatum" in elem.tag or "vdatum" in elem.tag) and elem.text:
                    txt = elem.text.lower()
                    if "navd88" in txt or "88" in txt:
                        meta["vertical_datum"] = "NAVD88"
                    elif "geoid18" in txt:
                        meta["vertical_datum"] = "NAVD88 (Geoid18)"
                    elif "ellipsoid" in txt:
                        meta["vertical_datum"] = "Ellipsoid"

            # 2. Time Period
            # <timeinfo> -> <rngdates> -> <begdate> / <enddate>
            beg_dates = []
            end_dates = []
            for elem in root.iter():
                if "begdate" in elem.tag and elem.text:
                    beg_dates.append(elem.text)
                if "enddate" in elem.tag and elem.text:
                    end_dates.append(elem.text)

            if beg_dates:
                meta["start_date"] = sorted(beg_dates)[0]
            if end_dates:
                meta["end_date"] = sorted(end_dates)[-1]

            return meta

        except Exception as e:
            logger.warning(f"Error parsing XML {xml_path}: {e}")
            return {}

    def _ensure_spatial_index(self) -> None:
        """
        Loads the spatial index of project extents (GeoJSON).
        If missing, attempts to build it (this may take time).
        """
        if self._spatial_index is not None:
            return

        index_path = self.metadata_dir / "noaa_project_extents.geojson"

        if not index_path.exists():
            logger.warning(
                "Spatial index missing. Building NOAA Project Index (this may take several minutes)..."
            )
            try:
                # Run the builder script logic
                # To avoid circular imports, we import the function here
                # Note: `scripts` must be a package for this to work relative within `topobathysim` context
                from ..scripts.build_noaa_index import main as build_index

                # If running purely as script, build_index execution context might differ,
                # but since we are in `topobathysim` package, it should work.
                build_index()
            except ImportError:
                # Fallback if scripts isn't importable as package
                logger.error(
                    "Could not import build_noaa_index script. Please run "
                    "'python -m topobathysim.scripts.build_noaa_index' manually."
                )
                return
            except Exception as e:
                logger.error(f"Failed to build spatial index: {e}")
                return

        if index_path.exists():
            try:
                self._spatial_index = gpd.read_file(index_path)
                # Parse dates for sorting
                import pandas as pd

                if "end_date" in self._spatial_index.columns:
                    self._spatial_index["end_date"] = pd.to_datetime(
                        self._spatial_index["end_date"], errors="coerce"
                    )

                logger.info(f"Loaded NOAA Spatial Index with {len(self._spatial_index)} projects.")
            except Exception as e:
                logger.error(f"Failed to load spatial index {index_path}: {e}")
                self._spatial_index = None

    def find_project_by_box(self, west: float, south: float, east: float, north: float) -> str | None:
        """
        Identifies the best project ID for the bounding box using the spatial index.
        Prioritizes:
        1. Spatial Overlap (intersects)
        2. Recency (end_date)
        3. Vertical Datum Quality (NAVD88/Geoid18 > Ellipsoid > Unknown)
        """
        self._ensure_project_list()

        # Try loading spatial index
        self._ensure_spatial_index()

        if self._spatial_index is not None and not self._spatial_index.empty:
            query_box = box(west, south, east, north)

            # CRS check
            search_geom = query_box
            if self._spatial_index.crs and self._spatial_index.crs.to_string() != "EPSG:4326":
                # Index should be 4326 per build script, but verify
                pass

            # Filter by intersection
            # Use spatial index if available, or brute force intersection
            candidates = self._spatial_index[self._spatial_index.intersects(search_geom)].copy()

            if not candidates.empty:
                # Conflict Resolution
                # 1. Datum Priority (Geoid18 > NAVD88 > Ellipsoid)
                def datum_score(d: object) -> int:
                    d_str = str(d).lower()
                    if "geoid18" in d_str:
                        return 3
                    if "navd88" in d_str:
                        return 2
                    if "ellipsoid" in d_str:
                        return 1
                    return 0

                candidates["datum_score"] = candidates["vertical_datum"].apply(datum_score)

                # 2. Sort by Datum Score (Desc), then End Date (Desc)
                candidates = candidates.sort_values(by=["datum_score", "end_date"], ascending=[False, False])

                best_match = candidates.iloc[0]
                logger.info(
                    f"Auto-selected Project {best_match['project_id']} "
                    f"({best_match['project_name']}) for bbox."
                )
                return str(best_match["project_id"])

        # If spatial index is missing or empty, we fail loudly as requested
        if self._spatial_index is None or self._spatial_index.empty:
            raise RuntimeError(
                "NOAA Spatial Index is unavailable. "
                "Please run 'python -m topobathysim.scripts.build_noaa_index' to generate it."
            )

        return None

    def set_active_project(self, project_id: str) -> None:
        """
        Sets the active project and loads its tile index.
        """
        self._ensure_project_list()
        if project_id not in self._projects:
            logger.error(f"Project ID {project_id} not found in index.")
            return

        if self._active_project_id == project_id and self._tile_index is not None:
            return

        self._active_project_id = project_id

        # Find tile index in laz/geoid18/{ID}, laz/geoid12b/{ID}, or dem/{FOLDER}/
        # We try geoid18 first, then geoid12b, then fallback to dem/ folder
        candidates = [f"laz/geoid18/{project_id}/", f"laz/geoid12b/{project_id}/"]

        folder_name = self._projects.get(project_id)
        if folder_name:
            candidates.append(f"dem/{folder_name}/")

        index_file_key = None

        for prefix in candidates:
            try:
                # s3fs ls returns full paths
                # e.g. ["bucket/key", ...]
                path_to_list = f"{self.BUCKET_BASE}/{prefix}"
                try:
                    files = self.fs.ls(path_to_list)
                except FileNotFoundError:
                    continue

                if not files:
                    continue

                for f in files:
                    # f is full path e.g. noaa-nos.../laz/...
                    name = Path(f).name.lower()  # ensure case-insensitive check
                    if "tileindex" in name and (
                        name.endswith(".gpkg") or name.endswith(".zip") or name.endswith(".shp")
                    ):
                        index_file_key = f
                        break
                if index_file_key:
                    break
            except Exception:
                continue

        if not index_file_key:
            # Fallback for datasets with non-standard tile index locations
            pass

        if not index_file_key:
            logger.warning(f"No tile index found for Project {project_id} in standard locations.")
            return

        # Download Index
        local_index_path = self.cache_dir / Path(index_file_key).name
        if not local_index_path.exists():
            logger.info(f"Downloading Tile Index: {index_file_key}")
            try:
                self.fs.get(index_file_key, str(local_index_path))
            except Exception as e:
                logger.error(f"Failed to download tile index: {e}")
                return

        try:
            self._tile_index = gpd.read_file(local_index_path)
            # Ensure CRS?
        except Exception as e:
            logger.error(f"Failed to load tile index {local_index_path}: {e}")
            self._tile_index = None

    def resolve_tiles_in_bbox(self, west: float, south: float, east: float, north: float) -> list[str]:
        """
        Returns list of tile filenames (or download URLs) for the bbox.
        """
        if self._tile_index is None:
            return []

        search_box = box(west, south, east, north)

        # Reproject search box to index CRS if different
        query_geom = search_box
        if self._tile_index.crs and self._tile_index.crs.to_string() != "EPSG:4326":
            gdf_box = gpd.GeoSeries([search_box], crs="EPSG:4326")
            with contextlib.suppress(Exception):
                query_geom = gdf_box.to_crs(self._tile_index.crs)[0]

        matches = self._tile_index[self._tile_index.intersects(query_geom)]

        results = []
        for _, row in matches.iterrows():
            fname = None
            for col in ["Name", "name", "URL", "url", "id", "TileName"]:
                if col in row:
                    fname = str(row[col])
                    break

            if fname:
                if not fname.endswith(".tif"):
                    fname += ".tif"  # Append extension if missing
                results.append(fname)

        return list(set(results))

    def fetch_tile(self, tile_filename: str) -> xr.DataArray | None:
        """
        Fetches the specific COG, applies VDatum corrections, and caches as Zarr.
        """
        if not self._active_project_id:
            return None

        # Resolve Project Folder Name
        if self._active_project_id not in self._projects:
            self._ensure_project_list()

        if self._active_project_id not in self._projects:
            logger.error(f"Cannot resolve folder for Project ID {self._active_project_id}")
            return None

        folder_name = self._projects[self._active_project_id]

        # Construct Remote and Local paths
        http_url = f"https://{self.BUCKET_BASE}.s3.amazonaws.com/dem/{folder_name}/{tile_filename}"
        local_filename = f"{self._active_project_id}_{tile_filename}"
        local_cog_path = self.cache_dir / local_filename

        # Zarr Cache Setup
        zarr_name = local_filename.replace(".tif", "").replace(".tiff", "") + "_navd88.zarr"
        zarr_path = self.cache_dir / "zarr" / zarr_name

        # 1. Zarr Cache Hit (Fast Path)
        if zarr_path.exists():
            try:
                da = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.debug(f"Topobathy Zarr Cache Hit: {zarr_name}")
                return da
            except Exception as e:
                logger.warning(f"Corrupt Zarr {zarr_path}: {e}")
                import shutil

                shutil.rmtree(zarr_path, ignore_errors=True)

        # 2. Cache Miss - Acquire Lock
        import fcntl

        lock_path = self.cache_dir / "zarr" / (zarr_name + ".lock")

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                # Double Check inside lock
                if zarr_path.exists():
                    try:
                        da = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                        logger.info(f"Topobathy Zarr Cache Hit (Post-Lock): {zarr_name}")
                        return da
                    except Exception:
                        pass

                logger.info(f"Topobathy Zarr Cache Miss: Creating {zarr_name}")

                # 3. Ensure Raw COG is Downloaded
                if not local_cog_path.exists():
                    cog_lock = self.cache_dir / f"{local_filename}.lock"
                    temp_cog = self.cache_dir / f".tmp_{local_filename}"
                    try:
                        with open(cog_lock, "w") as cl:
                            fcntl.flock(cl, fcntl.LOCK_EX)
                            if not local_cog_path.exists():
                                logger.info(f"Downloading COG: {http_url}")
                                with requests.get(http_url, stream=True, timeout=30) as r:
                                    r.raise_for_status()
                                    with open(temp_cog, "wb") as f:
                                        for chunk in r.iter_content(chunk_size=32768):
                                            f.write(chunk)
                                Path(temp_cog).rename(local_cog_path)
                    except Exception as e:
                        logger.error(f"Failed to download COG {http_url}: {e}")
                        if Path(temp_cog).exists():
                            Path(temp_cog).unlink()
                        return None

                # 4. Open COG and Apply Adjustments
                try:
                    # Load lazily
                    da_raw = cast(
                        xr.DataArray,
                        rioxarray.open_rasterio(local_cog_path, chunks={"x": 2048, "y": 2048}, masked=True),
                    )

                    # Metadata & VDatum Logic
                    meta = self.fetch_inport_metadata(self._active_project_id) or {}

                    # --- PROVENANCE METADATA ---
                    import datetime

                    # Core identity
                    da_raw.attrs["survey_source"] = self._active_project_id
                    da_raw.attrs["source_url"] = http_url

                    # Time
                    da_raw.attrs["date_created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    da_raw.attrs["start_date"] = meta.get("start_date", "Unknown")
                    da_raw.attrs["end_date"] = meta.get("end_date", "Unknown")

                    # Vertical Reference
                    vdatum = meta.get("vertical_datum", "Unknown").lower()
                    da_raw.attrs["vertical_datum_original"] = vdatum
                    da_raw.attrs["vertical_datum"] = vdatum  # Current state

                    # Links
                    info_url = self._projects_metadata_urls.get(self._active_project_id)
                    if info_url:
                        da_raw.attrs["metadata_url"] = info_url

                    # Apply Correction if Ellipsoid
                    if "ellipsoid" in vdatum:
                        try:
                            from pyproj import Transformer

                            bounds = da_raw.rio.bounds()
                            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

                            crs = da_raw.rio.crs
                            if crs:
                                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                                lon, lat = t.transform(cx, cy)

                                offset = self.vdatum.get_ellipsoid_to_navd88_offset(lat, lon)
                                logger.info(f"Applying VDatum Offset {offset:.3f}m to {local_filename}")
                                da_raw = da_raw + offset

                                # Update Attributes post-correction
                                da_raw.attrs["vertical_datum"] = "NAVD88"
                                da_raw.attrs["correction_method"] = "VDatum Geoid18"
                                da_raw.attrs["vdatum_offset"] = offset
                        except Exception as e:
                            logger.warning(f"VDatum correction failed: {e}")

                    # 5. Write to Zarr
                    # Ensure good chunks for writing
                    if "x" in da_raw.dims and "y" in da_raw.dims:
                        da_raw = da_raw.chunk({"y": 1024, "x": 1024})

                    da_raw.to_zarr(zarr_path, mode="w", consolidated=True)
                    logger.info(f"Created Zarr Cache: {zarr_path.name}")

                    # Return re-opened Zarr
                    return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

                except Exception as e:
                    logger.error(f"Failed to process/write Zarr for {local_filename}: {e}")
                    return None

        except Exception as e:
            logger.error(f"Zarr lock/process failed: {e}")
            return None
        finally:
            if lock_path.exists():
                lock_path.unlink()


# Register
registry.register("topobathy", NoaaTopobathyProvider)
