import contextlib
import logging
import re
from pathlib import Path

import fsspec
import geopandas as gpd
import requests
import rioxarray
import xarray as xr
from shapely.geometry import box

from .vdatum import VDatumResolver

logger = logging.getLogger(__name__)


class NoaaTopobathyProvider:
    """
    Provider for NOAA Topobathymetric LiDAR DEMs (Tier 0).
    Distinguished from Terrestrial Lidar (USGS 3DEP) by its use of green-wavelength
    lasers that penetrate the water column to capture submerged topography.

    Bucket: s3://noaa-nos-coastal-lidar-pds/
    """

    INDEX_URL = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/index.html"
    BUCKET_BASE = "noaa-nos-coastal-lidar-pds"

    def __init__(self, cache_dir: str = "~/.cache/topobathysim") -> None:
        self.cache_dir = Path(cache_dir).expanduser() / "noaa_topobathy"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vdatum = VDatumResolver()

        # Internal state
        self._projects: dict[str, str] = {}  # ID -> FolderName
        self._active_project_id: str | None = None
        self._tile_index: gpd.GeoDataFrame | None = None
        self.fs = fsspec.filesystem("s3", anon=True)

    def _ensure_project_list(self) -> None:
        """
        Parses the main index to map ID -> Project Folder Name.
        """
        if self._projects:
            return

        try:
            logger.info("Fetching NOAA Coastal Lidar PDS Index...")
            response = requests.get(self.INDEX_URL, timeout=10)
            response.raise_for_status()

            # Simple regex parser for Bulk Download links
            # Link: [Bulk Download](https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NY_LakeOntario_DEM_2023_10402/index.html)
            # Regex: dem/([^/]+)/index\.html
            pattern = r"dem/([^/]+)/index\.html"
            for match in re.finditer(pattern, response.text):
                folder_name_match = match.group(1)
                # Extract ID from end of folder name
                # e.g. NY_LakeOntario_DEM_2023_10402 -> 10402
                parts = folder_name_match.split("_")
                if parts and parts[-1].isdigit():
                    pid = parts[-1]
                    self._projects[pid] = folder_name_match

            logger.info(f"Discovered {len(self._projects)} Coastal Lidar Projects.")

        except Exception as e:
            logger.error(f"Failed to load project index: {e}")

    def find_project_by_box(self, west: float, south: float, east: float, north: float) -> str | None:
        """
        Identifies the best project ID for the bounding box.
        Currently relies on keyword heuristics or manual ID injection
        since we lack a global spatial index.
        """
        self._ensure_project_list()

        # Heuristic: Check for known project IDs first (e.g. LIS 2023 = 10274)
        # In a real impl, we might check 'roi.geojson' if we had it.
        # For Phase 8 prompt, we specifically look for LIS (10274).

        # Validating hardcoded knowledge as fallback
        # LIS 2023 -> 10274
        if "10274" in self._projects:
            # Check if bbox is roughly in LIS?
            # LIS approx: -74 to -71, 40 to 42.
            lis_box = box(-74.5, 40.5, -71.5, 41.5)
            query_box = box(west, south, east, north)
            if lis_box.intersects(query_box):
                return "10274"

        # TODO: Implement full spatial search via checking metadata of all projects
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

        # Find tile index in laz/geoid18/{ID} or laz/geoid12b/{ID}
        # We try geoid18 first as per prompt requirement
        candidates = [f"laz/geoid18/{project_id}/", f"laz/geoid12b/{project_id}/"]

        index_file_key = None

        for prefix in candidates:
            try:
                files = self.fs.ls(f"{self.BUCKET_BASE}/{prefix}")
                for f in files:
                    # f is full path e.g. noaa-nos.../laz/...
                    name = Path(f).name
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
            logger.warning(f"No tile index found for Project {project_id}.")
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
        Fetches the specific COG from the project DEM folder.
        """
        if not self._active_project_id:
            return None

        folder_name = self._projects[self._active_project_id]

        # Using /vsicurl/ directly
        http_url = f"https://s3.amazonaws.com/{self.BUCKET_BASE}/dem/{folder_name}/{tile_filename}"
        vsi_path = f"/vsicurl/{http_url}"

        try:
            da = rioxarray.open_rasterio(vsi_path)
            return da
        except Exception as e:
            logger.warning(f"Failed to stream tile {vsi_path}: {e}")
            return None

    def get_grid(
        self, west: float, south: float, east: float, north: float, project_id: str | None = None
    ) -> xr.DataArray | None:
        """
        High level interface to get merged grid.
        """
        if project_id:
            self.set_active_project(project_id)
        elif self._active_project_id is None:
            # Try to auto-detect
            pid = self.find_project_by_box(west, south, east, north)
            if pid:
                self.set_active_project(pid)

        if not self._active_project_id:
            return None

        tiles = self.resolve_tiles_in_bbox(west, south, east, north)
        if not tiles:
            return None

        das = []
        for t in tiles:
            da = self.fetch_tile(t)
            if da is not None:
                das.append(da)

        if not das:
            return None

        if len(das) == 1:
            return das[0]

        from rioxarray.merge import merge_arrays

        try:
            merged = merge_arrays(das)
            return merged
        except Exception as e:
            logger.error(f"Merge error: {e}")
            return None
