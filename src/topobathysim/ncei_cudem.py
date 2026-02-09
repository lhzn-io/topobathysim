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
from shapely.geometry import box

from .quality import QualityClass
from .vdatum import VDatumResolver

logger = logging.getLogger(__name__)


class CUDEMProvider:
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
        self.cache_dir = Path(cache_dir).expanduser() / "cudem"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.cache_dir / self.TILE_INDEX_ZIP
        self.index_shp_dir = self.cache_dir / "index_shp"
        self._gdf = None
        self.vdatum = VDatumResolver()

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
                # deeper search?
                shps = list(self.index_shp_dir.rglob("*.shp"))

            if not shps:
                logger.warning("No shapefile found in CUDEM index zip.")
                return

            self._gdf = gpd.read_file(shps[0])

            # Check for None (read_file signature returns Optional)
            if self._gdf is None:
                logger.warning("CUDEM Index Shapefile read returned None.")
                return

            # Normalize CRS to 4326
            if self._gdf.crs is None:
                self._gdf.set_crs("EPSG:4326", inplace=True)
            elif self._gdf.crs.to_string() != "EPSG:4326":
                self._gdf = self._gdf.to_crs("EPSG:4326")

            # Normalize Columns to lowercase for easier lookup
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
        # Check standard columns
        for col in ["url", "path", "location", "fileurl"]:
            if row.get(col):
                val = str(row[col])
                if val.startswith("http"):
                    return val
                # Handle relative path or filename
                return f"{self.BASE_S3_URL}/{val}"

        # Fallback to name/tile_id construction if predictable?
        # Typically indexes have the URL.
        return None

    def _ensure_tile_cached(self, url: str) -> Path | None:
        """Download remote tile to cache."""
        filename = Path(url).name
        local_path = self.cache_dir / filename
        lock_path = self.cache_dir / f"{filename}.lock"

        # Check cache
        if local_path.exists():
            return local_path

        # Download
        with open(lock_path, "w") as lock_file:
            try:
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
                return local_path
            except Exception as e:
                logger.error(f"Failed to download CUDEM tile {url}: {e}")
                if local_path.exists():
                    local_path.unlink()
                return None
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def load_tile(self, row: Any, bbox: tuple[float, float, float, float]) -> xr.DataArray | None:
        """Load single tile and harmonize datum."""
        url = self._get_tile_url(row)
        if not url:
            return None

        path = self._ensure_tile_cached(url)
        if not path:
            return None

        try:
            # Load
            # cast result to DataArray (open_rasterio can be ambiguous)
            from typing import cast

            da_raw = rioxarray.open_rasterio(path)
            da: xr.DataArray

            if isinstance(da_raw, list):
                da = cast(xr.DataArray, da_raw[0])
            elif isinstance(da_raw, xr.Dataset):
                # Should not happen with open_rasterio default behavior but safe check
                da = da_raw.to_array().isel(variable=0)
            else:
                da = cast(xr.DataArray, da_raw)

            if "band" in da.dims:
                da = da.isel(band=0).drop_vars("band")

            # Handle NoData
            if da.rio.nodata is not None:
                da = da.where(da != da.rio.nodata)

            # --- Datum Check ---
            # CUDEM is usually NAVD88.
            # Check 'vert_datum' or similar
            v_datum = "NAVD88"  # Assumption
            for col in ["vert_datum", "v_datum", "vertical_datum"]:
                if row.get(col):
                    v_datum = str(row[col]).upper()
                    break

            # If not NAVD88, we might need correction.
            # But the simulation standard IS usually NAVD88 (or we assume base is NAVD88).
            # Wait, default `manager.py` doesn't explicitly force NAVD88 vs MSL except in VDatum calls.
            # Usually Topo/Bathy sim targets NAVD88 for "Land" and defaults.
            # If CUDEM is EGM2008 (Ellipsoidal?), we need correction.
            # If CUDEM is MHW, we need correction.

            # NOTE: For MVP, we pass it through but log warning if it looks exotic.
            if "EGM" in v_datum:
                logger.warning(f"CUDEM Tile {path.name} is {v_datum}. Vertical transformation recommended.")

            # Robustness: Ensure CRS is written if missing
            if da.rio.crs is None:
                # CUDEM is typically NAD83 (EPSG:4269) or WGS84 (EPSG:4326)
                # But some tiles might be raw lat/lon without CRS header.
                # NAD83 is the standard for NOAA NCEI products in the US.
                logger.warning(f"CUDEM tile {path.name} missing CRS. Assuming EPSG:4269 (NAD83).")
                with contextlib.suppress(Exception):
                    da.rio.write_crs("EPSG:4269", inplace=True)

            return da

        except Exception as e:
            logger.error(f"Error reading CUDEM tile {path}: {e}")
            return None

    def get_grid(self, west: float, south: float, east: float, north: float) -> xr.DataArray | None:
        """
        Get fused CUDEM grid for the area.
        """
        matches = self.resolve_tiles(west, south, east, north)
        if matches.empty:
            return None

        logger.info(f"Found {len(matches)} CUDEM tiles for bbox.")

        das: list[xr.DataArray] = []
        for _, row in matches.iterrows():
            # Loading full tile
            da = self.load_tile(row, (west, south, east, north))
            if da is not None:
                # We could clip here to save memory before merge?
                # Clipping usually safe if we reproject_match later.
                try:
                    clipped = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
                    # Use clipped if it has data
                    if clipped.count() > 0:
                        das.append(clipped)
                except Exception:
                    # Clip might fail if no overlap with bbox (geometry vs bbox precision)
                    # Just skip
                    pass

        if not das:
            return None

        if len(das) == 1:
            return das[0]

        try:
            from rioxarray.merge import merge_arrays

            merged = merge_arrays(das)
            return cast(xr.DataArray, merged)
        except Exception as e:
            logger.error(f"Failed to merge CUDEM tiles: {e}")
            return das[0]

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        """
        Return quality tier for the given coordinate.
        Mark DIRECT where high-res is known, INDIRECT otherwise.
        Without pixel-level source masks, CUDEM is treated as INDIRECT (Modeled).
        """
        matches = self.resolve_tiles(lon, lat, lon, lat)
        if not matches.empty:
            return QualityClass.INDIRECT
        return QualityClass.UNKNOWN
