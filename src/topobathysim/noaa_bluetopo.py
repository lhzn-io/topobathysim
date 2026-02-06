import logging
from pathlib import Path
from typing import Any

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from shapely.geometry import Point

from .quality import QualityClass
from .vdatum import VDatumResolver

logger = logging.getLogger(__name__)


class NoaaBlueTopoProvider:
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

    def __init__(self, cache_dir: str = "~/.cache/topobathysim"):
        self.vdatum = VDatumResolver()
        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "bluetopo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Tile Scheme stays in root or moves? Let's move to bluetopo dir too to be clean.
        self.scheme_path = self.cache_dir / "BlueTopo_Tile_Scheme.gpkg"
        self._gdf = None

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

        if not self.scheme_path.exists():
            try:
                # Ensure directory exists (it might have been cleared runtime)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Dynamic Resolution
                download_url = self._resolve_scheme_url()

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

            # Load Lazy for memory efficiency
            da = rioxarray.open_rasterio(path, chunks={"x": 2048, "y": 2048})

            if "band" in da.dims:
                da = da.isel(band=0).drop_vars("band")

            # Clip
            # WARNING: Clipping here can cause edge artifacts (transparency triangles)
            # if the reprojection of the bbox doesn't fully cover the rotated raster corner.
            # We return the full tile and let main.py's reproject_match handle the cropping.
            return da

        except Exception as e:
            logger.error(f"Load Tile Error: {e}")
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

        Strategy Cascade:
        1. **Embedded RAT**: Inspects Band 3 of the local GeoTIFF.
        2. **Sidecar RAT**: Downloads the linked .vat.dbf from the Tile Scheme.
        3. **HSMDB API**: Queries the NOAA NCEI spatial API directly.
        """

        # 1. Tile Resolution
        tile_id = self.get_tile_id(lat, lon)
        if not tile_id:
            logger.debug("No BlueTopo tile found. Fallback to API.")
            return self._resolve_from_hsmdb_api(lat, lon)

        # 2. Local File Inspection (Embedded RAT + Pixel Value)
        local_path = self._ensure_tile_cached(tile_id)
        pixel_val: int | None = None

        if local_path:
            try:
                # Use rioxarray/rasterio as osgeo.gdal is unreliable in this env
                with rioxarray.open_rasterio(local_path, masked=True) as da:
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
        """Helper to transform lat/lon to pixel coordinates."""
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
        """Helper to query a GDAL RAT."""
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
        """Downloads and parses the sidecar RAT linked in the GPKG."""
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
            # Download Sidecar
            filename = Path(rat_link).name
            sidecar_dir = Path(self.cache_dir) / "bluetopo" / "sidecars"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_file = sidecar_dir / filename

            if not sidecar_file.exists():
                logger.info(f"Downloading Sidecar RAT: {rat_link}")
                r = requests.get(rat_link, timeout=30)
                if r.status_code == 200:
                    with open(sidecar_file, "wb") as f:
                        f.write(r.content)
                else:
                    logger.warning(f"Failed to download sidecar: {r.status_code}")
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
        """Parses GDAL PAM XML to find Survey ID."""
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
        """Tertiary fallback: Query NCEI HSMDB API."""
        url = (
            "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/nos_hydro_dynamic/MapServer/0/query"
        )
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "SURVEY_ID",
            "returnGeometry": "false",
            "f": "json",
        }
        try:
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
