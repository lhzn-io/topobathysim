import logging
from functools import lru_cache
from pathlib import Path
from typing import cast

import laspy
import numpy as np
import rioxarray as rxr
import s3fs
import xarray as xr
from affine import Affine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _query_3dep_stac(bbox: tuple[float, float, float, float]) -> dict | None:
    """
    Cached STAC query for 3DEP Lidar (In-Memory Only).
    """
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    import planetary_computer
    from pystac_client import Client

    try:
        logger.debug(f"Querying STAC: {stac_url} with bbox {bbox}")
        catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
        search = catalog.search(collections=["3dep-lidar-copc"], bbox=bbox, limit=1)
        items = list(search.items())

        if not items:
            return None

        # Extract only what we need to return
        item = items[0]
        assets = item.assets
        href = assets["data"].href

        props = item.properties
        native_epsg = props.get("proj:epsg")
        projjson = props.get("proj:projjson", {})

        return {
            "href": href,
            "native_epsg": native_epsg,
            "projjson": projjson,
            "id": item.id,
            "bbox": item.bbox,
            "properties": item.properties,
        }
    except Exception as e:
        logger.warning(f"STAC Query Error: {e}")
        return None


class UsgsLidarProvider:
    """
    Fetches and processes Lidar data from NOAA LAZ files.
    Filters for 'Bare Earth' (Class 2) and rasterizes to GeoTIFF.
    """

    def __init__(self, cache_dir: str = "~/.cache/topobathysim", offline_mode: bool = False):
        self.cache_dir = Path(cache_dir).expanduser() / "lidar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode
        # S3 filesystem
        self.fs = s3fs.S3FileSystem(anon=True)

        # Manifest for Offline Lookup
        from .manifest import OfflineManifest

        self.manifest = OfflineManifest(self.cache_dir)

    def _get_cache_path(self, url: str) -> Path:
        """
        Derives the local cache path from the URL.
        """
        import hashlib

        # Strip query params
        base_url = url.split("?")[0]
        url_hash = hashlib.md5(base_url.encode()).hexdigest()

        original_name = Path(base_url).name
        if not original_name or len(original_name) > 50:
            filename = f"{url_hash}.laz"
        else:
            filename = f"{original_name.split('.')[0]}_{url_hash[:8]}.laz"

        return self.cache_dir / filename

    def _download_and_cache(self, url: str) -> Path | None:
        """
        Downloads a file from a URL to the cache directory.
        """
        import fcntl
        import shutil
        import urllib.request

        local_path = self._get_cache_path(url)
        lock_path = self.cache_dir / f"{local_path.name}.lock"

        # 1. Fast Path
        if local_path.exists():
            return local_path

        if self.offline_mode:
            logger.warning(f"Offline Mode: Custom download required but file missing: {local_path.name}")
            return None

        # 2. Acquire Lock
        try:
            with open(lock_path, "w") as lock_file:
                # Non-blocking lock attempt? No, we want to wait if another process is downloading
                # But if we are in a background thread, blocking is fine.
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # 3. Double Check
                    if local_path.exists():
                        return local_path

                    logger.info(f"Downloading Lidar asset to {local_path}...")

                    # Use temp file
                    temp_path = self.cache_dir / f".tmp_{local_path.name}"

                    if url.startswith("s3://"):
                        self.fs.get(url, str(temp_path))
                        Path(temp_path).rename(local_path)
                    else:
                        with (
                            urllib.request.urlopen(url) as response,
                            open(temp_path, "wb") as out_file,
                        ):
                            shutil.copyfileobj(response, out_file)
                        Path(temp_path).rename(local_path)

                    logger.info("Background Download complete.")
                    return local_path

                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Failed to download Lidar asset: {e}")
            if "temp_path" in locals() and Path(cast(str, temp_path)).exists():
                Path(cast(str, temp_path)).unlink()
            return None

    def _read_laz_file(
        self,
        local_path: Path,
        bounds: tuple[float, float, float, float] | None = None,  # Optional bounds for cropping?
        resolution: float = 4.0,
        target_crs: str = "EPSG:4326",
        native_crs_str: str | None = None,
    ) -> xr.DataArray | None:
        """
        Reads a local LAZ file, filters Class 2, and rasterizes.
        """
        try:
            with laspy.open(local_path) as fh:
                las = fh.read()

            # Filter Class 2
            ground_points = las.points[las.classification == 2]

            if len(ground_points) == 0:
                logger.warning(f"No ground points (Class 2) found in {local_path.name}.")
                return None

            x = ground_points.x
            y = ground_points.y
            z = ground_points.z

            if len(x) == 0:
                return None

            # Rasterize Logic (Binning)
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)

            width = int(np.ceil((max_x - min_x) / resolution))
            height = int(np.ceil((max_y - min_y) / resolution))

            if width <= 0 or height <= 0:
                return None

            x_idx = ((x - min_x) / resolution).astype(int)
            # y_idx = ((max_y - y) / resolution).astype(int)
            # Use Cartesian coordinates (0 is min_y)
            y_idx = ((y - min_y) / resolution).astype(int)

            x_idx = np.clip(x_idx, 0, width - 1)
            y_idx = np.clip(y_idx, 0, height - 1)

            grid_sum = np.zeros((height, width), dtype=np.float32)
            grid_count = np.zeros((height, width), dtype=np.int32)
            flat_idx = y_idx * width + x_idx

            np.add.at(grid_sum.ravel(), flat_idx, z)
            np.add.at(grid_count.ravel(), flat_idx, 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                grid_vals = grid_sum / grid_count
                grid_vals[grid_count == 0] = np.nan

            coords_x = min_x + (np.arange(width) + 0.5) * resolution
            coords_y = min_y + (np.arange(height) + 0.5) * resolution

            da = xr.DataArray(
                grid_vals,
                coords={"y": coords_y, "x": coords_x},
                dims=("y", "x"),
                name="elevation",
            )

            # Explicitly write transform (Bottom-Up Grid)
            transform = Affine.translation(min_x, min_y) * Affine.scale(resolution, resolution)
            da.rio.write_transform(transform, inplace=True)
            da.rio.set_spatial_dims("x", "y", inplace=True)

            # Assign CRS
            if native_crs_str:
                da.rio.write_crs(native_crs_str, inplace=True)
            else:
                try:
                    native_crs = las.header.parse_crs()
                    if native_crs:
                        da.rio.write_crs(native_crs, inplace=True)
                except Exception:
                    pass

            if target_crs and da.rio.crs and da.rio.crs != target_crs:
                da = da.rio.reproject(target_crs)

            return da

        except Exception as e:
            logger.error(f"Lidar Read Error: {e}", exc_info=True)
            return None

    def fetch_lidar_from_laz(
        self,
        s3_url: str,
        resolution: float = 4.0,  # Meters (approx, if projected)
        target_crs: str = "EPSG:4326",
    ) -> xr.DataArray | None:
        """
        Fetches a specific LAZ file, filters Class 2, and rasterizes.

        Args:
            s3_url: S3 path (e.g. noaa-nos-coastal-lidar-pds/laz/geoid18/4938/...)
            resolution: Grid resolution (degrees if 4326, meters otherwise)
            target_crs: Desired output CRS.

        Returns:
            xr.DataArray: Rasterized elevation.
        """
        try:
            filename = Path(s3_url).name
            local_path = self.cache_dir / filename

            # 1. Download (if not cached)
            if not local_path.exists():
                if self.offline_mode:
                    logger.warning(f"Offline Mode: Missing Lidar file {filename}")
                    return None

                logger.info(f"Downloading {s3_url} to {local_path}...")
                self.fs.get(s3_url, str(local_path))

            # 2. Read
            return self._read_laz_file(local_path, resolution=resolution, target_crs=target_crs)

        except Exception as e:
            logger.error(f"Lidar Fetch Error: {e}", exc_info=True)
            return None

    def fetch_lidar_from_ept(
        self,
        ept_url: str,
        bounds: tuple[float, float, float, float],
        resolution: float = 4.0,
        target_crs: str = "EPSG:4326",
    ) -> xr.DataArray | None:
        """
        Fetches Lidar data from an Entwine Point Tile (EPT) source using PDAL.

        Args:
            ept_url: URL to ept.json (e.g., https://.../ept.json)
            bounds: Tuple (minx, min_y, max_x, max_y) in the EPT's native CRS (usually).
                    Wait, EPT readers typically want bounds in native CRS.
                    For NOAA/USGS, this is crucial.
            resolution: Grid resolution in native units.
            target_crs: Desired output CRS.

        Returns:
            xr.DataArray: Rasterized elevation.
        """
        import json
        import tempfile

        import pdal

        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                output_filename = tmp.name

            # Construct PDAL Pipeline
            # bounds format for EPT reader: "([xmin, xmax], [ymin, ymax])"
            minx, miny, maxx, maxy = bounds
            pdal_bounds = f"([{minx}, {maxx}], [{miny}, {maxy}])"

            pipeline_config = {
                "pipeline": [
                    {
                        "type": "readers.ept",
                        "filename": ept_url,
                        "bounds": pdal_bounds,
                        "tag": "reader",
                    },
                    {
                        "type": "filters.range",
                        "limits": "Classification[2:2]",  # Bare Earth only
                        "tag": "filter",
                    },
                    {
                        "type": "writers.gdal",
                        "filename": output_filename,
                        "bounds": pdal_bounds,
                        "resolution": resolution,
                        "output_type": "mean",  # Grid method
                        "data_type": "float32",
                        "nodata": -9999.0,
                    },
                ]
            }

            # Execute PDAL
            logger.info(f"Executing PDAL Pipeline for {ept_url}...")
            # print(json.dumps(pipeline_config, indent=2))
            pipeline = pdal.Pipeline(json.dumps(pipeline_config))
            count = pipeline.execute()

            logger.info(f"PDAL executed. Points processed: {count}")

            # Load Result
            if Path(output_filename).exists():
                da_raw = rxr.open_rasterio(output_filename, masked=True)
                da: xr.DataArray
                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = da_raw.to_array().isel(variable=0)
                else:
                    da = cast(xr.DataArray, da_raw)

                da = da.rename({"band": "variable"}).squeeze("variable")
                da.name = "elevation"
                da.attrs.pop("long_name", None)

                # EPT is almost always 3857 (Web Mercator).
                # If CRS is missing from the GDAL output, assume 3857.
                if da.rio.crs is None:
                    da.rio.write_crs("EPSG:3857", inplace=True)

                # Reproject if needed
                if target_crs and da.rio.crs != target_crs:
                    # e.g., if EPT is Web Mercator (3857) but we want 4326
                    da = da.rio.reproject(target_crs)

                # Cleanup
                Path(output_filename).unlink()
                return da
            else:
                logger.error("PDAL failed to produce output file.")
                return None

        except ImportError:
            logger.error("pda/python-pdal not installed. Please install via conda/micromamba.")
            return None
        except Exception as e:
            logger.error(f"EPT Fetch Error: {e}", exc_info=True)
            if "output_filename" in locals() and Path(output_filename).exists():
                Path(output_filename).unlink()
            return None

    def fetch_lidar_from_stac(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float = 4.0,
        target_crs: str = "EPSG:4326",
        force_cache: bool = True,  # Now means "Cache in background if not present"
    ) -> xr.DataArray | None:
        """
        Fetches Lidar from Microsoft Planetary Computer STAC API (3DEP COPC).
        """
        import json
        import tempfile
        import threading

        import pdal

        # MPC STAC Endpoint
        try:
            # Call cached query function
            bbox_tuple = tuple(bounds)
            result = None

            # 1. Offline Mode / Manifest Lookup
            if self.offline_mode:
                logger.debug(f"Offline Mode: Checking Manifest for Lidar in {bounds}")
                manifest_items = self.manifest.find_items("3dep-lidar-copc", bounds)
                if manifest_items:
                    m_item = manifest_items[0]
                    result = {
                        "href": m_item["href"],
                        "native_epsg": m_item.get("properties", {}).get("native_epsg"),
                        "projjson": m_item.get("properties", {}).get("projjson"),
                        "id": "manifest-item",
                    }
                    logger.info(f"Offline Manifest found Lidar asset: {result['href']}")

            # 2. Online Mode
            if not result and not self.offline_mode:
                result = _query_3dep_stac(bbox_tuple)
                if result:
                    self.manifest.add_item(
                        collection_id="3dep-lidar-copc",
                        bbox=result.get("bbox", bbox_tuple),
                        asset_href=result["href"],
                        properties=result.get("properties") or {},
                    )

            if not result:
                if self.offline_mode:
                    logger.warning("Offline Mode: No Lidar coverage found in manifest.")
                else:
                    logger.warning(
                        f"No 3DEP COPC Lidar found in Planetary Computer STAC for bbox={bounds}. "
                        "Region may be covered by legacy datasets not yet indexed as COPC."
                    )
                return None

            href = result["href"]
            native_epsg = result["native_epsg"]

            # --- CACHING STRATEGY (Hybrid) ---
            local_path = self._get_cache_path(href)

            # Case A: Already Cached -> Use Local File (Fast/Offline)
            if local_path.exists():
                logger.info(f"Lidar Cache Hit: {local_path}")
                native_crs_str = f"EPSG:{native_epsg}" if native_epsg else None
                return self._read_laz_file(local_path, bounds, resolution, target_crs, native_crs_str)

            # Case B: Offline Mode + Not Cached -> Fail
            if self.offline_mode:
                logger.warning(f"Offline Mode: Missing Lidar file {local_path.name}")
                return None

            # Case C: Online + Not Cached -> Stream + Background Download
            logger.info(f"Streaming COPC Asset: {href}")

            if force_cache:
                logger.info("Spawning background thread to cache asset.")
                t = threading.Thread(target=self._download_and_cache, args=(href,))
                t.daemon = True  # Daemonize to not block exit, though download might be interrupted
                t.start()

            # --- START STREAMING (PDAL) ---
            if native_epsg:
                logger.debug(f"STAC Item Properties found. Native EPSG: {native_epsg}")

            reader_bounds = None

            if native_epsg:
                # Reproject bounds to native for efficient reading
                from pyproj import Transformer

                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{native_epsg}", always_xy=True)

                xs = [bounds[0], bounds[0], bounds[2], bounds[2]]
                ys = [bounds[1], bounds[3], bounds[1], bounds[3]]

                tx, ty = transformer.transform(xs, ys)

                minx, maxx = min(tx), max(tx)
                miny, maxy = min(ty), max(ty)

                # Check dimensions before PDAL execution to prevent OOM
                width_est = (maxx - minx) / resolution
                height_est = (maxy - miny) / resolution
                logger.debug(
                    f"STAC Native Bounds: X[{minx:.2f}, {maxx:.2f}] Y[{miny:.2f}, {maxy:.2f}] "
                    f"Res={resolution} -> Grid: {int(width_est)}x{int(height_est)}"
                )

                if width_est > 20_000 or height_est > 20_000:
                    logger.warning(
                        f"Estimated Lidar Grid size too large ({int(width_est)}x{int(height_est)}). "
                        "Aborting STAC fetch for this tile to prevent PDAL crash."
                    )
                    return None

                if width_est <= 0 or height_est <= 0:
                    logger.warning(
                        f"Estimated Lidar Grid size invalid ({width_est}x{height_est}). "
                        "Calculated bounds may be effectively zero."
                    )
                    return None

                reader_bounds = f"([{minx}, {maxx}], [{miny}, {maxy}])"
            else:
                # Check PROJJSON if native_epsg was missing
                pjson = result.get("projjson", {})
                if "components" in pjson:
                    for comp in pjson["components"]:
                        if comp.get("type") == "ProjectedCRS" and "id" in comp:
                            native_epsg = comp["id"].get("code")
                            break

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                output_filename = tmp.name

            pipeline_config = {
                "pipeline": [
                    {
                        "type": "readers.copc",
                        "filename": href,
                        "tag": "reader",
                    },
                    {
                        "type": "filters.range",
                        "limits": "Classification[2:2]",  # Bare Earth
                        "tag": "filter",
                    },
                    {
                        "type": "writers.gdal",
                        "filename": output_filename,
                        "resolution": resolution,
                        "output_type": "mean",
                        "data_type": "float32",
                        "nodata": -9999.0,
                    },
                ]
            }

            if reader_bounds:
                pipeline_config["pipeline"][0]["bounds"] = reader_bounds
                # Also force writer bounds
                pipeline_config["pipeline"][2]["bounds"] = reader_bounds

            # Execute
            pipeline = pdal.Pipeline(json.dumps(pipeline_config))
            count = pipeline.execute()
            logger.debug(f"PDAL executed. Points: {count}")

            if Path(output_filename).exists():
                with rxr.open_rasterio(output_filename, masked=True) as da_raw:  # type: ignore
                    if isinstance(da_raw, list):
                        da = da_raw[0]
                    elif isinstance(da_raw, xr.Dataset):
                        da = da_raw.to_array().isel(variable=0)
                    else:
                        da = da_raw

                    from typing import cast

                    da = cast(xr.DataArray, da)

                    da = da.rename({"band": "variable"}).squeeze("variable")
                    da.name = "elevation"
                    da.attrs = {}
                    da.rio.write_nodata(-9999.0, inplace=True)

                    if native_epsg:
                        from contextlib import suppress

                        with suppress(Exception):
                            da.rio.write_crs(f"EPSG:{native_epsg}", inplace=True)

                    if target_crs and da.rio.crs and da.rio.crs != target_crs:
                        da = da.rio.reproject(target_crs)

                    da.load()

                Path(output_filename).unlink()
                # from typing import cast (already imported)

                return cast(xr.DataArray, da)

            return None

        except Exception as e:
            logger.error(f"STAC Fetch Error: {e}", exc_info=True)
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
        Defaults to STAC access for best coverage.
        """
        return self.fetch_lidar_from_stac(
            bounds=(west, south, east, north),
            resolution=4.0,  # Default to ~4m resolution (usually appropriate for 3DEP)
            target_crs="EPSG:4326",
        )
