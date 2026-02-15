import logging
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays

from .gebco_2025 import GEBCO2025Provider
from .ncei_bag import BAGDiscovery, BAGProvider
from .ncei_cudem import CUDEMProvider
from .noaa_bluetopo import NoaaBlueTopoProvider
from .noaa_topobathy import NoaaTopobathyProvider
from .usgs_3dep import Usgs3DepProvider
from .usgs_lidar import UsgsLidarProvider

logger = logging.getLogger(__name__)


SOURCE_ID_BAG = 10
SOURCE_ID_LIDAR = 20
SOURCE_ID_TOPOBATHY = 21
SOURCE_ID_FUSION = 22
SOURCE_ID_CUDEM = 30
SOURCE_ID_BLUETOPO = 40
SOURCE_ID_LAND = 50
SOURCE_ID_GEBCO = 60
SOURCE_ID_CANVAS = 0


class BathyManager:
    """
    Orchestrates bathymetry data access, intelligently switching between
    Global GEBCO 2025 and High-Res BlueTopo based on coverage and availability.
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/topobathysim",
        use_blue_topo: bool = True,
        use_cudem: bool = True,
        use_topobathy: bool = True,
        use_lidar: bool = True,
        use_land: bool = True,
        offline_mode: bool = False,
    ):
        """
        Args:
            cache_dir: Directory for storing tiles/data
            use_blue_topo: Enable BlueTopo (High Res Bathy) (Tier 1)
            use_lidar: Enable 3DEP Lidar (High Res Land) (Tier 2)
            use_land: Enable 3DEP/NASADEM Land (Mid Res Land) (Tier 3)
            offline_mode: If True, fail if network request is required. Use only cached data.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode
        self.use_blue_topo = use_blue_topo
        self.use_lidar = use_lidar
        self.use_land = use_land
        self.use_cudem = use_cudem

        self.gebco: GEBCO2025Provider
        self.blue_topo: NoaaBlueTopoProvider | None = None
        self.topobathy: NoaaTopobathyProvider | None = None
        self.lidar: UsgsLidarProvider | None = None
        self.land: Usgs3DepProvider | None = None
        self.bag: BAGProvider | None = None
        self.cudem: CUDEMProvider | None = None

        # Initialize providers
        # GEBCO is always fallback
        self.gebco = GEBCO2025Provider(cache_dir=str(self.cache_dir))

        self.bag = BAGProvider(cache_dir=str(self.cache_dir))

        if use_cudem:
            self.cudem = CUDEMProvider(cache_dir=str(self.cache_dir))

        if use_blue_topo:
            self.blue_topo = NoaaBlueTopoProvider(cache_dir=str(self.cache_dir))

        if use_topobathy:
            self.topobathy = NoaaTopobathyProvider(cache_dir=str(self.cache_dir))

        if use_lidar:
            self.lidar = UsgsLidarProvider(cache_dir=str(self.cache_dir), offline_mode=offline_mode)

        if use_land:
            self.land = Usgs3DepProvider(cache_dir=str(self.cache_dir))

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Returns the best available elevation for the coordinate.
        Prioritizes BlueTopo if covered and fetchable. Fallback to GEBCO.
        """
        # 1. Try BlueTopo (High Res, US Waters)
        if self.blue_topo and self.blue_topo.is_covered(lat, lon):
            topo_val = self.blue_topo.fetch_elevation(lat, lon)
            if topo_val is not None:
                return topo_val

        # 2. Fallback to GEBCO 2025 (Global)
        return self.gebco.sample_elevation(lat, lon)

    def get_source_info(self, lat: float, lon: float) -> dict[str, str]:
        """
        Returns metadata about the active data source for the coordinate.
        """
        # Tier 0-1: Check CUDEM (Gap Fill / Coastal)
        # Priority over BlueTopo in coastal/gap zones.
        if self.cudem:
            bbox_matches = self.cudem.resolve_tiles(lon, lat, lon, lat)
            if not bbox_matches.empty:
                return {
                    "source": "CUDEM (NOAA/NCEI)",
                    "type": "Integrated Topobathy Model",
                    "quality_tier": "Tier 1 (Indirect/Gap Fill)",
                }

        # Tier 2: BlueTopo
        if self.blue_topo and self.blue_topo.is_covered(lat, lon):
            return {
                "source": "BlueTopo (NOAA)",
                "type": "High-Res Survey",
                "quality_tier": "Tier 1 (Measured)",
            }

        # GEBCO Logic
        return {
            "source": "GEBCO 2025",
            "type": "Global Grid",
            "quality_tier": "Variable (Check Metadata)",
        }

    def fetch_global_context(self, north: float, south: float, west: float, east: float) -> None:
        """
        Pre-loads GEBCO data for the region.
        """
        self.gebco = GEBCO2025Provider(north=north, south=south, west=west, east=east)
        self.gebco.fetch()

    def get_grid(
        self,
        south: float,
        north: float,
        west: float,
        east: float,
        target_shape: tuple[int, int] | None = None,
        return_source_mask: bool = False,
    ) -> "xr.DataArray | tuple[xr.DataArray, xr.DataArray | None]":
        """
        Returns the best available bathymetry grid for the bbox.
        Strategy: Composition from Tiered Layers.
        """
        logger.info(f"Requesting Grid: S={south}, N={north}, W={west}, E={east}, Shape={target_shape}")

        # 1. Collect all potential layers
        valid_layers: list[tuple[float, str, xr.DataArray]] = []

        # Tier 0: BAG (Absolute Priority)
        if self.bag:
            try:
                # Resolve ID
                center_lat, center_lon = (south + north) / 2, (west + east) / 2

                # Method A: Via BlueTopo Metadata (Survey ID)
                survey_id = (
                    self.blue_topo.get_source_survey_id(center_lat, center_lon) if self.blue_topo else None
                )
                if survey_id:
                    da = self.bag.fetch_bag(survey_id)
                    if da is not None:
                        valid_layers.append((0, "BAG", da))

                # Method B: Direct Spatial Search (BBox)
                bag_urls = BAGDiscovery.find_bags_by_bbox(west, south, east, north)

                if bag_urls:
                    bbox_das = []
                    for url in bag_urls:
                        d = self.bag.fetch_bag("unknown", download_url=url)
                        if d is not None:
                            bbox_das.append(d)

                    if bbox_das:
                        try:
                            # Use rioxarray to merge multiple BAGs if found
                            logger.debug("Merging multiple BAG arrays...")
                            merged_bag = merge_arrays(bbox_das)
                            logger.debug("Merged BAG arrays successfully.")

                            # Filter out existing BAG entry if any (prioritize merged)
                            valid_layers = [v for v in valid_layers if v[1] != "BAG"]

                            valid_layers.append((0, "BAG", merged_bag))
                        except Exception as ex:
                            logger.warning(f"Failed to merge discovered BAGs: {ex}")
                            if not any(v[1] == "BAG" for v in valid_layers) and bbox_das:
                                valid_layers.append((0, "BAG", bbox_das[0]))

                # Legacy Fallback
                if not any(v[1] == "BAG" for v in valid_layers):
                    # We need a URL here, assume None is handled
                    bag_url: str | None = BAGDiscovery.find_bag_by_location(center_lat, center_lon)
                    if bag_url:
                        da = self.bag.fetch_bag("unknown", download_url=bag_url)
                        if da is not None:
                            valid_layers.append((0, "BAG", da))
            except Exception as e:
                logger.warning(f"BAG error: {e}")

        # Tier 1: Lidar/Topobathy
        lidar_da = None
        topo_da = None

        if self.lidar:
            try:
                lidar_da = self.lidar.get_grid(west, south, east, north)
            except Exception as e:
                logger.warning(f"Lidar error: {e}")

        if self.topobathy:
            try:
                topo_da = self.topobathy.get_grid(west, south, east, north)
            except Exception as e:
                logger.warning(f"Topobathy error: {e}")

        if lidar_da is not None and topo_da is not None:
            try:
                # Align and Fuse
                # Default to Lidar grid as reference (likely high res)
                if not lidar_da.rio.crs:
                    lidar_da.rio.write_crs("EPSG:4326", inplace=True)
                if not topo_da.rio.crs:
                    topo_da.rio.write_crs("EPSG:4326", inplace=True)

                logger.debug("Reprojecting/Matching Topobathy to Lidar grid...")
                topo_aligned = topo_da.rio.reproject_match(lidar_da, resampling=Resampling.bilinear)
                logger.debug("Reprojection complete.")

                # Condition: "Upland" defined as > 1.0m (User Threshold)
                upland_threshold = 1.0

                upland_mask = lidar_da > upland_threshold

                # 1. Start with Topo (The "Base" for this tier)
                fused = topo_aligned.copy()

                # 2. Overwrite with Lidar where Upland is detected
                fused = fused.where(~upland_mask, lidar_da)

                # 3. Fill gaps with High Res Lidar
                # Filter Lidar below threshold to prevent filling gaps with potential water noise.
                clean_lidar = lidar_da.where(lidar_da > upland_threshold)
                fused = fused.combine_first(clean_lidar)

                valid_layers.append((1, "FusedLidarTopo", fused))
                logger.info("Fused Lidar and Topobathy successfully")

            except Exception as e:
                logger.warning(f"Fusion of Lidar/Topo failed: {e}. using separate layers.")
                # If separate, we should still mask Lidar?
                if lidar_da is not None:
                    clean_lidar = lidar_da.where(lidar_da > 1.0)
                    valid_layers.append((1, "Lidar", clean_lidar))
                if topo_da is not None:
                    valid_layers.append((1, "Topobathy", topo_da))
        else:
            if lidar_da is not None:
                # Standalone Lidar: Mask Water Noise
                clean_lidar = lidar_da.where(lidar_da > 1.0)
                valid_layers.append((1, "Lidar", clean_lidar))
            if topo_da is not None:
                valid_layers.append((1, "Topobathy", topo_da))

        # Tier 2: BlueTopo (Prioritize over CUDEM for Bathy)
        if self.blue_topo:
            try:
                tile_ids = self.blue_topo.resolve_tiles_in_bbox(west, south, east, north)
                if tile_ids:
                    das = [
                        d
                        for tid in tile_ids
                        if (d := self.blue_topo.load_tile_as_da(tid, (west, south, east, north))) is not None
                    ]
                    if das:
                        da = merge_arrays(das)
                        valid_layers.append((2, "BlueTopo", da))
            except Exception as e:
                logger.warning(f"BlueTopo error: {e}")

        # Tier 3: CUDEM (Coastal Gap Fill)
        if self.cudem:
            try:
                da = self.cudem.get_grid(west, south, east, north)
                if da is not None:
                    valid_layers.append((3, "CUDEM", da))
            except Exception as e:
                logger.warning(f"CUDEM error: {e}")

        # Tier 3.5: Land (3DEP / NASADEM)
        if self.land:
            try:
                da = self.land.get_grid(west, south, east, north)
                if da is not None:
                    # Using priority 3.5 to ensure it fills gaps but respects Bathy sources (BlueTopo/CUDEM)
                    # unless it's Lidar (which is Tier 1, handled above).
                    # 'Land' here typically refers to lower-res DEMs if Lidar failed.
                    valid_layers.append((3.5, "Land", da))
            except Exception as e:
                logger.warning(f"Land error: {e}")

        # Tier 4: GEBCO (Always Fetch as last resort or context)
        try:
            g = GEBCO2025Provider(south=south, north=north, west=west, east=east)
            da = g.fetch()
            if "lat" in da.coords:
                da = da.rename({"lat": "y", "lon": "x"})
            valid_layers.append((4, "GEBCO", da))
        except Exception as e:
            logger.warning(f"GEBCO error: {e}")

        if not valid_layers:
            if return_source_mask:
                # Must satisfy Tuple[DataArray, DataArray | None] per annotation logic?
                # No, annotation says: "xr.DataArray | tuple[xr.DataArray, xr.DataArray | None]"
                # Checking None return is cleaner if we modify signature or cast.
                # Let's verify the signature.
                return cast(tuple[xr.DataArray, xr.DataArray | None], (None, None))

            return cast(xr.DataArray, None)

        # 2. Composition Strategy
        # Sort by Tier (0 is best: BAG)
        valid_layers.sort(key=lambda x: x[0])
        logger.info(f"Composing grid from {[x[1] for x in valid_layers]}")

        # Define the 'Base Canvas'
        base_da = None
        source_da = None

        if target_shape is not None:
            # Simple approach: Create linspace for centers
            height, width = target_shape
            lon_res = (east - west) / width
            lat_res = (north - south) / height

            # Centers
            xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=width)
            ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=height)

            base_da = xr.DataArray(np.nan, coords={"y": ys, "x": xs}, dims=("y", "x"))
            base_da.rio.set_spatial_dims("x", "y", inplace=True)
            base_da.rio.write_crs("EPSG:4326", inplace=True)
            transform = Affine.translation(west, north) * Affine.scale(lon_res, -lat_res)
            base_da.rio.write_transform(transform, inplace=True)

            base_da.attrs["source"] = "Canvas"

            if return_source_mask:
                source_da = xr.full_like(base_da, SOURCE_ID_CANVAS)

        else:
            # Legacy/Dynamic Mode (Fallback)
            # Find best resolution layer
            best_res_layer = valid_layers[0][2]

            try:
                if best_res_layer.rio.resolution():
                    res_x, res_y = best_res_layer.rio.resolution()
                    res_x, res_y = abs(res_x), abs(res_y)
                else:
                    res_x = res_y = 0.00005  # Default ~5m
            except Exception:
                res_x = res_y = 0.00005

            # Calc shape
            width = max(1, int((east - west) / res_x))
            height = max(1, int((north - south) / res_y))

            # Clamp potential OOM
            if width * height > 100_000_000:
                logger.warning(f"Requested grid too large ({width}x{height}), clamping resolution.")
                width, height = 2048, 2048
                lon_res = (east - west) / width
                lat_res = (north - south) / height
            else:
                lon_res = res_x
                lat_res = res_y

            xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=width)
            ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=height)

            base_da = xr.DataArray(np.nan, coords={"y": ys, "x": xs}, dims=("y", "x"))
            base_da.rio.set_spatial_dims("x", "y", inplace=True)
            base_da.rio.write_crs("EPSG:4326", inplace=True)
            transform = Affine.translation(west, north) * Affine.scale(lon_res, -lat_res)
            base_da.rio.write_transform(transform, inplace=True)

            base_da.attrs["source"] = "DynamicCanvas"

            if return_source_mask:
                source_da = xr.full_like(base_da, SOURCE_ID_CANVAS)

        # Apply helper
        def _ensure_crs(d: xr.DataArray) -> xr.DataArray:
            if not d.rio.crs:
                d.rio.write_crs("EPSG:4326", inplace=True)
            return d

        # Source ID Lookup
        def get_sid(tier: float, name: str) -> int:
            if name == "BAG":
                return SOURCE_ID_BAG
            if name == "Lidar":
                return SOURCE_ID_LIDAR
            if name == "Topobathy":
                return SOURCE_ID_TOPOBATHY
            if name == "FusedLidarTopo":
                return SOURCE_ID_FUSION
            if name == "CUDEM":
                return SOURCE_ID_CUDEM
            if name == "BlueTopo":
                return SOURCE_ID_BLUETOPO
            if name == "Land":
                return SOURCE_ID_LAND
            if name == "GEBCO":
                return SOURCE_ID_GEBCO
            return SOURCE_ID_GEBCO  # Default

        for tier, name, src_da in valid_layers:
            try:
                src_da = _ensure_crs(src_da)

                # Align Source to Canvas
                logger.debug(f"Reprojecting layer '{name}' to canvas match...")
                reproj_da = src_da.rio.reproject_match(base_da, resampling=Resampling.bilinear)
                logger.debug(f"Reprojected '{name}'.")

                # Update Source Mask
                if return_source_mask and source_da is not None:
                    # Identify pixels that WILL be filled by this layer
                    # i.e. base_da is NaN AND reproj_da is Valid
                    # combine_first fills gaps.

                    # 1. Valid data in new layer
                    new_valid = ~np.isnan(reproj_da)

                    # 2. Gap in existing canvas
                    curr_nan = np.isnan(base_da)

                    # 3. Pixels to update
                    update_mask = new_valid & curr_nan

                    sid = get_sid(tier, name)
                    source_da = source_da.where(~update_mask, sid)

                # Combine:
                # keep existing valid values in base_da (Higher tiers processed first!)
                # fill NaNs with new layer
                base_da = base_da.combine_first(reproj_da)
                base_da.attrs["source"] = (
                    (base_da.attrs.get("source", "") + f" + {name}") if "source" in base_da.attrs else name
                )

            except Exception as e:
                logger.warning(f"Failed to fill {name} into grid: {e}")

        if return_source_mask:
            return base_da, source_da

        return base_da
