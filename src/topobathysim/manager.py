import logging
from pathlib import Path

import xarray as xr

from .fusion import FusionEngine
from .gebco_2025 import GEBCO2025Provider
from .ncei_bag import BAGDiscovery, BAGProvider
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

        self.gebco: GEBCO2025Provider
        self.blue_topo: NoaaBlueTopoProvider | None = None
        self.topobathy: NoaaTopobathyProvider | None = None
        self.lidar: UsgsLidarProvider | None = None
        self.land: Usgs3DepProvider | None = None
        self.bag: BAGProvider | None = None

        # Initialize providers
        # GEBCO is always fallback
        # Note: GEBCO currently lacks offline_mode flag in its init, will ignoring for now or TODO
        self.gebco = GEBCO2025Provider(cache_dir=str(self.cache_dir))

        self.bag = BAGProvider(cache_dir=str(self.cache_dir))

        if use_blue_topo:
            self.blue_topo = NoaaBlueTopoProvider(cache_dir=str(self.cache_dir))

        if use_topobathy:
            self.topobathy = NoaaTopobathyProvider(cache_dir=str(self.cache_dir))

        if use_lidar:
            self.lidar = UsgsLidarProvider(cache_dir=str(self.cache_dir), offline_mode=offline_mode)

        if use_land:
            # Usgs3DepProvider usually just fetches, assuming we pass offline mode if supported
            # For now just init. Usgs3DepProvider doesn't take offline_mode yet?
            # Let's check Usgs3DepProvider init. It usually takes cache_dir.
            # We should update Usgs3DepProvider too if we want true offline support.
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
        if self.blue_topo and self.blue_topo.is_covered(lat, lon):
            # We assume if covered, we WOULD use it.
            # In a real scenario we'd check if specific tile exists.
            # For now we report based on coverage logic parity with get_elevation.
            # Note: fetching actual elevation is expensive so we might just report
            # 'Potential BlueTopo' vs 'GEBCO'.
            return {
                "source": "BlueTopo (NOAA)",
                "type": "High-Res Survey",
                "quality_tier": "Tier 1 (Measured)",
            }

        # GEBCO Logic
        # We need to sample TID to be precise, or just report GEBCO
        return {
            "source": "GEBCO 2025",
            "type": "Global Grid",
            "quality_tier": "Variable (Check Metadata)",
        }

    def fetch_global_context(self, north: float, south: float, west: float, east: float) -> None:
        """
        Pre-loads GEBCO data for the region.
        BlueTopo is loaded on-demand/tiled, so no global fetch needed there yet.
        """
        # Update GEBCO bounds and fetch
        # Note: Modifying the single GEBCO instance's bounds might be side-effect heavy if reused.
        # But GEBCO2025 design inherits from Topography which is stateful.
        # We re-instantiate or explicitly set properties if supported.
        # Topography usually expects init params.
        # For this agent/PoC, we create a new instance or assume the manager handles a 'primary' AOI.
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
        Prioritizes BlueTopo.
        """
        import numpy as np

        # 0. Fetch Topobathy Lidar (Tier 0 - Intertidal/Land-Sea)
        topobathy_da = None
        if self.topobathy:
            # Try to fetch topobathy logic
            try:
                topobathy_da = self.topobathy.get_grid(west, south, east, north)
            except Exception as e:
                logger.warning(f"Topobathy Lidar Fetch Failed: {e}")

        # 1. Try BAG Bridge (NCEI Deep Sea Precision)
        # Attempt to resolve survey ID at center point or corners
        # For simplicity, check center
        bag_da = None
        if self.blue_topo and self.bag:
            center_lat = (south + north) / 2
            center_lon = (west + east) / 2
            survey_id = self.blue_topo.get_source_survey_id(center_lat, center_lon)
            if survey_id:
                logger.info(f"BAG Survey Detected: {survey_id}. Attempting high-fidelity fetch.")
                bag_da = self.bag.fetch_bag(survey_id)

                if bag_da is None:
                    logger.info("ID-based fetch failed. Attempting spatial fallback...")
                    bag_url = BAGDiscovery.find_bag_by_location(center_lat, center_lon)
                    if bag_url:
                        logger.info(f"Spatial fallback successful: {bag_url}")
                        bag_da = self.bag.fetch_bag(survey_id, download_url=bag_url)

                # Ensure CRS is 4326 for consistency downstream?
                # NO! Reprojecting the full 1.8GB BAG to 4326 here is expensive and unncesary.
                # main.py does 'reproject_match' which warping from Source->Target CRS automatically.
                # So we return the Native (UTM) BAG.
                # if bag_da is not None and bag_da.rio.crs and bag_da.rio.crs != "EPSG:4326":
                #    try:
                #        bag_da = bag_da.rio.reproject("EPSG:4326")
                #    except Exception as e:
                #        logger.warning(f"BAG Reprojection Failed: {e}")

        # 1.5 Try BlueTopo
        # Check for intersecting tiles
        # We always attempt to fetch BlueTopo now, even if BAG exists,
        # to use it as a "Gap Fill" (Background) layer.
        blue_topo_da = None
        base_da = None

        # Track effective source for base_da
        # base_sid = SOURCE_ID_GEBCO

        if self.blue_topo:
            logger.info("Attempting to resolve BlueTopo tiles.")
            tile_ids = self.blue_topo.resolve_tiles_in_bbox(west, south, east, north)
            if tile_ids:
                logger.info(f"Found {len(tile_ids)} BlueTopo tiles.")
                # Load all tiles
                das = []
                for tid in tile_ids:
                    # logger.info(f"Loading BlueTopo tile: {tid}")
                    # Reduce log spam
                    d = self.blue_topo.load_tile_as_da(tid, (west, south, east, north))
                    if d is not None:
                        das.append(d)

                if das:
                    if len(das) == 1:
                        d = das[0]
                        if not d.rio.crs:
                            d.rio.write_crs("EPSG:4326", inplace=True)
                        blue_topo_da = d
                        logger.info("BlueTopo single tile loaded.")
                    else:
                        try:
                            from rioxarray.merge import merge_arrays

                            logger.info(f"Merging {len(das)} BlueTopo tiles.")
                            merged_da = merge_arrays(das)
                            if not merged_da.rio.crs:
                                merged_da.rio.write_crs("EPSG:4326", inplace=True)
                            blue_topo_da = merged_da
                            logger.info("BlueTopo tiles merged successfully.")
                        except Exception as e:
                            logger.warning(f"BlueTopo Merge Failed: {e}")
            else:
                logger.info("No BlueTopo tiles found for the bbox.")

        # Construct Base DA Logic + Source Mask Logic
        source_da = None

        if bag_da is not None:
            # GAP FILLING STRATEGY
            if blue_topo_da is not None:
                logger.info("Fusing BAG (Priority) with BlueTopo (Background Fill)...")
                try:
                    from rasterio.enums import Resampling

                    bg_fill = blue_topo_da.rio.reproject_match(bag_da, resampling=Resampling.nearest)
                    base_da = bag_da.combine_first(bg_fill)
                    base_da.attrs["source"] = f"Fused: {bag_da.attrs.get('survey_source', 'BAG')} + BlueTopo"

                    if return_source_mask:
                        # Construct mask
                        # Default is BAG (10)
                        source_da = xr.full_like(base_da, SOURCE_ID_BAG)
                        # Where BAG is NaN, we used BlueTopo (40)
                        # Identify where bag_da was NaN but bg_fill was Valid
                        mask_fill = np.isnan(bag_da) & ~np.isnan(bg_fill)
                        source_da = source_da.where(~mask_fill, SOURCE_ID_BLUETOPO)
                        # Where both are NaN is implicitly BAG ID but valid check later will handle it?
                        # Or set to 0?
                        # Let's update NaNs to 0 (Canvas)
                        source_da = source_da.where(~np.isnan(base_da), SOURCE_ID_CANVAS)

                except Exception as e:
                    logger.error(f"Gap Fill Fusion Failed: {e}")
                    base_da = bag_da
                    if return_source_mask:
                        source_da = xr.full_like(base_da, SOURCE_ID_BAG).where(
                            ~np.isnan(base_da), SOURCE_ID_CANVAS
                        )
            else:
                base_da = bag_da
                if return_source_mask:
                    source_da = xr.full_like(base_da, SOURCE_ID_BAG).where(
                        ~np.isnan(base_da), SOURCE_ID_CANVAS
                    )

            logger.info("Manager returning Bathy Grid (Base DA) from BAG data (possibly fused).")
            # We defer return to handle source tracking return structure

        # 1.6 Use BlueTopo as Base if no BAG
        if base_da is None and blue_topo_da is not None:
            base_da = blue_topo_da
            if return_source_mask:
                source_da = xr.full_like(base_da, SOURCE_ID_BLUETOPO).where(
                    ~np.isnan(base_da), SOURCE_ID_CANVAS
                )

        if base_da is None:
            logger.info("No BlueTopo or BAG data found. Falling back to GEBCO 2025.")
            g = GEBCO2025Provider(south=south, north=north, west=west, east=east)
            da = g.fetch()
            if "lat" in da.coords:
                da = da.rename({"lat": "y", "lon": "x"})

            # Ensure CRS for rioxarray operations
            if da.rio.crs is None:
                # Force 4326 for GEBCO
                try:
                    da.rio.write_crs("EPSG:4326", inplace=True)
                except Exception:
                    da.rio.write_crs("EPSG:4326", inplace=True)
            base_da = da

            if return_source_mask:
                source_da = xr.full_like(base_da, SOURCE_ID_GEBCO)

        # 3. Fusion: Topobathy > Base
        if topobathy_da is not None and base_da is not None:
            # Ensure Topobathy is in same CRS as Base
            if topobathy_da.rio.crs != base_da.rio.crs:
                try:
                    topobathy_da = topobathy_da.rio.reproject_match(base_da)
                except Exception as e:
                    logger.warning(f"Reprojection failed for Topobathy Lidar: {e}")
                    # Only return base_da if failure
                    # We need to handle source mask return structure
                    if return_source_mask:
                        return base_da, source_da
                    return base_da

            engine = FusionEngine()
            try:
                from typing import cast

                # We need to intersect logic to update source mask
                # If fusion happens, it's confusing.
                # Simple logic for source mask:
                # If topobathy is valid, it overwrites base_da?
                # FusionEngine 'fuse_seamline' does blending.
                # Let's say: if weight > 0.5 -> Topobathy(21), else Base(XX).
                # This requires access to the weight mask.
                # For now, simplistic approach: "Fused" (22) everywhere they overlap?
                # Fused source ID = 22

                fused = engine.fuse_seamline(lidar_da=cast(xr.DataArray, topobathy_da), bathy_da=base_da)

                if return_source_mask:
                    # Update source mask to reflect 'Fused' or 'Topobathy'
                    # Align source mask to fused result (which usually matches bathy_da geometry)
                    # Note: fuse_seamline returns matching bathy_da geometry

                    if source_da is not None:
                        # Overwrite with Topobathy ID where Topobathy valid?
                        # Or verify overlap.
                        valid_topo = ~np.isnan(cast(xr.DataArray, topobathy_da))

                        # Ideally we know the 'weight' but we don't here.
                        if source_da.shape != fused.shape:
                            # Should match base_da which matches fused
                            pass

                        source_da = source_da.where(~valid_topo, SOURCE_ID_TOPOBATHY)
                        # This implies Topobathy dominance.
                    else:
                        # Should exist if base_da exists
                        source_da = xr.full_like(fused, SOURCE_ID_TOPOBATHY)  # fallback

                if return_source_mask:
                    return fused, source_da
                return fused
            except Exception as e:
                logger.error(f"Fusion Failed: {e}")
                if return_source_mask:
                    return base_da, source_da
                return base_da

        if return_source_mask:
            return base_da, source_da
        return base_da
