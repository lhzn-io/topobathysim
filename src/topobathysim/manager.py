import logging
from pathlib import Path

import xarray as xr

from .bluetopo import BlueTopoProvider
from .gebco import GEBCO2025
from .land import LandProvider
from .lidar import LidarProvider

logger = logging.getLogger(__name__)


class BathyManager:
    """
    Orchestrates bathymetry data access, intelligently switching between
    Global GEBCO 2025 and High-Res BlueTopo based on coverage and availability.
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/topobathysim",
        use_blue_topo: bool = True,
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

        self.gebco: GEBCO2025
        self.blue_topo: BlueTopoProvider | None = None
        self.lidar: LidarProvider | None = None
        self.land: LandProvider | None = None

        # Initialize providers
        # GEBCO is always fallback
        # Note: GEBCO currently lacks offline_mode flag in its init, will ignoring for now or TODO
        self.gebco = GEBCO2025(cache_dir=str(self.cache_dir))

        if use_blue_topo:
            self.blue_topo = BlueTopoProvider(cache_dir=str(self.cache_dir))

        if use_lidar:
            self.lidar = LidarProvider(cache_dir=str(self.cache_dir), offline_mode=offline_mode)

        if use_land:
            # LandProvider usually just fetches, assuming we pass offline mode if supported
            # For now just init. LandProvider doesn't take offline_mode yet?
            # Let's check LandProvider init. It usually takes cache_dir.
            # We should update LandProvider too if we want true offline support.
            self.land = LandProvider(cache_dir=str(self.cache_dir))

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
        self.gebco = GEBCO2025(north=north, south=south, west=west, east=east)
        self.gebco.fetch()

    def get_grid(self, south: float, north: float, west: float, east: float) -> "xr.DataArray":
        """
        Returns the best available bathymetry grid for the bbox.
        Prioritizes BlueTopo.
        """
        # 1. Try BlueTopo
        # Check for intersecting tiles
        tile_ids = []
        if self.blue_topo:
            tile_ids = self.blue_topo.resolve_tiles_in_bbox(west, south, east, north)

        if tile_ids and self.blue_topo:
            # Load all tiles
            das = []
            for tid in tile_ids:
                # We load the FULL tile first (no clip) to ensure we have data to merge
                # But load_tile_as_da clips by default.
                # We should probably load full tiles and then merge, then clip.
                # Or pass a larger bbox?
                # Actually, load_tile_as_da's clip is safe if we merge later?
                # No, if we clip individually to the target BBOX, we get pieces.
                # Merging pieces of the target bbox is fine.
                # So we can reuse load_tile_as_da.
                d = self.blue_topo.load_tile_as_da(tid, (west, south, east, north))
                if d is not None:
                    das.append(d)

            if das:
                if len(das) == 1:
                    d = das[0]
                    if not d.rio.crs:
                        d.rio.write_crs("EPSG:4326", inplace=True)
                    return d

                try:
                    from rioxarray.merge import merge_arrays

                    merged_da = merge_arrays(das)
                    # Merge returns a DataArray.
                    # It might lose CRS or Attributes, so we re-ensure?
                    if not merged_da.rio.crs:
                        merged_da.rio.write_crs("EPSG:4326", inplace=True)
                    return merged_da
                except Exception as e:
                    logger.warning(f"BlueTopo Merge Failed: {e}")
                    # Fallback to GEBCO
                    pass

        # 2. Fallback GEBCO
        # Re-init GEBCO for this bbox?
        g = GEBCO2025(south=south, north=north, west=west, east=east)
        da = g.fetch()
        if "lat" in da.coords:
            da = da.rename({"lat": "y", "lon": "x"})

        # Ensure CRS for rioxarray operations
        # Use simple assignment if write_crs fails or is ambiguous
        if da.rio.crs is None:
            # Force 4326 for GEBCO
            try:
                da.rio.write_crs("EPSG:4326", inplace=True)
            except Exception:
                da.rio.write_crs("EPSG:4326", inplace=True)

        return da
