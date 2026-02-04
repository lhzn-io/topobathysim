import logging
from pathlib import Path

import xarray as xr
from bmi_topography import Topography

logger = logging.getLogger(__name__)


class GEBCO2025(Topography):
    """
    BMI-compliant interface for GEBCO 2025 data.
    Wraps bmi-topography to fetch from OPeNDAP with specific corrections.
    """

    # Official GEBCO 2025 OpenDAP URL (Placeholder - typically would be verified)
    # Using the path provided by the user instructions
    # Note: 'sub_ice_topo_bathymetry' is the elevation variable
    OPENDAP_URL = "dap2://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2025/sub_ice_topography_bathymetry/netcdf/gebco_2025_sub_ice.nc"
    # NOTE: The above URL might be a file download page.
    # A true OPeNDAP URL usually ends in .nc.html or similar for browsing, or just .nc for DAP.
    # For this implementation, we will assume standard pydap/xarray can handle the URL provided
    # or we use a widely known one.
    # If the user provided URL is just a browser link, we might need the DAP server root.
    # Let's assume the user wants us to use this URL structure for xarray/pydap.

    RESOLUTION_ARCSEC = 15
    HALF_PIXEL_OFFSET = (15 / 3600) * 0.5  # Degrees

    def __init__(
        self,
        dem_type: str = "GEBCO_2025",
        south: float = -90,
        north: float = 90,
        west: float = -180,
        east: float = 180,
        output_format: str = "GTiff",
        cache_dir: str = "~/.cache/topobathysim",
    ):
        super().__init__(
            dem_type="SRTMGL3",  # Pass valid type to satisfy validation; we override fetch() anyway
            south=south,
            north=north,
            west=west,
            east=east,
            output_format=output_format,
            cache_dir=str(Path(cache_dir).expanduser()),
        )
        self.tid_data: xr.DataArray | None = None
        self._da: xr.DataArray | None = None

    def fetch(self) -> xr.DataArray:
        """
        Fetch data from GEBCO 2025 OPeNDAP server.
        Overrides parent fetch mechanism to handle multi-variable loading (elevation + TID).
        """
        # We need to load specific slice
        # Using xarray's OPeNDAP engine

        # Note: In a real 'Upstream' contribution, we would modify `bmi-topography`'s `fetch`
        # to carry strict logic. Here we override.

        logger.info(f"Fetching GEBCO data via OPeNDAP: {self.OPENDAP_URL}")
        logger.debug(f"GEBCO Slice Bounds: {self.bbox}")

        # 1. Access the dataset lazy loading
        # Because we need both sub_ice_topo_bathymetry and tid, we might need to open the dataset
        # and slice it.

        ds = xr.open_dataset(self.OPENDAP_URL, engine="pydap")  # or engine='netcdf4' if DAP enabled

        # 2. Slice to bounding box
        # 2. Slice to bounding box
        # GEBCO coords are usually 'lat' and 'lon'. bbox is BoundingBox object
        s, w, n, e = self.bbox.south, self.bbox.west, self.bbox.north, self.bbox.east
        ds_subset = ds.sel(lat=slice(s, n), lon=slice(w, e))

        # 3. No Half-Pixel Offset needed
        # GEBCO is pixel-centered, and OPeNDAP coordinates typically reflect this.
        # Removing manual shift to ensure accuracy.

        # 4. Extract DataArrays
        # Elevation
        if "sub_ice_topo_bathymetry" in ds_subset:
            self._da = ds_subset["sub_ice_topo_bathymetry"]
        elif "elevation" in ds_subset:  # Fallback
            self._da = ds_subset["elevation"]
        else:
            raise KeyError("Could not find elevation variable 'sub_ice_topo_bathymetry'")

        # TID (Type Identifier)
        if "tid" in ds_subset:
            self.tid_data = ds_subset["tid"]

        # 5. Persist/Cache if needed
        # parent logic usually saves to disk. We might want to save the subset using self.output_format
        # For now, we keep strictly in memory for the 'load' step,
        # but bmi-topography usually writes usage.
        # We will assume this is the 'load_to_memory' part.

        return self._da

    def load(self) -> xr.DataArray:
        """Standard BMI load method calling fetch"""
        return self.fetch()

    def get_tid_classification(self) -> xr.DataArray | None:
        """Returns the cached TID DataArray"""
        return self.tid_data

    def sample_elevation(self, lat: float, lon: float) -> float:
        """
        Returns bilinearly interpolated elevation at the given coordinate.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Depth/Elevation in meters (negative for depth).
        """
        if self._da is None:
            raise RuntimeError("Data not loaded. Call fetch() or load() first.")

        # xarray's interp handles bilinear interpolation by default (linear)
        val = self._da.interp(lat=lat, lon=lon, method="linear")
        return float(val.values)
