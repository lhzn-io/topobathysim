import io
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from rasterio.enums import Resampling

from topobathysim.fusion import FusionEngine
from topobathysim.manager import BathyManager
from topobathysim.quality import source_report
from topobathysim.usgs_3dep import Usgs3DepProvider
from topobathysim.usgs_lidar import UsgsLidarProvider

from .models import ElevationResponse, TIDReportResponse

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("topobathyserve")
bathy_manager: BathyManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load .env (Resolve from topobathysim root)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Ensure bmi_topography finds the key
    if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
        os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

    global bathy_manager
    # Check for OFFLINE_MODE env var
    offline_mode = os.getenv("OFFLINE_MODE", "False").lower() in ("true", "1", "yes")
    bathy_manager = BathyManager(offline_mode=offline_mode)
    yield
    # clean up logic if needed


app = FastAPI(
    title="BathyServe",
    description="Microservice for Hybrid Bathymetry (GEBCO 2025 + BlueTopo)",
    version="0.2.0",
    lifespan=lifespan,
)

# Mount Viewer
static_dir = Path(__file__).parent.parent / "service" / "static"
# Note: we are in app/main.py. Parent is service. Parent.parent is topobathysim ??
# Actually `app/main.py` -> parent is `app`. parent.parent is `service`.
# `static` is in `service/static`.
# So `Path(__file__).parent.parent` is `service`.
static_dir = Path(__file__).resolve().parent.parent / "static"

if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="static")


def get_manager() -> BathyManager:
    if bathy_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return bathy_manager


# Visualization Constants
# Reasonable Global Defaults to prevent checkerboarding on slippy maps
# Visualization Constants
# Reasonable Global Defaults to prevent checkerboarding on slippy maps
GLOBAL_VMIN = -50.0
GLOBAL_VMAX = 20.0

# Feature Flags
# Set to True to skip querying Microsoft Planetary Computer (3DEP/Copernicus)
# Useful if API is throttled or to speed up local dev where land is not needed.
SKIP_LAND_BACKGROUND = os.getenv("SKIP_LAND_BACKGROUND", "False").lower() in ("true", "1", "yes")


def render_png(
    da: xr.DataArray, style: str = "default", vmin: float | None = None, vmax: float | None = None
) -> bytes:
    """Renders DataArray to PNG bytes using a terrain colormap, optionally with hillshade."""
    from io import BytesIO

    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    buf = BytesIO()
    vals = da.values.astype(np.float32)

    # Hillshade Calculation
    ls = LightSource(azdeg=315, altdeg=45)

    if style == "hillshade":
        # Pure Grayscale Hillshade
        hillshade = ls.hillshade(vals, vert_exag=10, dx=1.0, dy=1.0)
        mpimg.imsave(buf, hillshade, cmap="gray", format="png")
        buf.seek(0)
        return buf.getvalue()

    # Default: Color + Hillshade Overlay
    # Determine bounds: Explicit -> Global Default
    # To fix checkerboarding, we default to the Global Sane Range if no explicit bounds are given.
    eff_vmin = vmin if vmin is not None else GLOBAL_VMIN
    eff_vmax = vmax if vmax is not None else GLOBAL_VMAX

    # Adaptive Colormap Logic
    cmap = plt.cm.terrain
    norm = None

    # Logic:
    # 1. If Explicit/Global bounds cross zero: Use TwoSlopeNorm (Standard TopoBathy)
    # 2. If Purely Underwater: Use Turbo (High Contrast)
    # 3. If Purely Land: Use Texture

    # Logic:
    # 1. If Explicit/Global bounds cross zero: Use TwoSlopeNorm (Standard TopoBathy)
    # 2. If Purely Underwater: Use Turbo (High Contrast)
    # 3. If Purely Land: Use Texture

    if style in ["chart", "default"]:
        # Chart Style (Refined Hybrid) - NOW DEFAULT
        # Water: Blues (Dark -> Light)
        # Land: Yellow (Shore) -> Green (Low) -> Dark Green (High) -> Brown -> White (Peak)

        # 1. Sample 'Blues_r' for the bottom half (0.0 to 0.5)
        # Blues_r goes from Dark Blue to Light Blue/White.
        # We want Dark Blue (Deep) -> Light Blue (Shallow).
        blues = plt.get_cmap("Blues_r")

        # 2. Create Custom Land Colormap for top half (0.5 to 1.0)
        # Nodes relative to the land section (0.0 to 1.0 of the land part)
        # 0.0 (Shore) -> Pale Yellow
        # 0.2 (Lowland) -> Light Green
        # 0.5 (Mid) -> Forest Green
        # 0.8 (Mountain) -> Brown
        # 1.0 (Peak) -> White
        land_colors = [
            (0.0, "#F7E5B5"),  # Pale Yellow (Sand/Shore)
            (0.2, "lightgreen"),  # Lowland
            (0.5, "forestgreen"),  # Mid/Hills
            (0.8, "sienna"),  # Mountain
            (1.0, "snow"),  # Peak
        ]
        land_cmap = mcolors.LinearSegmentedColormap.from_list("land_custom", land_colors)

        # 3. Combine them
        # We need to construct a new colormap that stacks them.
        # We sample N colors from each and stack.
        n_bins = 256
        colors_water = blues(np.linspace(0.0, 1.0, n_bins))
        colors_land = land_cmap(np.linspace(0.0, 1.0, n_bins))

        # Stack: Water then Land
        colors_combined = np.vstack((colors_water, colors_land))

        cmap = mcolors.LinearSegmentedColormap.from_list("chart_hybrid", colors_combined)

        # 4. Use TwoSlopeNorm to ensure the center of the combined map (0.5) aligns with Z=0
        norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)

    elif style == "blues":
        # Blues Scale:
        # VMin (Deep) -> Dark Blue
        # VMax (Shallow/Surface) -> Light Blue
        # This is essentially 'Blues_r' (Reverse Blues)
        cmap = plt.get_cmap("Blues_r")
        norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)

    else:
        # "Classic" / Adaptive Fallback (was old default)

        # Check if we are strictly underwater (Global or Explicit)
        if eff_vmax <= 0:
            try:
                cmap = plt.get_cmap("turbo")
            except ValueError:
                cmap = plt.get_cmap("jet")
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)

        # Check if we are strictly land
        elif eff_vmin >= 0:
            land_colors = plt.cm.terrain(np.linspace(0.5, 1.0, 256))
            land_cmap = mcolors.LinearSegmentedColormap.from_list("terrain_land", land_colors)
            cmap = land_cmap
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)

        # Mixed / Standard Case (Crossing Zero)
        else:
            cmap = plt.cm.terrain
            # TwoSlopeNorm ensures 0 is always the transition from Blue to Green
            norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)
    try:
        rgb = ls.shade(vals, cmap=cmap, norm=norm, vert_exag=10, dx=1.0, dy=1.0, blend_mode="overlay")
        mpimg.imsave(buf, rgb, format="png")

    except Exception as e:
        logger.warning(f"Shade failed: {e}")
        mpimg.imsave(buf, vals, cmap="terrain", format="png")

    buf.seek(0)
    return buf.getvalue()


@app.get("/elevation", response_model=ElevationResponse)
async def get_elevation(
    lat: float, lon: float, manager: Annotated[BathyManager, Depends(get_manager)]
) -> ElevationResponse:
    try:
        # BathyManager handles fallback logic
        depth = manager.get_elevation(lat, lon)
        return ElevationResponse(elevation=depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/source_info")
async def get_source_info(
    lat: float, lon: float, manager: Annotated[BathyManager, Depends(get_manager)]
) -> dict[str, str]:
    """
    Returns metadata about the active data source for the coordinate.
    """
    return manager.get_source_info(lat, lon)


@app.get("/metadata", response_model=TIDReportResponse)
async def get_metadata(
    north: float,
    south: float,
    west: float,
    east: float,
    manager: Annotated[BathyManager, Depends(get_manager)],
) -> TIDReportResponse:
    """
    Returns TID quality report for the specified bounding box.
    Currently defaults to GEBCO context fetch.
    """
    try:
        # Pre-load/Fetch GEBCO context for this region
        # Note: In hybrid mode, this might ideally merge reports, but for now we look at GEBCO TID
        manager.fetch_global_context(north, south, west, east)

        # Access underlying GEBCO instance from manager
        # (Assuming we expose it or add reporting to manager)
        # For MVP: Access direct
        tid_data = manager.gebco.get_tid_classification()

        if tid_data is None:
            raise HTTPException(status_code=404, detail="No TID data found for region")

        report = source_report(tid_data)
        return TIDReportResponse(report=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/tiles/{z}/{x}/{y}")
@app.get("/tiles/{z}/{x}/{y}.tif")
@app.get("/tiles/{z}/{x}/{y}.png")
def get_xyz_tile(
    z: int,
    x: int,
    y: int,
    manager: Annotated[BathyManager, Depends(get_manager)],
    format: str = "tiff",
    lidar_url: str | None = None,
    ept_url: str | None = None,
    use_seam_blending: bool = True,
    style: str = Query("default", description="Visualization style"),
    vmin: float | None = Query(None, description="Explicit Min Elevation for colormap normalization"),
    vmax: float | None = Query(None, description="Explicit Max Elevation for colormap normalization"),
) -> Response:
    """
    XYZ Tile Endpoint (Slippy Map / Web Mercator).
    Emulates a standard Tile Server returning GeoTIFF heightmaps.
    """

    # Convert XYZ to Lat/Lon Bounding Box (Web Mercator / EPSG:3857 scheme)
    # Y is typically Top-Down in Google/OSM scheme
    n = 2.0**z

    # Longitude
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    # Latitude
    # rad = arctan(sinh(pi * (1 - 2 * y / n)))
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)

    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    # Implement simple disk cache
    cache_dir = Path.home() / ".cache" / "topobathysim" / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Differentiate cache by style/vmin/vmax params if rendering PNG
    if format == "png":
        parts = [str(z), str(x), str(y)]
        if style != "default":
            parts.append(style)
        if vmin is not None:
            parts.append(f"min{vmin}")
        if vmax is not None:
            parts.append(f"max{vmax}")
        tile_filename = "_".join(parts) + ".png"
    else:
        tile_filename = f"{z}_{x}_{y}.{format}"

    tile_path = cache_dir / tile_filename

    # Check cache
    if tile_path.exists():
        media_type = "image/png" if format == "png" else "image/tiff"
        with open(tile_path, "rb") as f:
            return Response(content=f.read(), media_type=media_type)

    start_time = time.time()
    logger.info(
        f"XYZ Request: z={z} x={x} y={y} | Bounds: N={north:.5f} S={south:.5f} W={west:.5f} E={east:.5f}"
    )

    # Delegate to get_fused_tile logic
    response = get_fused_tile(
        north=north,
        south=south,
        west=west,
        east=east,
        format=format,
        lidar_url=lidar_url,
        ept_url=ept_url,
        use_seam_blending=use_seam_blending,
        manager=manager,
        style=style,
        vmin=vmin,
        vmax=vmax,
    )

    # Save to cache if successful
    # Save to cache if successful
    if response.status_code == 200:
        # Atomic Write: Write to temp, then rename
        # This prevents partial reads by other processes
        try:
            tmp_tile_path = cache_dir / f".tmp_{tile_filename}_{os.getpid()}_{time.time_ns()}"
            with open(tmp_tile_path, "wb") as f:
                f.write(response.body)

            tmp_tile_path.rename(tile_path)
        except Exception as e:
            logger.warning(f"Failed to cache tile {z}/{x}/{y}: {e}")
            if "tmp_tile_path" in locals() and tmp_tile_path.exists():
                tmp_tile_path.unlink()

    logger.info(f"Tile {z}/{x}/{y} generated in {time.time() - start_time:.2f}s")
    return response


@app.get("/fused_tile")
def get_fused_tile(
    north: float,
    south: float,
    west: float,
    east: float,
    manager: Annotated[BathyManager, Depends(get_manager)],
    format: str = "tiff",
    lidar_url: str | None = None,
    ept_url: str | None = None,
    use_seam_blending: bool = True,
    style: str = "default",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Response:
    """
    Returns a Logistic-Fused GeoTIFF merging NOAA Lidar (Land) and BlueTopo (Sea).
    """
    from pathlib import Path

    try:
        # 0. Define Target Grid (Standard Tile Size)
        # 512x512 ensures high detail for 4k screens or standard high-dpi
        tile_size = 512

        # Calculate centers for the 512x512 grid
        lon_res = (east - west) / tile_size
        lat_res = (north - south) / tile_size

        # X: West to East (Ascending)
        xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=tile_size)
        # Y: North to South (Descending)
        ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=tile_size)

        target_grid = xr.DataArray(np.nan, coords={"y": ys, "x": xs}, dims=("y", "x"))
        target_grid.rio.write_crs("EPSG:4326", inplace=True)

        # 1. Fetch Lidar (Land) - Buffered Fetch
        # We still fetch a slightly larger area (5% buffer) to provide source data for the resampling
        lat_span = north - south
        lon_span = east - west
        buf_factor = 0.2

        b_north = north + (lat_span * buf_factor)
        b_south = south - (lat_span * buf_factor)
        b_east = east + (lon_span * buf_factor)
        b_west = west - (lon_span * buf_factor)

        lidar_da = None
        # Use the manager's configured LidarProvider (which has offline_mode set)
        lidar_provider = manager.lidar
        # Fallback if manager.lidar is None (e.g. use_lidar=False)?
        if lidar_provider is None:
            # Just create one locally, though this ignores offline_mode if manager doesn't have it
            lidar_provider = UsgsLidarProvider()

        if ept_url:
            from pyproj import Transformer

            # 4326 to 3857 (Projected)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            minx, miny = transformer.transform(b_west, b_south)
            maxx, maxy = transformer.transform(b_east, b_north)

            lidar_da = lidar_provider.fetch_lidar_from_ept(
                ept_url, bounds=(minx, miny, maxx, maxy), target_crs="EPSG:4326"
            )
        elif lidar_url:
            lidar_da = lidar_provider.fetch_lidar_from_laz(lidar_url)
        else:
            # STAC Lookup
            lidar_da = lidar_provider.fetch_lidar_from_stac(
                bounds=(b_west, b_south, b_east, b_north), target_crs="EPSG:4326"
            )

        if lidar_da is not None:
            logger.debug(f"DEBUG: Lidar DA CRS: {lidar_da.rio.crs}")
            # reproject_match handles resampling and alignment
            # Use bilinear for smooth terrain, or nearest to match source pixels
            lidar_da = lidar_da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)

        # 1.5 Fetch Intermediate Land (Tier 3)
        # We fetch it but apply it LAST as a background fill.

        land_da = None
        if not SKIP_LAND_BACKGROUND:
            try:
                land_provider = Usgs3DepProvider()
                land_da = land_provider.fetch_dem(bounds=(b_west, b_south, b_east, b_north))
            except Exception as e:
                logger.warning(f"Background Land Fetch failed (Soft Skip): {e}")
        else:
            logger.debug("Skipping Background Land (Configured SKIP_LAND_BACKGROUND=True)")

        if land_da is not None:
            logger.debug(f"DEBUG: Land DA CRS: {land_da.rio.crs}")
            logger.debug(
                f"DEBUG: Land Provider returned data. "
                f"Lat Range: {land_da.y.max().values} to {land_da.y.min().values}"
            )
            land_da = land_da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)

        # 3. Fuse Land Sources (Tier 1 Lidar + Tier 2 Land)
        fusion = FusionEngine()
        combined_land = None

        if lidar_da is not None and land_da is not None:
            logger.debug("DEBUG: Fusing Lidar and LandProvider (Seamline Blend)...")
            # Lidar is priority, Land is background. Blend edges.
            combined_land = fusion.fuse_seamline(lidar_da, land_da, blend_dist=20.0)
        elif lidar_da is not None:
            combined_land = lidar_da
        elif land_da is not None:
            combined_land = land_da

        logger.info("Main calling manager.get_grid...")
        bathy_da = manager.get_grid(b_south, b_north, b_west, b_east)
        logger.info("Main received bathy_da from manager.")

        # FORCE ALIGNMENT: Reproject Bathy to Target Grid
        if bathy_da is not None:
            # Optimization: Clip to target bounding box (+ buffer) to minimize data loading during reproject
            try:
                # Use the buffered bounds (b_west, etc) which are slightly larger than target_grid
                # clip_box handles CRS transformation of bounds
                bathy_da = bathy_da.rio.clip_box(
                    minx=b_west, miny=b_south, maxx=b_east, maxy=b_north, crs="EPSG:4326"
                )
                logger.info("Clipped bathy_da to standard bounds.")
            except Exception as e:
                logger.warning(f"Clip failed (likely no overlap), proceeding: {e}")

            logger.info("Main starting reproject_match...")
            # Avoid computing stats on full Dask array, it triggers full read!
            # logger.debug(f"DEBUG: Bathy DA stats: min={bathy_da.min().values}, max={bathy_da.max().values}")
            logger.debug("DEBUG: Bathy DA found. Reprojecting...")
            bathy_da = bathy_da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
        else:
            logger.debug("DEBUG: Bathy DA is None!")

        # 5. Final Fusion (Land + Sea)
        fused_da = None

        if combined_land is not None:
            # Check if land is actually present (not just all NaNs)
            # Check if land is actually present (not just all NaNs)
            # This check requires computation if combined_land is Dask.
            # But combined_land might be small (fused)? NO, it's reprojected (512x512).
            # So 512x512 computation is cheap.
            # However, safer to skip strict check if it causes issues.
            # But let's keep it for now as it helps logic.
            # Wait, np.all(np.isnan(combined_land.values)) computes it.
            # If combined_land came from LandProvider (lazy dask), this triggers compute.
            # But it's 512x512 now (after reproject). So it's fast.
            # The Bathy stats above were on the FULL 15GB array BEFORE reproject.

            # Let's keep this check but log safely.
            # logger.debug(f"DEBUG: Combined Land Stats: ...")
            # We removed the stats logging above, let's remove strict value check if possible?
            # No, logic depends on it: if all NaN -> combined_land = None.
            # This is fine for 512x512.

            if not np.all(np.isnan(combined_land.values)):
                pass  # Valid
            else:
                logger.debug("DEBUG: Combined Land is All-NaN (Water Only) -> Treating as None")
                combined_land = None

        if bathy_da is not None:
            # logger.debug(
            #    f"DEBUG: Bathy DA Stats: Min={np.nanmin(bathy_da.values)}, Max={np.nanmax(bathy_da.values)}"
            # )
            pass

        if combined_land is not None and bathy_da is not None:
            if use_seam_blending:
                fused_da = fusion.fuse_seamline(combined_land, bathy_da)
            else:
                fused_da = fusion.fuse(combined_land, bathy_da)
        elif combined_land is not None:
            fused_da = combined_land
        elif bathy_da is not None:
            fused_da = bathy_da
        else:
            raise HTTPException(status_code=404, detail="No data available for region")

        # note: fuses_da is now guaranteed to match target_grid dimensions (512x512) and bounds.
        # No extra clipping needed.

        # 4. Serialize
        if format == "png":
            png_data = render_png(fused_da, style=style, vmin=vmin, vmax=vmax)
            return Response(content=png_data, media_type="image/png")

        elif format == "npy":
            buf = io.BytesIO()
            np.save(buf, fused_da.values.astype(np.float32))
            buf.seek(0)
            return Response(content=buf.read(), media_type="application/octet-stream")

        elif format == "tif":
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_name = tmp.name

                # Add Progress Indicator for serialization
                from dask.diagnostics import ProgressBar

                logger.debug("Starting raster serialization (compute)...")
                with ProgressBar():
                    fused_da.rio.to_raster(tmp_name)
                logger.debug("Raster serialization complete.")

                tmp.seek(0)
                tif_data = tmp.read()

            Path(tmp_name).unlink()
            return Response(content=tif_data, media_type="image/tiff")

        return Response(status_code=400, content="Unsupported format")

    except Exception as e:
        logger.error(f"Error in get_fused_tile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/cache/clear")
async def clear_cache(type: str = "output") -> dict[str, object]:
    """
    Clears server-side caches.

    Args:
        type (str): 'output' (tiles), 'input' (raw data), or 'all'.
    """
    import shutil

    # Base cache dir (from main.py context or env)
    # We ideally get this from manager, but for now hardcode default as it's standard
    cache_root = Path("~/.cache/topobathysim").expanduser()

    deleted = []

    try:
        if type in ["output", "all"]:
            tiles_dir = cache_root / "tiles"
            if tiles_dir.exists():
                shutil.rmtree(tiles_dir)
                deleted.append("tiles")

        if type in ["input", "all"]:
            # Clear specific input subdirs
            for subdir in ["land", "bluetopo", "lidar"]:
                p = cache_root / subdir
                if p.exists():
                    shutil.rmtree(p)
                    deleted.append(subdir)

            # Also clear GEBCO/BlueTopo root inputs (legacy)
            # Be careful not to delete 'tiles' if type is only 'input'
            # But 'tiles' is in root.
            # Strategy: Delete known input patterns or just everything except tiles?
            # Safer: Just delete the subdirs we know are inputs + legacy glob

            # Legacy BlueTopo/GEBCO in root
            for f in cache_root.glob("*.tiff"):
                f.unlink()
            for f in cache_root.glob("*.nc"):  # GEBCO often netcdf
                f.unlink()
            deleted.append("root_inputs")

        # Re-create structure if needed (lazy done by services usually)

        return {"status": "success", "cleared": deleted, "message": f"Cleared cache types: {type}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
