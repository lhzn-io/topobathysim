import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import AsyncGenerator

# Set Matplotlib Backend to Agg (Non-Interactive) for Server Use
import matplotlib

matplotlib.use("Agg")

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from rasterio.enums import Resampling

from topobathysim.manager import BathyManager
from topobathysim.quality import source_report

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
static_dir = Path(__file__).resolve().parent.parent / "static"

if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="static")


def get_manager() -> BathyManager:
    if bathy_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return bathy_manager


# Visualization Constants
GLOBAL_VMIN = -50.0
GLOBAL_VMAX = 20.0

# Feature Flags
SKIP_LAND_BACKGROUND = os.getenv("SKIP_LAND_BACKGROUND", "False").lower() in (
    "true",
    "1",
    "yes",
)


def render_png(
    da: xr.DataArray,
    style: str = "default",
    vmin: float | None = None,
    vmax: float | None = None,
    zoom: int = 13,
) -> bytes:
    """Renders DataArray to PNG bytes using a terrain colormap, optionally with hillshade."""

    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    buf = BytesIO()
    vals = da.values.astype(np.float32)

    # Source Visualization Style
    if style == "source":
        # Create categorical colormap for IDs
        # 0=Canvas (Black), 10=BAG (Red), 20=Lidar (Green), 21=Topobathy (Lime),
        # 22=Fused (Orange), 30=CUDEM (Blue), 40=BlueTopo (Cyan), 50=Land (Brown), 60=GEBCO (Gray)

        # Colors dict
        cd = {
            0: (0, 0, 0, 1),  # Black
            10: (1, 0, 0, 1),  # Red (BAG)
            20: (0, 1, 0, 1),  # Green (Lidar)
            21: (0.5, 1, 0, 1),  # Lime (Topobathy)
            22: (1, 0.5, 0, 1),  # Orange (Fused)
            30: (0, 0, 1, 1),  # Blue (CUDEM)
            40: (0, 1, 1, 1),  # Cyan (BlueTopo)
            50: (0.6, 0.4, 0.2, 1),  # Brown (Land)
            60: (0.5, 0.5, 0.5, 1),  # Gray (GEBCO)
        }

        # Create output RGBA
        h, w = vals.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        for pid, color in cd.items():
            mask = np.isclose(vals, pid)
            rgba[mask] = color

        mpimg.imsave(buf, rgba, format="png")
        buf.seek(0)
        return buf.getvalue()

    # Contours Style
    if style == "contours":
        h, w = vals.shape

        # Safety Check for NaN
        if np.isnan(vals).all():
            # Return transparent 1x1 pixel
            # Or just clean handling
            fig = plt.figure(figsize=(1, 1), dpi=100)
            fig.savefig(buf, format="png", transparent=True)
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))
        ax.set_axis_off()
        fig.add_axes(ax)

        # Adaptive Intervals based on Zoom
        # Zoom 13 represents ~2km across. 10m contours are reasonable.
        # Zoom 16 represents ~200m. 2m contours are reasonable.
        interval = 10.0
        if zoom >= 14:
            interval = 5.0
        if zoom >= 16:
            interval = 2.0
        if zoom >= 18:
            interval = 1.0

        vmin_cnt = math.floor(np.nanmin(vals) / interval) * interval
        vmax_cnt = math.ceil(np.nanmax(vals) / interval) * interval
        levels = np.arange(vmin_cnt, vmax_cnt + interval, interval)

        # origin='upper' is critical for matching array indexing (0,0 at top-left)
        # origin=None (default) puts (0,0) at bottom-left
        # Use two sets of contours: normal and index

        # Standard lines
        cs = ax.contour(vals, levels=levels, colors="black", linewidths=0.5, alpha=0.7, origin="upper")

        # Zero line (Coastline) - Thick Red/Black
        ax.contour(vals, levels=[0], colors="red", linewidths=1.5, alpha=0.8, origin="upper")

        # Labels
        if zoom >= 14:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")

        # Invert Y axis to match image coordinates if origin='upper' didn't behave as expected with 'extent'
        # With origin='upper', index [0,0] is at the top.
        # But matplotlib default axis is y-up.
        ax.invert_yaxis()

        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Hillshade Calculation
    ls = LightSource(azdeg=315, altdeg=45)

    if style == "hillshade":
        # Pure Grayscale Hillshade
        hillshade = ls.hillshade(vals, vert_exag=10, dx=1.0, dy=1.0)
        mpimg.imsave(buf, hillshade, cmap="gray", format="png")
        buf.seek(0)
        return buf.getvalue()

    # Default: Color + Hillshade Overlay
    eff_vmin = vmin if vmin is not None else GLOBAL_VMIN
    eff_vmax = vmax if vmax is not None else GLOBAL_VMAX

    cmap = plt.get_cmap("terrain")
    norm: mcolors.Normalize | None = None

    if style in ["chart", "default"]:
        # Chart Style logic
        blues = plt.get_cmap("Blues_r")
        land_colors = [
            (0.0, "#F7E5B5"),
            (0.2, "lightgreen"),
            (0.5, "forestgreen"),
            (0.8, "sienna"),
            (1.0, "snow"),
        ]
        land_cmap = mcolors.LinearSegmentedColormap.from_list("land_custom", land_colors)

        n_bins = 256
        colors_water = blues(np.linspace(0.0, 1.0, n_bins))
        colors_land = land_cmap(np.linspace(0.0, 1.0, n_bins))

        colors_combined = np.vstack((colors_water, colors_land))
        cmap = mcolors.LinearSegmentedColormap.from_list("chart_hybrid", colors_combined)
        norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)

    elif style == "blues":
        cmap = plt.get_cmap("Blues_r")
        norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)

    else:
        if eff_vmax <= 0:
            try:
                cmap = plt.get_cmap("turbo")
            except ValueError:
                cmap = plt.get_cmap("jet")
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)
        elif eff_vmin >= 0:
            _terrain = plt.get_cmap("terrain")
            land_rgba = _terrain(np.linspace(0.5, 1.0, 256))
            land_cmap = mcolors.LinearSegmentedColormap.from_list("terrain_land", land_rgba)
            cmap = land_cmap
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)
        else:
            cmap = plt.get_cmap("terrain")
            norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)

    try:
        rgb = ls.shade(
            vals,
            cmap=cmap,
            norm=norm,
            vert_exag=10,
            dx=1.0,
            dy=1.0,
            blend_mode="overlay",
        )
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
        depth = manager.get_elevation(lat, lon)
        return ElevationResponse(elevation=depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/source_info")
async def get_source_info(
    lat: float, lon: float, manager: Annotated[BathyManager, Depends(get_manager)]
) -> dict[str, str]:
    return manager.get_source_info(lat, lon)


@app.get("/metadata", response_model=TIDReportResponse)
async def get_metadata(
    north: float,
    south: float,
    west: float,
    east: float,
    manager: Annotated[BathyManager, Depends(get_manager)],
) -> TIDReportResponse:
    try:
        manager.fetch_global_context(north, south, west, east)
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
    vmin: float | None = Query(None, description="Explicit Min Elevation"),
    vmax: float | None = Query(None, description="Explicit Max Elevation"),
) -> Response:
    """
    XYZ Tile Endpoint.
    """
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    cache_dir = Path.home() / ".cache" / "topobathysim" / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    if tile_path.exists():
        media_type = "image/png" if format == "png" else "image/tiff"
        with open(tile_path, "rb") as f:
            return Response(content=f.read(), media_type=media_type)

    start_time = time.time()
    logger.info(
        f"XYZ Request: z={z} x={x} y={y} | Bounds: N={north:.5f} S={south:.5f} W={west:.5f} E={east:.5f}"
    )

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
        zoom=z,
    )

    if response.status_code == 200:
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
    zoom: int = 13,
) -> Response:
    """
    Returns a Logistic-Fused GeoTIFF merging NOAA Lidar (Land) and BlueTopo (Sea).
    """
    try:
        # 0. Define Target Grid (Standard Tile Size)
        tile_size = 512

        # 1. Fetch Lidar (Land) - Buffered Fetch
        lat_span = north - south
        lon_span = east - west
        buf_factor = 0.2

        b_north = north + (lat_span * buf_factor)
        b_south = south - (lat_span * buf_factor)
        b_east = east + (lon_span * buf_factor)
        b_west = west - (lon_span * buf_factor)

        # 2. Main Calc
        # Calculate target shape for buffer (to maintain resolution matching tile_size)
        b_height = max(1, int(tile_size * ((b_north - b_south) / (north - south))))
        b_width = max(1, int(tile_size * ((b_east - b_west) / (east - west))))

        logger.info(f"Requests Manager Grid with Shape: {b_height}x{b_width}")

        # Determine if we need source mask
        need_source = style == "source"

        result = manager.get_grid(
            south=b_south,
            north=b_north,
            west=b_west,
            east=b_east,
            target_shape=(b_height, b_width),
            return_source_mask=need_source,
        )

        source_da = None
        if need_source:
            if isinstance(result, tuple):
                bathy_da, source_da = result
            else:
                bathy_da = cast(xr.DataArray, result)
        else:
            bathy_da = cast(xr.DataArray, result)

        logger.info("Main received bathy_da from manager.")

        # Create Standard Target Grid for reproject
        lon_res = (east - west) / tile_size
        lat_res = (north - south) / tile_size
        xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=tile_size)
        ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=tile_size)
        target_grid = xr.DataArray(np.nan, coords={"y": ys, "x": xs}, dims=("y", "x"))
        target_grid.rio.write_crs("EPSG:4326", inplace=True)

        # If style=source, we SHORT CIRCUIT logic to return just the source map from manager
        if need_source and source_da is not None:
            # Reproject to target
            source_da = source_da.rio.reproject_match(target_grid, resampling=Resampling.nearest)
            png_data = render_png(cast(xr.DataArray, source_da), style="source")
            return Response(content=png_data, media_type="image/png")

        # The Manager returns a fully fused grid (Lidar + Topobathy + BlueTopo + CUDEM + GEBCO)
        if bathy_da is not None:
            final_da = bathy_da
            # Reproject to target grid
            final_da = final_da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
        else:
            raise HTTPException(status_code=404, detail="No data available")

        # 4. Serialize
        if format == "png":
            png_data = render_png(cast(xr.DataArray, final_da), style=style, vmin=vmin, vmax=vmax, zoom=zoom)
            return Response(content=png_data, media_type="image/png")

        elif format == "npy":
            buf = BytesIO()
            np.save(buf, final_da.values.astype(np.float32))
            buf.seek(0)
            return Response(content=buf.read(), media_type="application/octet-stream")

        elif format == "npz":
            buf = BytesIO()
            np.savez_compressed(buf, elevation=final_da.values.astype(np.float32))
            buf.seek(0)
            return Response(content=buf.read(), media_type="application/octet-stream")

        elif format == "tif":
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_name = tmp.name
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    final_da.rio.to_raster(tmp_name)
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
    import shutil

    cache_root = Path("~/.cache/topobathysim").expanduser()
    deleted = []
    try:
        if type in ["output", "all"]:
            tiles_dir = cache_root / "tiles"
            if tiles_dir.exists():
                shutil.rmtree(tiles_dir)
                deleted.append("tiles")

        if type in ["input", "all"]:
            for subdir in ["land", "bluetopo", "lidar"]:
                p = cache_root / subdir
                if p.exists():
                    shutil.rmtree(p)
                    deleted.append(subdir)
            for f in cache_root.glob("*.tiff"):
                f.unlink()
            deleted.append("root_inputs")

        return {
            "status": "success",
            "cleared": deleted,
            "message": f"Cleared cache types: {type}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
