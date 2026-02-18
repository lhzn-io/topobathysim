import logging
import math
import os
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

# Set Matplotlib Backend to Agg (Non-Interactive) for Server Use
import matplotlib

matplotlib.use("Agg")

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

from topobathysim.runtime import run

# from topobathysim.quality import source_report # Removed as not directly supported in runtime yet
from .models import ElevationResponse, TIDReportResponse, TileMetadataResponse

# Configure Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Parse Debug Level from Env (set by run_server.py or environment)
debug_mode = int(os.environ.get("TOPOBATHYSIM_DEBUG", "0"))
log_level = logging.DEBUG if debug_mode >= 1 else logging.INFO

# We need to aggressively configure logging because Uvicorn may have already set up handlers
# and basicConfig does nothing if handlers exist.
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler (Always add)
file_handler = logging.FileHandler(log_dir / "service.log")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Stream Handler (Add if not present, to avoid duplicates from Uvicorn)
# Uvicorn adds a stream handler to the root logger or 'uvicorn'.
# If we are running under uvicorn, the root logger might already have a handler.
# BUT we want to ensure our format is used and it goes to stdout.
# So we aggressively add it if we don't see one that looks like ours.
has_console = False
for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
        h.setFormatter(formatter)  # Force our formatter
        h.setLevel(log_level)
        has_console = True
        break

if not has_console:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)


# --- Silence External Libraries even in Debug Mode ---
# We generally want to inspect our own logic, but not the internals of rasterio/matplotlib/etc for every tile.
for noisy_logger in [
    "rasterio",
    "fiona",
    "shapely",
    "matplotlib",
    "PIL",
    "botocore",
    "urllib3",
    "asyncio",
    "multipart",
    "uvicorn.access",  # Uvicorn access logs are redundant if we have middleware
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Explicitly set level for our app loggers
logging.getLogger("topobathyserve").setLevel(log_level)
logging.getLogger("topobathysim").setLevel(log_level)

logger = logging.getLogger("topobathyserve")

# Default Policy Path
DEFAULT_POLICY = Path(__file__).resolve().parent.parent.parent / "policies" / "examples" / "wlis.yaml"
POLICY_PATH: Path | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load .env (Resolve from topobathysim root)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Ensure bmi_topography finds the key
    if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
        os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

    global POLICY_PATH
    policy_env = os.getenv("TOPOBATHY_POLICY")
    POLICY_PATH = Path(policy_env) if policy_env else DEFAULT_POLICY

    if not POLICY_PATH.exists():
        logger.warning(f"Policy file not found: {POLICY_PATH}. Service may fail.")
    else:
        logger.info(f"Using Fusion Policy: {POLICY_PATH}")

    yield
    # clean up logic if needed


app = FastAPI(
    title="BathyServe",
    description="Microservice for Hybrid Bathymetry (GEBCO 2025 + BlueTopo)",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )
    return response


# Mount Viewer
static_dir = Path(__file__).resolve().parent.parent / "static"

if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "topobathyserve"}


def get_policy_path() -> Path:
    if POLICY_PATH is None or not POLICY_PATH.exists():
        raise HTTPException(status_code=503, detail="Policy path not initialized")
    return POLICY_PATH


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
    # (Implementation remains mostly same, just checking compatibility)
    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    buf = BytesIO()
    vals = da.values.astype(np.float32)

    # Source Visualization Style
    if style == "source":
        # Create unique colors for provider IDs (Sequential or Mapped)
        # Using a fixed categorical palette
        unique_ids = np.unique(vals)
        # Basic hashing for colors?
        # Or Just strict mapping if we knew them.
        # Fallback: Just map unique values to colors

        # Colors dict (Basic Palette)
        palette = [
            (0, 0, 0, 1),  # 0: Black/Canvas
            (1, 0, 0, 1),  # 1: Red
            (0, 1, 0, 1),  # 2: Green
            (0, 0, 1, 1),  # 3: Blue
            (1, 1, 0, 1),  # 4: Yellow
            (1, 0, 1, 1),  # 5: Magenta
            (0, 1, 1, 1),  # 6: Cyan
            (1, 0.5, 0, 1),  # 7: Orange
            (0.5, 0, 1, 1),  # 8: Purple
            (0.5, 0.5, 0.5, 1),  # 9: Gray
        ]

        h, w = vals.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        for _i, uid in enumerate(unique_ids):
            mask = np.isclose(vals, uid)
            if np.isnan(uid):
                continue

            # If ID matches legacy, use legacy colors?
            # 10=BAG, 20=Lidar...
            color = None
            uid_int = int(uid)

            # Legacy Map
            legacy_map = {
                10: (1, 0, 0, 1),  # Red (BAG)
                20: (0, 1, 0, 1),  # Green (Lidar)
                30: (0, 0, 1, 1),  # Blue (CUDEM)
                40: (0, 1, 1, 1),  # Cyan (BlueTopo)
                60: (0.5, 0.5, 0.5, 1),  # Gray (GEBCO)
            }
            color = legacy_map[uid_int] if uid_int in legacy_map else palette[uid_int % len(palette)]

            rgba[mask] = color

        mpimg.imsave(buf, rgba, format="png")
        buf.seek(0)
        return buf.getvalue()

    # Contours Style
    if style == "contours":
        h, w = vals.shape

        # Safety Check for NaN
        if np.isnan(vals).all():
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

        cs = ax.contour(vals, levels=levels, colors="black", linewidths=0.5, alpha=0.7, origin="upper")
        ax.contour(vals, levels=[0], colors="red", linewidths=1.5, alpha=0.8, origin="upper")

        if zoom >= 14:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")

        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Hillshade Calculation
    ls = LightSource(azdeg=315, altdeg=45)

    if style == "hillshade":
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
            # Depth only
            try:
                cmap = plt.get_cmap("turbo")
            except ValueError:
                cmap = plt.get_cmap("jet")
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)
        elif eff_vmin >= 0:
            # Land only
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
    lat: float, lon: float, policy_path: Annotated[Path, Depends(get_policy_path)]
) -> ElevationResponse:
    try:
        # Request a reasonable area to ensure we catch grid cells (GEBCO is ~450m)
        delta = 0.02  # ~2km buffer
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)

        # Run inference
        # Use 10m resolution for sampling
        ds = run(str(policy_path), bbox, resolution=10.0)

        if ds["elevation"].size == 0:
            return ElevationResponse(elevation=None)  # type: ignore

        # Sample closest using x/y (Projected CRS or EPSG:4326 uses x/y dims in runtime)
        # Note: runtime run() returns Dataset with dims ('y', 'x')
        val = ds["elevation"].sel(x=lon, y=lat, method="nearest").item()

        return ElevationResponse(elevation=float(val) if not np.isnan(val) else None)
    except ValueError as e:
        # Catch CRS validation errors from runtime
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"get_elevation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/source_info")
async def get_source_info(
    lat: float, lon: float, policy_path: Annotated[Path, Depends(get_policy_path)]
) -> dict[str, str]:
    try:
        delta = 0.00015
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        ds = run(str(policy_path), bbox, resolution=10.0)

        if ds["source_elevation"].size == 0:
            return {"source": "No Data", "id": "NaN"}

        sid = ds["source_elevation"].sel(lat=lat, lon=lon, method="nearest").item()

        # TODO: Lookup ID in ds.attrs or legend
        return {"source": f"Provider ID {sid}", "id": str(sid), "policy": policy_path.name}
    except Exception as e:
        return {"error": str(e)}


@app.get("/metadata", response_model=TIDReportResponse)
async def get_metadata(
    north: float,
    south: float,
    west: float,
    east: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> TIDReportResponse:
    try:
        # Metadata logic in runtime is minimal, mostly about the policy
        # Just run a coarse integration to get attrs?
        # Or just return policy details?
        # For now, return a stub report
        report = {
            "policy": str(policy_path),
            "status": "Runtime Metadata Not Generally Available via API yet",
        }
        return TIDReportResponse(report=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/tiles/coverage")
def get_tile_coverage(
    north: float,
    south: float,
    west: float,
    east: float,
    zoom: int,
) -> dict[str, object]:
    """
    Calculates the XYZ tile indices covering a given bounding box.
    Response includes min/max indices and a list of tiles (capped at 100 items).
    """
    try:

        def deg2num(lat_deg: float, lon_deg: float, z: int) -> tuple[int, int]:
            lat_rad = math.radians(lat_deg)
            n = 2.0**z
            xtile = int((lon_deg + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (xtile, ytile)

        def tile_bounds_calc(x: int, y: int, z: int) -> dict[str, float]:
            n = 2.0**z
            lon_min = x / n * 360.0 - 180.0
            lon_max = (x + 1) / n * 360.0 - 180.0
            lat_rad_n = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
            lat_max = math.degrees(lat_rad_n)
            lat_rad_s = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
            lat_min = math.degrees(lat_rad_s)
            return {"north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max}

        # Get tile coordinates for top-left (NW) and bottom-right (SE)
        # Note: BBox is North, South, West, East
        # In XYZ, y increases southwards

        # Valid Latitude Checks to prevent math domain errors
        north = min(max(north, -85.0511), 85.0511)
        south = min(max(south, -85.0511), 85.0511)

        min_x, min_y = deg2num(north, west, zoom)
        max_x, max_y = deg2num(south, east, zoom)

        # Handle wrap around or ordering if needed, simplistically:
        # Swap if min > max (though deg2num logic should handle standard west<east)
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        count = (max_x - min_x + 1) * (max_y - min_y + 1)

        tiles = []
        # Only list specific tiles if count is reasonable
        # Increased limit from 100 to 1000 to support larger viewport debugging
        if count <= 1000:
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    info: dict[str, object] = {"z": zoom, "x": x, "y": y}
                    info["bounds"] = tile_bounds_calc(x, y, zoom)
                    tiles.append(info)

        return {
            "zoom": zoom,
            "bounds": {"north": north, "south": south, "west": west, "east": east},
            "limits": {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y},
            "count": count,
            "tiles": tiles,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/tiles/{z}/{x}/{y}/metadata", response_model=TileMetadataResponse)
def get_tile_metadata(
    z: int,
    x: int,
    y: int,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    lidar_url: str | None = None,
    ept_url: str | None = None,
    use_seam_blending: bool = True,
) -> TileMetadataResponse:
    """
    Returns metadata for a specific fused tile (XYZ).
    """
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    # Cache key based on Policy + Geometry
    # data_sig_str = f"n={north:.6f}_s={south:.6f}_w={west:.6f}_e={east:.6f}_z={z}_policy={policy_path.name}"
    # data_hash = hashlib.md5(data_sig_str.encode("utf-8")).hexdigest()

    # We don't really have a metadata cache in the same way anymore,
    # but we could check if the tile output exists.
    # For now, return a generic status
    bounds = {"north": north, "south": south, "west": west, "east": east}

    return TileMetadataResponse(
        z=z, x=x, y=y, bounds=bounds, cache_status="unknown", fusion_sources=policy_path.name
    )


@app.get("/tiles/{z}/{x}/{y}")
@app.get("/tiles/{z}/{x}/{y}.tif")
@app.get("/tiles/{z}/{x}/{y}.png")
@app.get("/tiles/{z}/{x}/{y}.npy")
@app.get("/tiles/{z}/{x}/{y}.npz")
def get_xyz_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    policy_path: Annotated[Path, Depends(get_policy_path)],
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
    # Infer format
    path = request.url.path
    if format == "tiff":  # Only override if default
        if path.endswith(".png"):
            format = "png"
        elif path.endswith(".npy"):
            format = "npy"
        elif path.endswith(".npz"):
            format = "npz"

    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    logger.info(f"XYZ Request: z={z} x={x} y={y} | Format={format} | Style={style} | VMin={vmin} VMax={vmax}")

    start_time = time.time()

    # Calculate Resolution in Meters for this Zoom Level (Approx at center Lat)
    center_lat = (north + south) / 2.0
    # Web Mercator resolution equation
    res_meters = 156543.03 * math.cos(math.radians(center_lat)) / (2**z)

    # Run Fusion
    try:
        ds = run(str(policy_path), (west, south, east, north), resolution=res_meters)
    except ValueError as e:
        # CRS Validation Error
        logger.warning(f"Tile {z}/{x}/{y} out of bounds for policy: {e}")
        return Response(content=f"Policy/CRS Mismatch: {e}", status_code=400)
    except Exception as e:
        logger.error(f"Runtime failed: {e}")
        return Response(content=f"Error: {e}", status_code=500)

    # Render
    if format == "png":
        if style == "source":
            data = render_png(ds["source_elevation"], style="source", vmin=vmin, vmax=vmax, zoom=z)
        else:
            data = render_png(ds["elevation"], style=style, vmin=vmin, vmax=vmax, zoom=z)
        media_type = "image/png"

    elif format in ["npy", "npz"]:
        import io

        buf = io.BytesIO()
        if format == "npy":
            np.save(buf, ds["elevation"].values)
        else:
            np.savez_compressed(buf, elevation=ds["elevation"].values)
        data = buf.getvalue()
        media_type = "application/octet-stream"

    else:  # TIFF
        import io

        buf = io.BytesIO()
        ds["elevation"].rio.to_raster(buf, driver="GTiff")
        data = buf.getvalue()
        media_type = "image/tiff"

    logger.info(f"Tile {z}/{x}/{y} generated in {time.time() - start_time:.2f}s")
    return Response(content=data, media_type=media_type)


@app.post("/cache/clear")
async def clear_cache(type: str = "output") -> dict[str, object]:
    import shutil

    # Default cache location
    cache_root = Path("~/.cache/topobathysim").expanduser()
    deleted = []

    try:
        if type in ["output", "all"]:
            tiles_dir = cache_root / "tiles"
            if tiles_dir.exists():
                shutil.rmtree(tiles_dir)
                deleted.append("tiles")

            fused_dir = cache_root / "fused"
            if fused_dir.exists():
                shutil.rmtree(fused_dir)
                deleted.append("fused")

            fused_zarr = cache_root / "fused_zarr"
            if fused_zarr.exists():
                shutil.rmtree(fused_zarr)
                deleted.append("fused_zarr")

        if type in ["source", "all"]:
            # Clear provider caches
            for subdir in [
                "usgs_3dep",
                "noaa_bluetopo",
                "usgs_lidar",
                "ncei_bag",
                "ncei_cudem",
                "gebco_2025",
            ]:
                p = cache_root / subdir
                if p.exists():
                    shutil.rmtree(p)
                    deleted.append(subdir)

        return {
            "status": "success",
            "cleared": deleted,
            "message": f"Cleared cache types: {type}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
