import json
import logging
import math
import os
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone

# Set Matplotlib Backend to Agg (Non-Interactive) for Server Use
import matplotlib

matplotlib.use("Agg")

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from topobathysim.runtime import hydrate, run, should_consolidate

# from topobathysim.quality import source_report # Removed as not directly supported in runtime yet
from .models import (
    ElevationResponse,
    FusionRequest,
    HydrateRequest,
    TIDReportResponse,
    TileMetadataResponse,
)
from .routers import cache_viewer

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


def deg2num(lat_deg: float, lon_deg: float, z: int) -> tuple[int, int]:
    """Converts WGS84 coordinates to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**z
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Converts tile coordinates to WGS84 coordinates (top-left)."""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tile_bounds_calc(x: int, y: int, z: int) -> dict[str, float]:
    """Calculates the bounding box for a given tile."""
    lat_max, lon_min = num2deg(x, y, z)
    lat_min, lon_max = num2deg(x + 1, y + 1, z)
    return {"north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max}


def _parse_bbox_params(
    bbox: str | None,
    west: float | None,
    south: float | None,
    east: float | None,
    north: float | None,
) -> tuple[float, float, float, float]:
    if bbox:
        parts = [p.strip() for p in bbox.split(",") if p.strip()]
        if len(parts) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox must be 'west,south,east,north' with 4 comma-separated values",
            )
        try:
            west, south, east, north = (float(p) for p in parts)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="bbox values must be numeric") from exc

    if west is None or south is None or east is None or north is None:
        raise HTTPException(
            status_code=400,
            detail="Provide bbox or west/south/east/north query parameters",
        )

    if west >= east or south >= north:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox ordering (expected west < east and south < north)",
        )

    return west, south, east, north


def _normalize_output_format(format_raw: str) -> str:
    fmt = format_raw.strip().lower()
    if fmt in {"geotiff", "tiff", "tif"}:
        return "geotiff"
    if fmt in {"zarr", "zip"}:
        return "zarr"
    raise HTTPException(status_code=400, detail=f"Unsupported format '{format_raw}'")


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _build_zarr_attrs(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    resolution: float,
    policy_name: str,
) -> dict[str, Any]:
    crs = None
    if hasattr(ds, "rio") and ds.rio.crs:
        crs = ds.rio.crs.to_string()

    provenance = ds.attrs.get("provenance_dict", {})
    providers_used: list[str] = []
    for entry in provenance.values():
        if isinstance(entry, dict):
            provider = entry.get("provider")
            if provider:
                providers_used.append(str(provider))

    return {
        "crs": crs or "EPSG:4326",
        "bbox": [float(v) for v in bbox],
        "resolution_m": float(resolution),
        "vertical_datum": ds.attrs.get("vertical_datum", "Unknown"),
        "policy": policy_name,
        "providers_used": sorted(set(providers_used)),
        "created": datetime.now(timezone.utc).isoformat(),
        "provenance_dict": _json_safe(provenance) if provenance else {},
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load .env (Resolve from topobathysim root)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    global POLICY_PATH
    policy_env = os.getenv("TOPOBATHY_POLICY")
    POLICY_PATH = Path(policy_env) if policy_env else DEFAULT_POLICY

    if not POLICY_PATH.exists():
        logger.warning(f"Policy file not found: {POLICY_PATH}. Service may fail.")
    else:
        logger.info(f"Using Fusion Policy: {POLICY_PATH}")

    yield
    # clean up logic if needed


# Job tracking
HYDRATION_JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="BathyServe",
    description="Microservice for Hybrid Bathymetry (GEBCO 2025 + BlueTopo)",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(cache_viewer.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(static_dir / "favicon.ico")


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


@app.get("/fuse")
def fuse(
    policy_path: Annotated[Path, Depends(get_policy_path)],
    bbox: str | None = Query(
        None,
        description="Bounding box as 'west,south,east,north'",
    ),
    west: float | None = Query(None, description="Bounding box west (lon)"),
    south: float | None = Query(None, description="Bounding box south (lat)"),
    east: float | None = Query(None, description="Bounding box east (lon)"),
    north: float | None = Query(None, description="Bounding box north (lat)"),
    resolution: float = Query(30.0, description="Output resolution in meters"),
    format: str = Query("geotiff", description="Output format: geotiff or zarr"),
) -> Response:
    bbox_tuple = _parse_bbox_params(bbox, west, south, east, north)
    fmt = _normalize_output_format(format)

    try:
        ds = run(str(policy_path), bbox_tuple, resolution=resolution, use_cache=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"/fuse failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "elevation" not in ds or ds["elevation"].size == 0:
        raise HTTPException(status_code=404, detail="Fusion returned no elevation data")

    if fmt == "zarr":
        import zarr.storage

        ds_out = ds.copy()
        keep_vars = ["elevation"]
        if "source_elevation" in ds_out:
            keep_vars.append("source_elevation")
        ds_out = ds_out[keep_vars]
        ds_out.attrs = _build_zarr_attrs(ds, bbox_tuple, resolution, policy_path.name)

        with TemporaryDirectory() as tmpdir:
            zarr_zip_path = Path(tmpdir) / "fused.zarr.zip"
            store = zarr.storage.ZipStore(str(zarr_zip_path), mode="w")
            try:
                ds_out.to_zarr(store, mode="w")
            finally:
                store.close()

            data = zarr_zip_path.read_bytes()

        headers = {"Content-Disposition": "attachment; filename=fused.zarr.zip"}
        return Response(content=data, media_type="application/zip", headers=headers)

    buf = BytesIO()
    ds["elevation"].rio.to_raster(buf, driver="GTiff")
    return Response(content=buf.getvalue(), media_type="image/tiff")


@app.post("/fuse")
def fuse_post(
    request: FusionRequest,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> Response:
    fmt = _normalize_output_format(request.format)

    # Dynamic Policy Logic
    policy_input: Path | str = policy_path
    if request.policy_yaml:
        policy_input = request.policy_yaml
    elif request.policy_name:
        # Simple lookup: check if a policy with this name exists in policies dir?
        # For now, let's just log and fallback or fail?
        # Actually, let's look for it in the same directory as default policy if possible.
        # But for strict compatibility with instructions, let's prioritize override string.
        pass

    try:
        ds = run(
            policy_input=policy_input,
            bbox=request.bbox,
            resolution=request.resolution,
            use_cache=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"POST /fuse failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "elevation" not in ds or ds["elevation"].size == 0:
        raise HTTPException(status_code=404, detail="Fusion returned no elevation data")

    if fmt == "zarr":
        import zarr.storage

        ds_out = ds.copy()
        keep_vars = ["elevation"]
        if "source_elevation" in ds_out:
            keep_vars.append("source_elevation")
        ds_out = ds_out[keep_vars]

        # Policy name for metadata
        p_name = policy_path.name
        if request.policy_name:
            p_name = request.policy_name
        elif request.policy_yaml:
            p_name = "custom_override"

        ds_out.attrs = _build_zarr_attrs(ds, request.bbox, request.resolution, p_name)

        with TemporaryDirectory() as tmpdir:
            zarr_zip_path = Path(tmpdir) / "fused.zarr.zip"
            store = zarr.storage.ZipStore(str(zarr_zip_path), mode="w")
            try:
                ds_out.to_zarr(store, mode="w")
            finally:
                store.close()

            data = zarr_zip_path.read_bytes()

        headers = {"Content-Disposition": "attachment; filename=fused.zarr.zip"}
        return Response(content=data, media_type="application/zip", headers=headers)

    buf = BytesIO()
    ds["elevation"].rio.to_raster(buf, driver="GTiff")
    return Response(content=buf.getvalue(), media_type="image/tiff")


# Visualization Constants
GLOBAL_VMIN = -2000.0
GLOBAL_VMAX = 1200.0

# Feature Flags
SKIP_LAND_BACKGROUND = os.getenv("SKIP_LAND_BACKGROUND", "False").lower() in (
    "true",
    "1",
    "yes",
)

# Tiling Constants
TILE_SIZE_PX = 512
TILE_PADDING_PX = 4  # Standard padding to avoid edge artifacts and ensure cache parity


# So I can change the signature of render_png to accept an optional 'legend' dict.


def render_png(
    da: xr.DataArray,
    style: str = "default",
    vmin: float | None = None,
    vmax: float | None = None,
    zoom: int = 13,
    legend: dict[int, str] | None = None,  # NEW ARGUMENT
    pad: int = 0,
    provenance_dict: dict[Any, dict[str, str]] | None = None,
) -> bytes:
    """Renders DataArray to PNG bytes using a terrain colormap, optionally with hillshade."""
    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    buf = BytesIO()
    vals = da.values.astype(np.float32)

    # Handle 3D Arrays (e.g. (1, h, w) or (C, h, w))
    # Handle 3D Arrays
    if vals.ndim == 3:
        if vals.shape[0] == 1:
            # (1, h, w) -> (h, w)
            vals = vals.squeeze(0)
        elif vals.shape[-1] == 1:
            # (h, w, 1) -> (h, w)
            vals = vals.squeeze(-1)
        else:
            # Ambiguous (C, h, w) or (h, w, C).
            # We assume (C, h, w) standard raster layout if not singleton.
            logger.warning(f"render_png received 3D array {vals.shape}. Using first band.")
            vals = vals[0, :, :]

    # Last resort check
    if vals.ndim != 2:
        logger.error(f"render_png expected 2D array, got {vals.shape}")
        # Try to flatten or reshape? No, just fail gracefully or let it crash with better log
        # But we can try to force 2D if it's (h, w, 1) or similar
        if vals.ndim == 3 and vals.shape[-1] == 1:
            vals = vals.squeeze(-1)

    # Source Visualization Style
    if style == "source":
        # Safeguard: Ensure source IDs are integers. Floating point noise or NaNs
        # can cause np.unique() to explode (O(N^2) loop), leading to timeouts/crashes.
        if not np.issubdtype(vals.dtype, np.integer):
            vals = np.nan_to_num(vals, nan=0).astype(np.int32)

        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
        h, w = vals.shape

        # Qualitative Palette (Set3 / Paired)
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        # Set3 has 12 colors. Paired has 12.
        # We map Provider ID -> Color Index

        # If no legend provided, fallback to random/hash
        cmap_name = "Set3"  # Good for up to 12 categories
        base_cmap = plt.get_cmap(cmap_name)

        # Create a custom colormap where specific integer values map to specific colors
        # We know IDs are 1..N
        # We want ID 1 -> Color 0, ID 2 -> Color 1, etc.

        # Helper to get color for an ID
        def get_color(uid: int) -> tuple[float, float, float, float]:
            from typing import cast

            if uid == 0:
                return cast(tuple[float, float, float, float], (0, 0, 0, 1))  # No Data / Background
            if np.isnan(uid):
                return cast(tuple[float, float, float, float], (0, 0, 0, 0))  # Transparent

            # Use provenance_dict to map detailed IDs back to their provider color consistently
            if provenance_dict:
                int_uid = int(uid)
                str_uid = str(int_uid)

                prov_info = provenance_dict.get(int_uid) or provenance_dict.get(str_uid)

                if prov_info:
                    provider_str = prov_info.get("provider", "")

                    if provider_str and legend:
                        # Look up the provider string in the legend map to get its global integer ID
                        for leg_id, leg_name in legend.items():
                            if leg_name == provider_str:
                                idx = (leg_id - 1) % 12
                                return cast(tuple[float, float, float, float], base_cmap(idx))

            # Fallback for generic provider IDs
            try:
                idx = (int(uid) - 1) % 12  # Cycle through 12 colors
            except (ValueError, TypeError):
                idx = 0
            return cast(tuple[float, float, float, float], base_cmap(idx))

        h, w = vals.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        unique_ids = np.unique(vals)
        for uid in unique_ids:
            if np.isnan(uid) or uid == 0:
                continue

            mask = np.isclose(vals, uid)
            rgba[mask] = get_color(uid)

        mpimg.imsave(buf, rgba, format="png")
        buf.seek(0)
        return buf.getvalue()

    # Contours Style
    if style == "contours":
        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
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
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_axis_off()

        interval = 10.0
        if zoom >= 14:
            interval = 5.0
        if zoom >= 16:
            interval = 2.0
        if zoom >= 18:
            interval = 1.0

        # Catch empty range
        if np.nanmin(vals) == np.nanmax(vals):
            levels = [np.nanmin(vals)]
        else:
            vmin_cnt = math.floor(np.nanmin(vals) / interval) * interval
            vmax_cnt = math.ceil(np.nanmax(vals) / interval) * interval
            levels = np.arange(vmin_cnt, vmax_cnt + interval, interval).tolist()

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
        if pad > 0:
            hillshade = hillshade[pad:-pad, pad:-pad]
        mpimg.imsave(buf, hillshade, cmap="gray", format="png")
        buf.seek(0)
        return buf.getvalue()

    # Default: Color + Hillshade Overlay
    eff_vmin = vmin if vmin is not None else GLOBAL_VMIN
    eff_vmax = vmax if vmax is not None else GLOBAL_VMAX

    cmap = plt.get_cmap("terrain")
    norm: mcolors.Normalize | None = None

    if style in ["chart", "default"]:
        # Use standard Blues_r for water, but truncate it before pure white.
        # Rationale: Standard Blues_r hits pure white at 1.0 (0m).
        # This washes out shallow texture because hillshading on white is weak.
        # The standalone "Blues" style looks better because it normalizes VMIN-VMAX linearly,
        # placing 0m (~60% of range) at a nice pale blue (not white).
        # We manually emulate this by sampling Blues_r only up to 0.7 for water.
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
        # Truncate water colormap at 0.7 to keep shallowest water pale blue (not white)
        colors_water = blues(np.linspace(0.0, 0.7, n_bins))
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
        if pad > 0:
            rgb = rgb[pad:-pad, pad:-pad]

        mpimg.imsave(buf, rgb, format="png")

    except Exception as e:
        logger.warning(f"Shade failed: {e}")
        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
        mpimg.imsave(buf, vals, cmap="terrain", format="png")

    buf.seek(0)
    return buf.getvalue()


@app.get("/legend")
async def get_legend(policy_path: Annotated[Path, Depends(get_policy_path)]) -> dict[str, Any]:
    """
    Returns the legend mapping (Provider Name -> Color Hex) for the active policy.
    Used by the frontend to render the dynamic legend.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from topobathysim.policy.loader import generate_provider_legend, load_policy

    # Reload policy to be safe (or assume it's loaded in runtime, but runtime doesn't expose object easily)
    # We load it here to get the legend mapping
    try:
        policy = load_policy(str(policy_path))
        legend_map = generate_provider_legend(policy)  # {1: "ProviderA", 2: "ProviderB"}

        # Generate Colors matching render_png logic (Set3)
        cmap = plt.get_cmap("Set3")

        # Build priority map: provider_key -> step_index (0-based, higher = higher priority)
        priority_map: dict[str, int] = {}
        for variable in policy.variables:
            for step_idx, step in enumerate(variable.steps):
                priority_map[step.provider] = step_idx

        items = []
        for pid, name in legend_map.items():
            # ID to Color
            idx = (pid - 1) % 12
            rgba = cmap(idx)
            hex_color = mcolors.to_hex(rgba)

            items.append({"id": pid, "name": name, "color": hex_color, "priority": priority_map.get(name, 0)})

        return {"items": items}

    except Exception as e:
        logger.error(f"Failed to generate legend: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/elevation")
async def get_elevation(
    lat: float,
    lon: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    zoom: int | None = Query(None, description="Preferred zoom level/resolution to check in cache"),
) -> ElevationResponse:
    try:
        logger.info(f"Elevation Lookup: lat={lat}, lon={lon}, zoom={zoom}")

        # Check if we have any cached fused datasets for this coordinate
        # We start with the requested zoom, or the highest available in the tile cache
        base_cache_dir = Path(os.environ.get("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser()

        check_zooms = []
        if zoom is not None:
            check_zooms = [zoom]
        else:
            # Dynamically discover available zoom levels from the tile data cache
            tile_data_dir = base_cache_dir / "tiles" / "data"
            if tile_data_dir.exists():
                try:
                    # Collect all zoom level directories (must be numeric)
                    zooms = []
                    for d in tile_data_dir.iterdir():
                        if d.is_dir() and d.name.isdigit():
                            zooms.append(int(d.name))
                    # Sort descending for highest resolution first
                    check_zooms = sorted(zooms, reverse=True)
                except Exception as e:
                    logger.debug(f"Failed to scan tile cache for zooms: {e}")

        # Fallback to defaults if no zooms discovered (Z15 is ~4.7m, Z14 is ~9.5m)
        if not check_zooms:
            check_zooms = [15, 14, 13, 12]

        for z in check_zooms:
            try:
                # 1. Determine Tile Center Resolution
                # We must match the resolution used during tile generation,
                # which is based on the TILE CENTER latitude,
                # not the specific point latitude.
                tx, ty = deg2num(lat, lon, z)
                bounds = tile_bounds_calc(tx, ty, z)
                center_lat = (bounds["north"] + bounds["south"]) / 2.0

                n = 2.0**z
                res_meters = 156543.03 * math.cos(math.radians(center_lat)) / n
                res_meters = round(res_meters, 8)

                # 2. Check Cache for the Standard Grid Cell containing this point
                # We do NOT use padded_bbox here anymore because run() caches
                # by standardized grid cells (0.05 deg), not by arbitrary tile bboxes.
                try:
                    from topobathysim.runtime import get_cache_info_for_point

                    cache_path, key_hash = get_cache_info_for_point(str(policy_path), lon, lat, res_meters)
                except ValueError:
                    continue

                logger.debug(
                    f"Checking Elevation Cache: Z={z} res={res_meters} "
                    f"point=({lon:.5f},{lat:.5f}) key={key_hash}"
                )

                if cache_path.exists():
                    logger.info(f"Elevation Cache Hit (Cell Z{z}): {cache_path.name}")
                    from filelock import FileLock

                    with FileLock(cache_path.with_suffix(".lock")):
                        ds = xr.open_dataset(
                            cache_path,
                            engine="zarr",
                            chunks=None,
                            decode_coords="all",
                            consolidated=should_consolidate(),
                        )
                        ds = ds.load()

                    # Sample closest using projected coordinates (EPSG:3857)
                    earth_r = 6378137.0
                    x_m = earth_r * math.radians(lon)
                    y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

                    val = ds["elevation"].sel(x=x_m, y=y_m, method="nearest").item()
                    return ElevationResponse(elevation=float(val) if not np.isnan(val) else None)

            except Exception as e:
                logger.debug(f"Failed to check/load tile cache at Z{z}: {e}")
                continue

        logger.info(f"Elevation Cache Miss for {lat}, {lon} (Zoom {zoom}). Falling back to 10m global run.")

        # Request a reasonable area to ensure we catch grid cells (GEBCO is ~450m)
        delta = 0.02  # ~2km buffer
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)

        # Run inference
        # Use 10m resolution for sampling
        ds = run(str(policy_path), bbox, resolution=10.0)

        if ds["elevation"].size == 0:
            return ElevationResponse(elevation=None)  # type: ignore

        # Sample closest using projected coordinates (EPSG:3857)
        earth_r = 6378137.0
        x_m = earth_r * math.radians(lon)
        y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        val = ds["elevation"].sel(x=x_m, y=y_m, method="nearest").item()

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

        # Sample closest using projected coordinates (EPSG:3857)
        earth_r = 6378137.0
        x_m = earth_r * math.radians(lon)
        y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        sid = ds["source_elevation"].sel(x=x_m, y=y_m, method="nearest").item()

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
        # Get tile coordinates for top-left (NW) and bottom-right (SE)
        # Note: BBox is North, South, West, East
        # In XYZ, y increases southwards

        # Valid Latitude Checks to prevent math domain errors
        north_clamped = min(max(north, -85.0511), 85.0511)
        south_clamped = min(max(south, -85.0511), 85.0511)

        min_x, min_y = deg2num(north_clamped, west, zoom)
        max_x, max_y = deg2num(south_clamped, east, zoom)

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
    style: str = Query("default", description="Visualization style"),
    vmin: float | None = Query(None, description="Explicit Min Elevation"),
    vmax: float | None = Query(None, description="Explicit Max Elevation"),
    tile_size: int = Query(512, description="Output pixel dimension for standard square XYZ tiles"),
) -> TileMetadataResponse:
    """
    Returns metadata for a specific fused tile (XYZ).
    """
    import datetime
    import hashlib

    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    bounds = {"north": north, "south": south, "west": west, "east": east}

    # Check Cache existence
    base_cache_dir = Path(os.getenv("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser() / "tiles"
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    # Check NPY (data) first
    npy_path = base_cache_dir / "data" / str(z) / str(x) / f"{y}_{param_hash}.npy"
    # Then try Visual
    safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
    png_path = base_cache_dir / "visual" / safe_style / str(z) / str(x) / f"{y}_{param_hash}.png"

    created_at = None
    cache_status = "miss"
    provenance_dict = None

    for path_to_check in [npy_path, png_path]:
        if path_to_check.exists() and path_to_check.stat().st_size > 0:
            mtime = path_to_check.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
            created_at = dt.strftime("%Y%m%d-%H:%M:%S UTC")
            cache_status = "hit"

            # Check for sidecar metadata JSON
            import json

            meta_path = path_to_check.parent / f"{y}_{param_hash}_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        provenance_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read provenance {meta_path}: {e}")

            break

    return TileMetadataResponse(
        z=z,
        x=x,
        y=y,
        bounds=bounds,
        created_at=created_at,
        cache_status=cache_status,
        fusion_sources=policy_path.name,
        request_params=params_str,
        provenance_dict=provenance_dict,
    )


@app.get("/tiles/{z}/{x}/{y}/pixel")
def get_tile_pixel_source(
    z: int,
    x: int,
    y: int,
    lat: float,
    lon: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    style: str = Query("default"),
    vmin: float | None = Query(None),
    vmax: float | None = Query(None),
    tile_size: int = Query(512),
) -> dict[str, Any]:
    """
    Returns the source that contributed the pixel at (lat, lon) within tile (z, x, y).

    Requires the PNG tile to have been generated first (reads the ``_src.npz`` sidecar
    written alongside the PNG cache file).
    """
    import hashlib
    import json

    earth_r = 6378137.0
    half = math.pi * earth_r  # 20037508.342789244

    # Tile bounds in Web Mercator (EPSG:3857)
    tile_width_m = 2 * half / (2**z)
    west_m = x * tile_width_m - half
    north_m = half - y * tile_width_m

    # Project lat/lon (EPSG:4326) to Web Mercator
    x_m = earth_r * math.radians(lon)
    y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    # Map to pixel coordinates in the tile_size x tile_size grid
    col = int((x_m - west_m) / tile_width_m * tile_size)
    row = int((north_m - y_m) / tile_width_m * tile_size)
    col = max(0, min(tile_size - 1, col))
    row = max(0, min(tile_size - 1, row))

    base_cache_dir = Path(os.getenv("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser() / "tiles"
    safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
    tile_dir = base_cache_dir / "visual" / safe_style / str(z) / str(x)
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    src_npz_path = tile_dir / f"{y}_{param_hash}_src.npz"
    if not src_npz_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Pixel source data not available. Request the PNG tile first.",
        )

    source_arr = np.load(src_npz_path)["source_elevation"]
    row = min(row, source_arr.shape[0] - 1)
    col = min(col, source_arr.shape[1] - 1)
    source_id = int(source_arr[row, col])

    meta_path = tile_dir / f"{y}_{param_hash}_meta.json"
    provenance_dict: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                provenance_dict = json.load(f)
        except Exception:
            pass

    source_info = provenance_dict.get(str(source_id))
    return {"source_id": source_id, "source": source_info, "pixel": {"row": row, "col": col}}


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
    tile_size: int = Query(512, description="Output pixel dimension for standard square XYZ tiles"),
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

    # --- Caching Logic ---
    base_cache_dir = Path(os.getenv("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser() / "tiles"

    # Determine Cache Subdirectory based on format/style (Parity with Legacy)
    if format in ["png", "jpg", "jpeg"]:
        # Visual Cache: tiles/visual/{style}/{z}/{x}/
        safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
        tile_dir = base_cache_dir / "visual" / safe_style / str(z) / str(x)
        ext = format
    elif format in ["npy", "npz"]:
        # Data Cache: tiles/data/{z}/{x}/
        tile_dir = base_cache_dir / "data" / str(z) / str(x)
        ext = format
    else:
        # Raw/Other Cache (TIFF): tiles/raw/{z}/{x}/
        tile_dir = base_cache_dir / "raw" / str(z) / str(x)
        ext = "tif" if format == "tiff" else format

    # Hash unique parameters (Policy, VMin/Max) to handle different renderings of same tile
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    import hashlib

    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"{y}_{param_hash}.{ext}"

    media_type = "image/tiff" if format == "tiff" else f"image/{format}"
    if format in ["npy", "npz"]:
        media_type = "application/octet-stream"

    # 1. Check Cache
    if tile_path.exists() and tile_path.stat().st_size > 0:
        logger.debug(f"Cache Hit: {tile_path}")
        return FileResponse(tile_path, media_type=media_type)

    start_time = time.time()

    # Calculate Resolution in Meters for this Zoom Level (Approx at center Lat)
    center_lat = (north + south) / 2.0
    # Web Mercator resolution equation
    res_meters = 156543.03 * math.cos(math.radians(center_lat)) / (2**z)
    res_meters = round(res_meters, 8)

    # Pad requested bbox to supply surrounding context for PNG Hillshade rendering
    # Ensures Matplotlib 3x3 surface normal calculation interpolates accurately
    # Use pad_px=2 to completely remove any 3x3 kernel edge effects
    pad_px = 2 if format == "png" else 0
    dx_deg = (east - west) / tile_size
    dy_deg = (north - south) / tile_size

    # Fetch slightly larger region than strictly necessary to prevent edge artifacts.
    # For source/data outputs, avoid fetch halo to prevent deterministic seam stripes
    # from neighboring-tile context leaking into per-tile provenance.
    fetch_pad = 0 if style == "source" or format in ["npy", "npz"] else TILE_PADDING_PX
    req_west = west - dx_deg * fetch_pad
    req_east = east + dx_deg * fetch_pad
    req_south = south - dy_deg * fetch_pad
    req_north = north + dy_deg * fetch_pad

    logger.info(
        f"Tile Run Call: z={z} x={x} y={y} res={res_meters} bbox={(req_west, req_south, req_east, req_north)}"
    )
    # Run Fusion
    try:
        ds = run(
            str(policy_path),
            (req_west, req_south, req_east, req_north),
            resolution=res_meters,
            use_cache=True,
        )
        # Add logging here
        logger.info(f"Tile Fusion Result: res={res_meters} bbox={(req_west, req_south, req_east, req_north)}")
        logger.debug(f"/tiles run() returned ds with attrs: {ds.attrs}")
    except ValueError as e:
        # CRS Validation Error
        logger.warning(f"Tile {z}/{x}/{y} out of bounds for policy: {e}")
        return Response(content=f"Policy/CRS Mismatch: {e}", status_code=400)
    except Exception as e:
        import traceback

        logger.error(f"Runtime failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return Response(content=f"Error: {e}", status_code=500)

    # Force alignment to explicitly bounded 256x256 grid for Leaflet rendering
    # Mathematically resample exact Web Mercator bounds to eliminate visual overlap seams
    try:
        if ds.rio.crs is not None and ds.rio.crs.to_string() == "EPSG:3857":
            # Compute exact EPSG:3857 meter bounds for this XYZ tile
            n_tiles = 2.0**z

            # WGS 84 Semimajor axis
            r_earth = 6378137.0

            # Spherical Mercator math
            min_x_m = x / n_tiles * (2 * math.pi * r_earth) - (math.pi * r_earth)
            max_x_m = (x + 1) / n_tiles * (2 * math.pi * r_earth) - (math.pi * r_earth)

            lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n_tiles)))
            min_y_m = r_earth * math.log(math.tan(math.pi / 4 + lat_rad_south / 2))

            lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n_tiles)))
            max_y_m = r_earth * math.log(math.tan(math.pi / 4 + lat_rad_north / 2))

            dx_m = (max_x_m - min_x_m) / tile_size
            dy_m = (max_y_m - min_y_m) / tile_size

            # Add 1x pixel padding specifically for PNG Hillshade rendering to avoid 3x3 kernel edge artifacts
            n_dim = tile_size + 2 * pad_px
            target_x = np.linspace(
                min_x_m - dx_m * (pad_px - 0.5),
                max_x_m + dx_m * (pad_px - 0.5),
                n_dim,
            )
            target_y = np.linspace(
                max_y_m + dy_m * (pad_px - 0.5), min_y_m - dy_m * (pad_px - 0.5), n_dim
            )  # North up

            crs = ds.rio.crs
            ds_attrs = ds.attrs
            ds_elev = ds["elevation"].interp(x=target_x, y=target_y, method="linear")
            if "source_elevation" in ds:
                # Ensure integer source IDs after interpolation
                # (nearest maintains values, but NaNs promote to float)
                ds_src = ds["source_elevation"].interp(x=target_x, y=target_y, method="nearest")
                ds_src = ds_src.fillna(0).astype("int32")
                ds = xr.Dataset({"elevation": ds_elev, "source_elevation": ds_src})
            else:
                ds = xr.Dataset({"elevation": ds_elev})
            ds.attrs = ds_attrs
            if crs is not None:
                ds.rio.write_crs(crs, inplace=True)
        else:
            # Fallback for non-EPSG:3857 to explicitly enforce output dimensions using scipy zoom
            target_px = tile_size + 2 * pad_px
            if ds["elevation"].shape != (target_px, target_px):
                import warnings

                from scipy.ndimage import zoom as _zoom

                elev = ds["elevation"].values
                zy = target_px / elev.shape[0]
                zx = target_px / elev.shape[1]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    elev_resized = _zoom(elev.astype(np.float32), (zy, zx), order=1)

                old_x = ds.coords["x"].values
                old_y = ds.coords["y"].values
                new_x = np.linspace(old_x[0], old_x[-1], target_px)
                new_y = np.linspace(old_y[0], old_y[-1], target_px)

                ds_out = {
                    "elevation": xr.DataArray(elev_resized, dims=("y", "x"), coords={"y": new_y, "x": new_x})
                }

                if "source_elevation" in ds:
                    src_elev = ds["source_elevation"].values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        src_elev_resized = _zoom(src_elev.astype(np.float32), (zy, zx), order=0).astype(
                            np.uint16
                        )
                    ds_out["source_elevation"] = xr.DataArray(
                        src_elev_resized, dims=("y", "x"), coords={"y": new_y, "x": new_x}
                    )

                crs = ds.rio.crs
                ds_attrs = ds.attrs
                ds = xr.Dataset(ds_out)
                ds.attrs = ds_attrs
                if crs is not None:
                    ds.rio.write_crs(crs, inplace=True)

    except Exception as e:
        logger.warning(f"Failed to cleanly interpolate/resize tile bounds: {e}")

    # Render
    if format == "png":
        if style == "source":
            # Generate Legend Map for Dynamic Coloring
            from topobathysim.policy.loader import generate_provider_legend, load_policy

            # We have policy_path, but need the object.
            # load_policy is cached? Let's hope it's fast enough or LRU cached.
            # In production, we'd cache this at app level.
            # Assuming load_policy is cheap enough for now (parsing YAML)
            try:
                pol = load_policy(str(policy_path))
                legend_map = generate_provider_legend(pol)
            except Exception as e:
                logger.warning(f"Failed to load legend for source rendering: {e}")
                legend_map = None

            data = render_png(
                ds["source_elevation"],
                style="source",
                vmin=vmin,
                vmax=vmax,
                zoom=z,
                legend=legend_map,
                pad=pad_px,
                provenance_dict=ds.attrs.get("provenance_dict"),
            )
        else:
            data = render_png(ds["elevation"], style=style, vmin=vmin, vmax=vmax, zoom=z, pad=pad_px)
        media_type = "image/png"

    elif format in ["npy", "npz"]:
        import io

        buf = io.BytesIO()
        if format == "npy":
            np.save(buf, ds["elevation"].values)
        else:
            np.savez_compressed(
                buf, elevation=ds["elevation"].values, source_elevation=ds["source_elevation"].values
            )
        data = buf.getvalue()
        media_type = "application/octet-stream"

    else:  # TIFF
        import io

        buf = io.BytesIO()
        da_out = ds["elevation"]

        # Ensure proper dimension order (Band, Y, X) for GDAL
        if da_out.ndim == 3 and da_out.shape[-1] == 1:
            # If shape is (H, W, 1), move band to front
            da_out = da_out.transpose("band", "y", "x")
            # If shape is (1, H, W) it's already fine

        # Write to Raster
        da_out.rio.to_raster(buf, driver="GTiff")
        data = buf.getvalue()
        media_type = "image/tiff"

    # 2. Write to Cache
    try:
        with open(tile_path, "wb") as f:
            f.write(data)

        logger.debug(f"Tile attributes before json write: {ds.attrs}")

        # Write Sidecar Metadata if we have provenance, filtered to IDs present in this tile
        if "provenance_dict" in ds.attrs:
            import json

            all_provenance = ds.attrs["provenance_dict"]
            if "source_elevation" in ds:
                present_ids = {int(v) for v in np.unique(ds["source_elevation"].values) if v != 0}
                filtered_provenance = {k: v for k, v in all_provenance.items() if int(k) in present_ids}
            else:
                filtered_provenance = dict(all_provenance)
            meta_path = tile_dir / f"{y}_{param_hash}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(filtered_provenance, f)

        # Write source_elevation NPZ sidecar for pixel-level source lookups
        if format == "png" and "source_elevation" in ds:
            src_npz_path = tile_dir / f"{y}_{param_hash}_src.npz"
            if not src_npz_path.exists():
                src_arr = ds["source_elevation"].values
                # Strip padding applied for hillshade rendering so array aligns with the 512x512 PNG
                if pad_px > 0 and src_arr.ndim == 2 and src_arr.shape[0] == tile_size + 2 * pad_px:
                    src_arr = src_arr[pad_px:-pad_px, pad_px:-pad_px]
                buf = BytesIO()
                np.savez_compressed(buf, source_elevation=src_arr)
                with open(src_npz_path, "wb") as f:
                    f.write(buf.getvalue())

    except Exception as e:
        logger.warning(f"Failed to write cache file {tile_path}: {e}")

    logger.info(f"Tile {z}/{x}/{y} generated in {time.time() - start_time:.2f}s")
    return Response(content=data, media_type=media_type)


@app.get("/policy/reload")
async def reload_policy_cache() -> dict[str, object]:
    from topobathysim.policy.loader import clear_policy_cache

    clear_policy_cache()
    return {
        "status": "success",
        "message": "Policy cache cleared",
    }


@app.post("/cache/clear")
@app.get("/cache/clear")
async def clear_cache(type: str = "output") -> dict[str, object]:
    import shutil

    # Default cache location
    cache_root = Path(os.getenv("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser()
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
                "noaa_topobathy",
                "usgs_lidar",
                "ncei_bag",
                "ncei_cudem",
                "gebco_2025",
            ]:
                p = cache_root / subdir
                if p.exists():
                    shutil.rmtree(p)
                    deleted.append(subdir)

        if type in ["orphaned", "cleanup", "all"]:
            count_locks = 0
            count_tmps = 0
            try:
                for root, _, files in os.walk(cache_root):
                    for file in files:
                        if file.endswith(".lock") or ".tmp" in file:
                            try:
                                fp = Path(root) / file
                                fp.unlink(missing_ok=True)
                                if file.endswith(".lock"):
                                    count_locks += 1
                                else:
                                    count_tmps += 1
                            except Exception as e:
                                logger.warning(f"Failed to delete orphaned file {file}: {e}")

                if count_locks > 0 or count_tmps > 0:
                    deleted.append(f"orphaned_locks({count_locks})")
                    deleted.append(f"orphaned_tmps({count_tmps})")
            except Exception as e:
                logger.warning(f"Failed to cleanup orphaned files: {e}")

        return {
            "status": "success",
            "cleared": deleted,
            "message": f"Cleared cache types: {type}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _hydrate_task(
    job_id: str,
    policy_path: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    max_workers: int = 4,
) -> None:
    try:
        HYDRATION_JOBS[job_id]["status"] = "running"
        logger.info(
            f"Starting background hydration [{job_id}] for {bbox} @ {resolution}m with {max_workers} workers"
        )
        stats = hydrate(
            policy_path,
            bbox,
            resolution=resolution,
            max_workers=max_workers,
            job_id=job_id,
            jobs_dict=HYDRATION_JOBS,
        )
        HYDRATION_JOBS[job_id]["status"] = "completed"
        HYDRATION_JOBS[job_id]["stats"] = stats
        logger.info(f"Hydration [{job_id}] finished: {stats}")
    except Exception as e:
        HYDRATION_JOBS[job_id]["status"] = "failed"
        HYDRATION_JOBS[job_id]["error"] = str(e)
        logger.error(f"Hydration [{job_id}] failed: {e}", exc_info=True)


@app.post("/hydrate")
async def trigger_hydrate(
    request: HydrateRequest,
    background_tasks: BackgroundTasks,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> dict[str, Any]:
    """
    Trigger a background hydration process for the specified bounding box and resolution.
    This ensures all underlying grid cells are computed and cached.
    """
    start_time = datetime.now()
    import uuid

    job_id = str(uuid.uuid4())

    HYDRATION_JOBS[job_id] = {
        "id": job_id,
        "status": "pending",
        "bbox": request.bbox,
        "resolution": request.resolution,
        "submitted_at": start_time.isoformat(),
        "total_cells": 0,
        "processed_cells": 0,
        "cached_cells": 0,
        "failed_cells": 0,
    }

    # Handle custom policy or default to loaded policy
    policy_input = request.policy_yaml if request.policy_yaml else str(policy_path)

    background_tasks.add_task(
        _hydrate_task,
        job_id,
        policy_input,
        request.bbox,
        request.resolution,
        request.max_workers,
    )

    return {
        "status": "accepted",
        "message": "Hydration task started in background",
        "job_id": job_id,
        "bbox": request.bbox,
        "resolution": request.resolution,
        "timestamp": start_time.isoformat(),
    }


@app.get("/hydrate/{job_id}")
async def get_hydration_status(job_id: str) -> dict[str, Any]:
    """Get the status of a specific hydration job."""
    if job_id not in HYDRATION_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return HYDRATION_JOBS[job_id]
