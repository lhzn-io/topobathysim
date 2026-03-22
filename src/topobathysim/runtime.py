"""
Core Runtime for TopoBathySim.

This module executes fusion policies to generate topobathymetric datasets.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer

from topobathysim.operators.blend import metric_feather, overwrite
from topobathysim.policy.loader import (
    generate_provider_legend,
    hash_policy,
    load_policy,
    load_policy_from_str,
)
from topobathysim.policy.schema import OperatorType
from topobathysim.providers.base import ProviderNoDataError
from topobathysim.providers.registry import registry

logger = logging.getLogger(__name__)


def should_consolidate() -> bool:
    """
    Check if Zarr consolidation should be enabled based on environment variables.
    Defaults to True for performance, but can be disabled via TOPOBATHY_ZARR_CONSOLIDATED=0/false
    to support non-Python Zarr readers like Zarr.jl.
    """
    val = os.environ.get("TOPOBATHY_ZARR_CONSOLIDATED", "true").lower()
    return val in ("true", "1", "yes", "on")


def _get_grid_cells(
    bbox: tuple[float, float, float, float], target_crs: str
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Calculate the discrete standard grid cells covering the requested bounding box."""
    start_lon, start_lat, end_lon, end_lat = bbox

    # Either EPSG:4326 or default to 0.05 anyways for standard cells
    grid_size = 0.05

    # Snap to grid
    min_x_idx = np.floor(start_lon / grid_size)
    max_x_idx = np.floor(end_lon / grid_size)
    min_y_idx = np.floor(start_lat / grid_size)
    max_y_idx = np.floor(end_lat / grid_size)

    cells = []
    # +1 to include the max index
    for x in range(int(min_x_idx), int(max_x_idx) + 1):
        for y in range(int(min_y_idx), int(max_y_idx) + 1):
            cell_min_lon = x * grid_size
            cell_min_lat = y * grid_size
            cell_max_lon = (x + 1) * grid_size
            cell_max_lat = (y + 1) * grid_size
            cells.append((cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat))

    return cells, grid_size


def _run_cell(
    policy: Any,
    cell_bbox: tuple[float, float, float, float],
    resolution: float | None,
    time: datetime | None,
    target_crs: str,
    halo_pct: float = 0.10,
) -> xr.Dataset:
    """Execute fusion on a standard grid cell with a small context halo."""
    # 1. Expand bbox by halo percentage to fetch context for seamless blending
    min_lon, min_lat, max_lon, max_lat = cell_bbox
    lon_halo = (max_lon - min_lon) * halo_pct
    lat_halo = (max_lat - min_lat) * halo_pct

    fetch_bbox = (
        min_lon - lon_halo,
        min_lat - lat_halo,
        max_lon + lon_halo,
        max_lat + lat_halo,
    )

    # We calculate explicit Reprojection bounds for Target CRS canvases.
    # The Halo padding ensures seamless pixel transitions across edges.

    # Reproject BBox if Target CRS is not EPSG:4326
    if target_crs != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        # Fetch canvas bounds using the HALO bbox
        f_min_x, f_min_y = transformer.transform(fetch_bbox[0], fetch_bbox[1])
        f_max_x, f_max_y = transformer.transform(fetch_bbox[2], fetch_bbox[3])
        fetch_min_x, fetch_max_x = min(f_min_x, f_max_x), max(f_min_x, f_max_x)
        fetch_min_y, fetch_max_y = min(f_min_y, f_max_y), max(f_min_y, f_max_y)

        # We also need the EXACT cell bounds in the target CRS to clip the halo away later
        c_min_x, c_min_y = transformer.transform(min_lon, min_lat)
        c_max_x, c_max_y = transformer.transform(max_lon, max_lat)
        clip_min_x, clip_max_x = min(c_min_x, c_max_x), max(c_min_x, c_max_x)
        clip_min_y, clip_max_y = min(c_min_y, c_max_y), max(c_min_y, c_max_y)

    else:
        fetch_min_x, fetch_min_y, fetch_max_x, fetch_max_y = fetch_bbox
        clip_min_x, clip_min_y, clip_max_x, clip_max_y = cell_bbox

    # Resolve Resolution
    # Default to 30.0 meters if not provided
    input_resolution_meters = resolution if resolution is not None else 30.0

    # Smart Resolution Logic
    if target_crs == "EPSG:4326":
        # Global isotropic alignment to guarantee flawless xarray mosaicing
        # 1 degree at equator is approx 111320.0 meters
        res_x = input_resolution_meters / 111320.0
        res_y = input_resolution_meters / 111320.0
    else:
        res_x = input_resolution_meters
        res_y = input_resolution_meters

    # 3. Snap Canvas to Global Resolution Grid
    # This guarantees that regardless of the requested cell or halo,
    # the pixel centers align perfectly globally, avoiding
    # interpolation/interleaving artifacts when mosaicing.
    x_idx_start = int(np.floor(fetch_min_x / res_x))
    x_idx_end = int(np.ceil(fetch_max_x / res_x))
    width = max(1, x_idx_end - x_idx_start)

    y_idx_start = int(np.floor(fetch_min_y / res_y))
    y_idx_end = int(np.ceil(fetch_max_y / res_y))
    height = max(1, y_idx_end - y_idx_start)

    # Actual snapped bounds
    snapped_min_x = x_idx_start * res_x
    snapped_max_y = y_idx_end * res_y

    transform = Affine.translation(snapped_min_x, snapped_max_y) * Affine.scale(res_x, -res_y)

    # Calculate pixel centers globally
    xs = (np.arange(x_idx_start, x_idx_start + width) + 0.5) * res_x
    ys = (np.arange(y_idx_end - 1, y_idx_end - 1 - height, -1) + 0.5) * res_y

    # Optional: round coordinates to 8 decimal places for floating-point stability
    xs = np.round(xs, 8)
    ys = np.round(ys, 8)

    elevation = xr.DataArray(
        data=np.full((height, width), np.nan, dtype=np.float32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="elevation",
    )
    elevation.rio.write_crs(target_crs, inplace=True)
    elevation.rio.write_transform(transform, inplace=True)

    source_elevation = xr.DataArray(
        data=np.full((height, width), 0, dtype=np.uint32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="source_elevation",
    )
    source_elevation.rio.write_crs(target_crs, inplace=True)
    source_elevation.rio.write_transform(transform, inplace=True)

    cell_provenance_dict: dict[int, str] = {}

    # 5. Execution Loop
    for variable in policy.variables:
        if variable.name != "elevation":
            continue

        if variable.background is not None:
            elevation = elevation.fillna(variable.background)

        logger.info(f"Execution order: {[s.provider for s in variable.steps]}")

        for step in variable.steps:
            provider_cls = registry.get_provider_class(step.provider)
            provider_instance = provider_cls()

            try:
                # Fetch using the expanded halo bbox to ensure we grab edge data
                fetched_data = provider_instance.fetch_layer(
                    fetch_bbox,
                    resolution=input_resolution_meters,
                    crs="EPSG:4326",
                    filter=step.filter.model_dump() if step.filter else {},
                )
            except ProviderNoDataError as e:
                logger.info(f"{step.provider}: NoData - {e}")
                continue
            except Exception as e:
                logger.error(f"{step.provider}: Failed to fetch - {e}", exc_info=True)
                continue

            if fetched_data is None:
                logger.info(f"[PROBE-RUNTIME] Provider {step.provider} returned None.")
                continue

            # Extract per-provider source provenance from Dataset
            provider_source_id_da = None
            if isinstance(fetched_data, xr.Dataset):
                if "provenance_dict" in fetched_data.attrs:
                    cell_provenance_dict.update(fetched_data.attrs["provenance_dict"])
                provider_source_id_da = fetched_data.get("source_id")
                fetched_data = fetched_data["elevation"]

            # --- Global Filter Constraints ---
            if step.filter:
                if step.filter.min_elevation is not None:
                    fetched_data = fetched_data.where(fetched_data >= step.filter.min_elevation)
                if step.filter.max_elevation is not None:
                    fetched_data = fetched_data.where(fetched_data <= step.filter.max_elevation)

            # --- Enforce 2D Array Shape ---
            # Providers might return (Band, Y, X) or (Y, X, Band)
            if fetched_data.ndim == 3:
                if fetched_data.shape[0] == 1:
                    # (1, y, x) -> (y, x)
                    fetched_data = fetched_data.squeeze(axis=0)
                elif fetched_data.shape[-1] == 1:
                    # (y, x, 1) -> (y, x)
                    fetched_data = fetched_data.squeeze(axis=-1)
                else:
                    # Multi-band (C, y, x) -> Take first band
                    logger.warning(
                        f"Provider {step.provider} returned multi-band data {fetched_data.shape}. "
                        "Using first band only."
                    )
                    fetched_data = fetched_data[0, :, :]

            # Final check to ensure we are 2D
            if fetched_data.ndim != 2:
                logger.warning(
                    f"Provider {step.provider} returned unexpected shape {fetched_data.shape}. Skipping."
                )
                continue
            # ------------------------------

            # Align to Canvas
            try:
                # Defensive check on duplicates before reproject
                # Handle both projected (x,y) and geographic (lon,lat) dimensions
                for dim in ["x", "y", "lon", "lat"]:
                    if (
                        dim in fetched_data.dims
                        and dim in fetched_data.indexes
                        and not fetched_data.indexes[dim].is_unique
                    ):
                        fetched_data = fetched_data.drop_duplicates(dim=dim)

                aligned_data = fetched_data.rio.reproject_match(elevation)

            except Exception as e:
                logger.error(f"Reprojection/Index Alignment failed for {step.provider}: {e}")
                # Try dropping duplicates forcefully again? Or continue
                continue

            new_data_mask = aligned_data.notnull()

            aligned_source_id = None
            if provider_source_id_da is not None:
                try:
                    from rasterio.enums import Resampling

                    aligned_source_id = provider_source_id_da.rio.reproject_match(
                        elevation, resampling=Resampling.nearest
                    )
                    # Squeeze any spurious band dimension introduced by reproject
                    if "band" in aligned_source_id.dims:
                        aligned_source_id = aligned_source_id.squeeze("band", drop=True)
                    if aligned_source_id.ndim != 2:
                        aligned_source_id = aligned_source_id.squeeze(drop=True)
                except Exception as e:
                    logger.warning(f"Failed to reproject source_id mask for {step.provider}: {e}")

            # Use provider data wherever elevation is valid. Source IDs may be missing after
            # reprojection, so we can fall back to provider-level IDs for attribution.
            provider_valid_mask = new_data_mask

            if not provider_valid_mask.any():
                continue

            # 4. Generate Provider Legend & IDs
            legend = generate_provider_legend(policy)
            provider_to_id = {v: k for k, v in legend.items()}

            provider_id = provider_to_id.get(step.provider, 0)

            # --- Pairwise Logic ---
            # 1. Initialize mask of pixels to be processed by default rule
            remaining_mask = provider_valid_mask.copy()

            # Apply Transitions (Specific Overrides)
            for rule in step.transitions:
                target_id = provider_to_id.get(rule.target_provider)
                if target_id is None:
                    continue

                # Identify overlap: Valid New Data AND Underlying Source == TargetID
                transition_mask = provider_valid_mask & (source_elevation == target_id)

                if not transition_mask.any():
                    continue

                # Apply specific operator
                if rule.operator == OperatorType.metric_feather and rule.blend_distance:
                    blended = metric_feather(elevation, aligned_data, rule.blend_distance)
                elif rule.operator == OperatorType.overwrite:
                    blended = overwrite(elevation, aligned_data, transition_mask)
                else:
                    # Fallback to overwrite for unsupported operators in V0.
                    # Edge Case: If operator is 'metric_feather' but blend_distance is None,
                    # we fall through to here (Safe Default: Hard Cut).
                    blended = overwrite(elevation, aligned_data, transition_mask)

                # Update Canvas for this transition region
                elevation = elevation.where(~transition_mask, blended)

                # Remove these pixels from the default processing list
                remaining_mask = remaining_mask & (~transition_mask)

            # Apply Default Operation to remaining pixels
            if remaining_mask.any():
                if step.operation == OperatorType.metric_feather and step.blend_distance:
                    blended_default = metric_feather(elevation, aligned_data, step.blend_distance)
                elif step.operation == OperatorType.overwrite:
                    blended_default = overwrite(elevation, aligned_data, remaining_mask)
                else:
                    blended_default = overwrite(elevation, aligned_data, remaining_mask)

                elevation = elevation.where(~remaining_mask, blended_default)

            # Update Provenance (Global update for all valid pixels)
            if aligned_source_id is not None:
                source_detail_mask = (
                    provider_valid_mask & aligned_source_id.notnull() & (aligned_source_id > 0)
                )
                source_elevation = xr.where(
                    source_detail_mask,
                    aligned_source_id,
                    source_elevation,  # type: ignore
                )
                source_fallback_mask = provider_valid_mask & (~source_detail_mask)
                source_elevation = xr.where(source_fallback_mask, provider_id, source_elevation)  # type: ignore
            else:
                source_elevation = xr.where(provider_valid_mask, provider_id, source_elevation)  # type: ignore
            # Enforce 2D — xr.where can reintroduce a band dim from aligned_source_id
            if "band" in source_elevation.dims:
                source_elevation = source_elevation.squeeze("band", drop=True)

    # 6. Clip Halo Away! We only want the exact cell bounds
    # The cell bounds are clip_min_x, clip_min_y, clip_max_x, clip_max_y
    # We clip cleanly before saving to cache. Because we are projecting/translating,
    # pixel alignment might mean we should slice cleanly.

    # We slice based on 'x' and 'y' coordinates of the dataset.
    # Note: ys are usually decreasing (max_y to min_y) in raster.

    # Use epsilon > 0.5 * res_x so that any pixel center within one step of a cell boundary is
    # always included by at least one adjacent cell.  The 0.05° WGS84 cell grid does not align
    # exactly with the EPSG:3857 pixel grid, so the clip boundary can fall between two pixel
    # centers leaving a 1-pixel gap.  0.6 * res_x guarantees overlap (handled by drop_duplicates).
    epsilon = res_x * 0.6
    ds = xr.Dataset({"elevation": elevation, "source_elevation": source_elevation})

    # Slice X
    ds = ds.sel(x=slice(clip_min_x - epsilon, clip_max_x + epsilon))

    # Slice Y (Handle both Ascending and Descending Y coordinates)
    if len(ds.y) > 0 and ds.y[0] > ds.y[-1]:
        # Descending (North Up)
        ds = ds.sel(y=slice(clip_max_y + epsilon, clip_min_y - epsilon))
    else:
        # Ascending (South Up)
        ds = ds.sel(y=slice(clip_min_y - epsilon, clip_max_y + epsilon))

    # Ensure valid dimensions
    if ds.x.size == 0 or ds.y.size == 0:
        logger.warning(f"Clip resulted in empty dimensions for cell {cell_bbox}")

    logger.debug(f"_run_cell returning provenance: {cell_provenance_dict}")
    ds.attrs["provenance_dict"] = cell_provenance_dict

    return cast(xr.Dataset, ds)


def _resolve_policy(policy_input: str | Path) -> Any:
    """
    Resolve policy input to a FusionPolicy object.
    Accepts a filepath (str/Path) OR a raw YAML string.
    """
    # Naive check: if it looks like a path and exists, load it
    s_input = str(policy_input)
    if (s_input.endswith(".yaml") or s_input.endswith(".yml")) and os.path.exists(s_input):
        return load_policy(s_input)

    # Otherwise treat as raw content
    return load_policy_from_str(s_input)


def get_fused_cache_info(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> tuple[Path, str]:
    """
    Returns the expected Path and the hash key for a fused Zarr cache file.
    Hashes the POLICY CONTENT + BBOX + RES to create a unique ID.
    This centralized function ensures metadata parity across all endpoints.
    """
    import json
    import os

    policy = _resolve_policy(policy_input)
    target_crs = policy.crs

    default_cache = "~/.cache/topobathysim/fused_zarr"
    cache_dir_str = os.environ.get("TOPOBATHYSIM_CACHE_DIR", default_cache)
    if cache_dir_str != default_cache and not cache_dir_str.endswith("fused_zarr"):
        cache_dir = Path(cache_dir_str).expanduser() / "fused_zarr"
    else:
        cache_dir = Path(cache_dir_str).expanduser()

    # MD5 Hashing logic (Standardized Precision: 8 decimals)
    # We hash the policy content digest itself to keep the key short but content-addressed
    policy_content_hash = hash_policy(policy.model_dump())

    key_dict = {
        "policy_hash": policy_content_hash,
        "cell": [round(x, 8) for x in bbox],
        "res": round(resolution, 8),
        "crs": target_crs,
    }
    key_str = json.dumps(key_dict, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()

    return cache_dir / f"{key_hash}.zarr", key_hash


def get_cache_info_for_point(
    policy_path: str,
    lon: float,
    lat: float,
    resolution: float,
) -> tuple[Path, str]:
    """
    Finds the fused cache file corresponding to the standard grid cell containing the point.
    This ensures alignment with the tiling engine which caches by standard grid cells.
    """
    policy = load_policy(policy_path)
    # Treat point as a degenerate bbox to verify grid cell alignment
    cells, _ = _get_grid_cells((lon, lat, lon, lat), policy.crs)

    if not cells:
        # Should be covered by at least one cell if valid
        # If outside area of use (rarely checked here), fallback
        raise ValueError(f"Point {lon},{lat} not covered by any '{policy.crs}' grid cell.")

    # There should only be one cell for a point
    cell_bbox = cells[0]
    return get_fused_cache_info(policy_path, cell_bbox, resolution)


def get_fused_cache_path(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> Path:
    """Legacy wrapper for get_fused_cache_info returning only the Path."""
    path, _ = get_fused_cache_info(policy_input, bbox, resolution)
    return path


def hydrate(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
) -> dict[str, int]:
    """
    Hydrate the cache for a given bbox and resolution by processing all covering grid cells.
    Does not return the fused dataset, only ensures cells are cached.
    Mocks the behavior of run() but without aggregating results in memory.
    """
    import shutil

    policy = _resolve_policy(policy_input)
    target_crs = policy.crs
    res = resolution if resolution else 30.0

    # Determine Grid Cells
    cells, grid_cell_size = _get_grid_cells(bbox, target_crs)
    legend = generate_provider_legend(policy)

    # Use centralized cache helper to ensure base dir exists
    # We call get_fused_cache_info just to get the parent dir, though we do it per-cell below
    # because the hash differs per cell.
    # We'll just rely on the loop to create dirs.

    stats = {"total": len(cells), "cached": 0, "processed": 0, "failed": 0}

    logger.info(f"Starting hydration for {len(cells)} cells. BBox: {bbox}, Res: {res}")

    for i, cell_bbox in enumerate(cells, 1):
        # Use centralized cache helper
        cache_path, key_hash = get_fused_cache_info(policy_input, cell_bbox, resolution=res)

        # Ensure parent dir exists (lazy creation)
        if not cache_path.parent.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        is_cached = False
        if cache_path.exists():
            try:
                # Lightweight check: valid Zarr?
                ds_meta = xr.open_dataset(cache_path, engine="zarr", chunks="auto", decode_coords="all")

                valid_grid = ds_meta.attrs.get("grid_cell_size") == grid_cell_size
                # We trust our hash includes policy content, so a hit is a hit on policy content.
                # But we double check legend just in case of collisions or old data.
                cached_legend = ds_meta.attrs.get("policy_legend")
                valid_legend = not (cached_legend and cached_legend != str(legend))

                if valid_grid and valid_legend:
                    is_cached = True
                    stats["cached"] += 1

                ds_meta.close()
            except Exception as e:
                logger.warning(f"Cache check failed for {cache_path}: {e}")

        if is_cached:
            if i % 10 == 0:
                logger.info(f"Hydration progress: {i}/{len(cells)} (Cached)")
            continue

        logger.info(f"Hydrating cell {i}/{len(cells)}: {cell_bbox} -> {cache_path.name}")

        # Process and Cache
        try:
            ds_cell = _run_cell(
                policy=policy,
                cell_bbox=cell_bbox,
                resolution=res,
                time=time,
                target_crs=target_crs,
                halo_pct=0.10,
            )

            new_attrs = {
                "policy_hash": key_hash,
                "policy_legend": str(legend),
                "grid_cell_size": grid_cell_size,
                "crs": target_crs,
                "created_at": datetime.utcnow().isoformat(),
                "cell_bbox": list(cell_bbox),
                "hydrated": "true",
            }
            ds_cell.attrs.update(new_attrs)
            if "provenance_dict" in ds_cell.attrs:
                ds_cell.attrs["provenance_dict_json"] = json.dumps(ds_cell.attrs["provenance_dict"])

            # Save to Cache
            ds_chunked = ds_cell.chunk({"y": 2048, "x": 2048})
            tmp_path = cache_path.with_suffix(f".tmp.{key_hash}.zarr")

            if tmp_path.exists():
                import shutil

                shutil.rmtree(tmp_path)

            ds_chunked.to_zarr(tmp_path, mode="w", consolidated=should_consolidate())

            if cache_path.exists():
                import shutil

                shutil.rmtree(cache_path)

            import shutil

            shutil.move(str(tmp_path), str(cache_path))

            stats["processed"] += 1
            ds_cell.close()

        except Exception as e:
            logger.error(f"Failed to hydrate cell {cell_bbox}: {e}", exc_info=True)
            stats["failed"] += 1

    logger.info(f"Hydration complete. Stats: {stats}")
    return stats


def run(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
    use_cache: bool = True,
) -> xr.Dataset:
    """
    Execute a fusion policy to generate a topobathymetric dataset.
    This public entrypoint chunks arbitrary bboxes into standard grid cells for caching.
    """
    policy = _resolve_policy(policy_input)
    target_crs = policy.crs

    # If the bbox exactly matches a single grid cell, we can bypass the expensive tiling logic
    # and directly check/write the singular cache entry.
    # This is critical for /elevation hits because it requests the exact PADDED tile bbox.
    is_standard_cell = False
    try:
        # We check if the requested bbox matches what _get_grid_cells would produce for a single tile
        cells, _ = _get_grid_cells(bbox, target_crs)
        if len(cells) == 1:
            c = cells[0]
            # Floating point tolerance
            if all(abs(a - b) < 1e-10 for a, b in zip(bbox, c, strict=False)):
                is_standard_cell = True
    except Exception:
        pass

    if is_standard_cell:
        # FAST PATH: Single cache-hit check for the whole request
        cache_path, key_hash = get_fused_cache_info(
            policy_input,
            bbox,
            resolution=resolution if resolution else 30.0,
        )
        if use_cache and cache_path.exists():
            logger.info(f"Fast Path Cache Hit: {cache_path.name}")
            ds = xr.open_dataset(cache_path, engine="zarr", chunks="auto", decode_coords="all")
            return cast(xr.Dataset, ds.load())

    start_lon, start_lat, end_lon, end_lat = bbox

    # Validate CRS Area of Use
    try:
        crs_obj = CRS(target_crs)
        if crs_obj.area_of_use:
            # Area of Use bounds are (west, south, east, north)
            aou_w, aou_s, aou_e, aou_n = crs_obj.area_of_use.bounds

            # Simple intersection check
            if start_lon > aou_e or end_lon < aou_w or start_lat > aou_n or end_lat < aou_s:
                error_msg = (
                    f"Requested bounds {bbox} are outside the validity area for policy CRS {target_crs}. "
                    f"Area of Use: {crs_obj.area_of_use.name} ({aou_w}, {aou_s}, {aou_e}, {aou_n})"
                )
                raise ValueError(error_msg)
    except ImportError:
        pass
    except ValueError as e:
        raise e
    except Exception:
        pass

    # Determine Grid Cells
    cells, grid_cell_size = _get_grid_cells(bbox, target_crs)

    # For caching
    import os

    default_cache = "~/.cache/topobathysim/fused_zarr"
    cache_dir_str = os.environ.get("TOPOBATHYSIM_CACHE_DIR", default_cache)
    if cache_dir_str != default_cache and not cache_dir_str.endswith("fused_zarr"):
        cache_dir = Path(cache_dir_str).expanduser() / "fused_zarr"
    else:
        cache_dir = Path(cache_dir_str).expanduser()

    cache_dir.mkdir(parents=True, exist_ok=True)

    legend = generate_provider_legend(policy)

    # Process all cells
    cell_datasets = []

    for cell_bbox in cells:
        # Use centralized cache helper to ensure metadata parity across all endpoints
        cache_path, key_hash = get_fused_cache_info(
            policy_input,
            cell_bbox,
            resolution=resolution if resolution else 30.0,
        )

        ds_cell = None
        if use_cache and cache_path.exists():
            try:
                ds_cell = xr.open_dataset(cache_path, engine="zarr", chunks="auto", decode_coords="all")
                # Load fully
                ds_cell = ds_cell.load()

                valid_grid = ds_cell.attrs.get("grid_cell_size") == grid_cell_size
                # If cached grid size doesn't match current logic, invalidate
                if not valid_grid:
                    logger.info(f"Cache Grid Mismatch for {cache_path.name}. Invalidating.")
                    ds_cell = None
                else:
                    # Clean up random dimensions
                    for var in ["elevation", "source_elevation"]:
                        if var in ds_cell and "band" in ds_cell[var].dims:
                            ds_cell[var] = ds_cell[var].squeeze("band", drop=True)

                    # Double check legend
                    cached_legend = ds_cell.attrs.get("policy_legend")
                    if cached_legend and cached_legend != str(legend):
                        logger.info(f"Policy Legend Mismatch for {cache_path.name}. Invalidating Cache.")
                        ds_cell = None

            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                ds_cell = None

        if ds_cell is None:
            # Cache miss! Execute the cell
            ds_cell = _run_cell(
                policy=policy,
                cell_bbox=cell_bbox,
                resolution=resolution,
                time=time,
                target_crs=target_crs,
                halo_pct=0.10,
            )

            # Attributes
            new_attrs = {
                "policy_hash": hash_policy(policy.model_dump()),  # type: ignore
                "policy_legend": str(legend),
                "grid_cell_size": grid_cell_size,
                "crs": target_crs,
                "created_at": datetime.utcnow().isoformat(),
                "cell_bbox": list(cell_bbox),
            }
            ds_cell.attrs.update(new_attrs)

            if "provenance_dict" in ds_cell.attrs:
                # Keep it as is or JSON dump it
                ds_cell.attrs["provenance_dict_json"] = json.dumps(ds_cell.attrs["provenance_dict"])

            # Attempt to save to cache
            if use_cache:
                try:
                    logger.info(f"Writing Fused Grid Cell Zarr to {cache_path}")
                    ds_chunked = ds_cell.chunk({"y": 2048, "x": 2048})
                    tmp_path = cache_path.with_suffix(f".tmp.{key_hash}.zarr")

                    if tmp_path.exists():
                        import shutil

                        shutil.rmtree(tmp_path)

                    ds_chunked.to_zarr(tmp_path, mode="w", consolidated=should_consolidate())

                    if cache_path.exists():
                        import shutil

                        shutil.rmtree(cache_path)

                    import shutil

                    shutil.move(str(tmp_path), str(cache_path))
                except Exception as e:
                    logger.error(f"Failed to cache cell {cache_path}: {e}")

        cell_datasets.append(ds_cell)

    # Mosaic the cells together
    if len(cell_datasets) == 1:
        merged_ds = cell_datasets[0]
    else:
        # Sort coordinates to ensure monotonic indexes before combining.
        sorted_cells = [ds.sortby("y", ascending=False).sortby("x", ascending=True) for ds in cell_datasets]
        # combine_attrs="override" discards per-cell attrs (created_at, cell_bbox); reapplied below.
        try:
            merged_ds = cast(xr.Dataset, xr.combine_by_coords(sorted_cells, combine_attrs="override"))
        except ValueError:
            # Fallback: chain combine_first so NaN never overrides valid data.
            # xr.merge(compat="override") uses first-value-wins which lets NaN from a cell that
            # doesn't cover a coordinate silently mask valid data from another cell.
            merged_ds = sorted_cells[0]
            for next_ds in sorted_cells[1:]:
                merged_ds = cast(xr.Dataset, merged_ds.combine_first(next_ds))

        # combine_by_coords sorts all coords ascending; restore north-up Y ordering.
        merged_ds = merged_ds.sortby("y", ascending=False).sortby("x", ascending=True)

        # Drop float duplicate coordinates that can arise at grid cell boundaries to prevent
        # downstream "Reindexing only valid with uniquely valued Index" errors.
        merged_ds = merged_ds.drop_duplicates(dim="x").drop_duplicates(dim="y")

    # Clip to the exact requested bbox, reprojecting bounds to target CRS if needed.
    if target_crs != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        r_min_x, r_min_y = transformer.transform(start_lon, start_lat)
        r_max_x, r_max_y = transformer.transform(end_lon, end_lat)
        req_min_x, req_max_x = min(r_min_x, r_max_x), max(r_min_x, r_max_x)
        req_min_y, req_max_y = min(r_min_y, r_max_y), max(r_min_y, r_max_y)
    else:
        req_min_x, req_min_y = start_lon, start_lat
        req_max_x, req_max_y = end_lon, end_lat

    # Use item access for coordinates to avoid potential AttributeError on property access
    # if dataset structure is irregular
    x_coords = merged_ds["x"]
    # Fallback for single pixel or malformed grids
    epsilon = float(abs((x_coords[1] - x_coords[0]).values * 0.1)) if len(x_coords) > 1 else 0.0001

    # Ensure strictly python floats for slicing
    req_min_x = float(req_min_x)
    req_max_x = float(req_max_x)
    req_min_y = float(req_min_y)
    req_max_y = float(req_max_y)

    # Re-verify numeric index to prevent slicing errors
    from pandas.api.types import is_numeric_dtype

    if not is_numeric_dtype(merged_ds.indexes["x"].dtype):
        logger.warning(f"X Index is not numeric: {merged_ds.indexes['x'].dtype}. Slicing may fail.")

    merged_ds = merged_ds.sel(x=slice(req_min_x - epsilon, req_max_x + epsilon))
    if len(merged_ds.y) > 0 and merged_ds.y[0] > merged_ds.y[-1]:
        merged_ds = merged_ds.sel(y=slice(req_max_y + epsilon, req_min_y - epsilon))
    else:
        merged_ds = merged_ds.sel(y=slice(req_min_y - epsilon, req_max_y + epsilon))

    # Add combined attributes
    global_provenance: dict[int, dict[str, str]] = {}
    for cell_ds in cell_datasets:
        if "provenance_dict" in cell_ds.attrs:
            for k, v in cell_ds.attrs["provenance_dict"].items():
                global_provenance[int(k)] = v
        elif "provenance_dict_json" in cell_ds.attrs:
            loaded_dict = json.loads(cell_ds.attrs["provenance_dict_json"])
            for k, v in loaded_dict.items():
                global_provenance[int(k)] = v

    merged_ds.attrs = {
        "policy_hash": hash_policy(policy.model_dump()),  # type: ignore
        "policy_legend": str(legend),
        "crs": target_crs,
        "mosaiced_from_cells": len(cells),
        "provenance_dict": global_provenance,
    }
    # PROBE FINAL RESULT
    m_elev = merged_ds["elevation"]
    m_nans = int(np.isnan(m_elev).sum())
    m_total = m_elev.size
    logger.info(f"[PROBE] Final: Shape={m_elev.shape}, NaNs={m_nans}/{m_total} ({100.0*m_nans/m_total:.1f}%)")
    return cast(xr.Dataset, merged_ds)
