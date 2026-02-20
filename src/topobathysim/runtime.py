"""
Core Runtime for TopoBathySim.

This module executes fusion policies to generate topobathymetric datasets.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from pyproj import Transformer

from topobathysim.operators.blend import metric_feather, overwrite
from topobathysim.policy.loader import generate_provider_legend, hash_policy, load_policy
from topobathysim.policy.schema import OperatorType
from topobathysim.providers.registry import registry

logger = logging.getLogger(__name__)


def run(
    policy_path: str,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
    use_cache: bool = True,
) -> xr.Dataset:
    """
    Execute a fusion policy to generate a topobathymetric dataset.

    Args:
        policy_path: Path to the fusion policy YAML file.
        bbox: Input Bounding box in EPSG:4326 (min_lon, min_lat, max_lon, max_lat).
        resolution: Grid resolution. Units depend on Policy CRS:
                    - Projected CRS: Meters
                    - Geographic CRS: Meters (will be auto-converted to degrees)
        time: Optional timestamp for time-dependent queries (Sprint v2).
        use_cache: If True, look for/save reuse fused Zarr datasets.

    Returns:
        xr.Dataset: The generated dataset containing 'elevation' and 'source_elevation'.

    Note:
        **Edge Case**: If a policy rule specifies `operator: metric_feather` but omits `blend_distance`
        (i.e., it is None), the runtime falls back to `overwrite` (Hard Cut) behavior.
    """
    # 1. Load Policy
    policy = load_policy(policy_path)
    target_crs = policy.crs

    start_lon, start_lat, end_lon, end_lat = bbox

    # Validate CRS Area of Use
    try:
        from pyproj import CRS

        crs_obj = CRS(target_crs)
        if crs_obj.area_of_use:
            # Area of Use bounds are (west, south, east, north)
            aou_w, aou_s, aou_e, aou_n = crs_obj.area_of_use.bounds

            # Simple intersection check (1D overlap in both X and Y)
            # Fail if completely disjoint
            if start_lon > aou_e or end_lon < aou_w or start_lat > aou_n or end_lat < aou_s:
                error_msg = (
                    f"Requested bounds {bbox} are outside the validity area for policy CRS {target_crs}. "
                    f"Area of Use: {crs_obj.area_of_use.name} ({aou_w}, {aou_s}, {aou_e}, {aou_n})"
                )
                # User asked for loud failure
                raise ValueError(error_msg)
    except ImportError:
        pass  # pyproj not installed? Unlikely given usage
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        # Don't block execution if validation itself fails weirdly (e.g. CRS lookup fail)
        # But maybe we should?
        pass

    # 2. Setup Canvas Extents & Resolution
    min_lon, min_lat, max_lon, max_lat = bbox

    # Reproject BBox if Target CRS is not EPSG:4326
    if target_crs != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)
        # Ensure min/max are correct orientation
        min_x, max_x = min(min_x, max_x), max(min_x, max_x)
        min_y, max_y = min(min_y, max_y), max(min_y, max_y)
    else:
        min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat

    # Resolve Resolution
    # Default to 30.0 meters if not provided
    input_resolution_meters = resolution if resolution is not None else 30.0

    # Smart Resolution Logic
    if target_crs == "EPSG:4326":
        # Convert Meters -> Degrees
        # 1 deg lat ~= 111,320m
        meters_per_deg_lat = 111320.0
        center_lat = (min_lat + max_lat) / 2
        meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))

        res_x = input_resolution_meters / meters_per_deg_lon
        # Use aspect-corrected square pixels (res_lat approx same as res_lon in meters)
        res_y = input_resolution_meters / meters_per_deg_lat
    else:
        # Projected CRS uses meters directly
        res_x = input_resolution_meters
        res_y = input_resolution_meters

    # --- Caching Check ---
    cache_path = None
    if use_cache:
        try:
            cache_dir = Path("~/.cache/topobathysim/fused_zarr").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Hash Key: Policy Content + BBox + Resolution + CRS
            # Round floats to 6 decimals to avoid micro-mismatches (parity with legacy)
            key_dict = {
                "policy": policy.model_dump(),  # type: ignore
                "bbox": [round(x, 6) for x in [min_x, min_y, max_x, max_y]],
                "res": [round(x, 6) for x in [res_x, res_y]],
                "crs": target_crs,
            }
            # Use json.dumps with sort_keys for stability
            key_str = json.dumps(key_dict, sort_keys=True, default=str)
            key_hash = hashlib.md5(key_str.encode()).hexdigest()

            cache_path = cache_dir / f"{key_hash}.zarr"

            if cache_path.exists():
                logger.info(f"Fused Zarr Cache Hit: {cache_path}")
                # Use FileLock if available, otherwise just try open
                try:
                    from filelock import FileLock

                    with FileLock(cache_path.with_suffix(".lock")):
                        ds = xr.open_dataset(cache_path, engine="zarr", chunks="auto", decode_coords="all")
                        return ds
                except ImportError:
                    ds = xr.open_dataset(cache_path, engine="zarr", chunks="auto", decode_coords="all")
                    return ds
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_path}: {e}")
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

    # 3. Initialize Canvas
    width = int((max_x - min_x) / res_x)
    height = int((max_y - min_y) / res_y)

    # Affine Transform (Standard: North-Up, so y scale is negative)
    transform = Affine.translation(min_x, max_y) * Affine.scale(res_x, -res_y)

    xs = np.linspace(min_x + res_x / 2, max_x - res_x / 2, width)
    ys = np.linspace(max_y - res_y / 2, min_y + res_y / 2, height)

    elevation = xr.DataArray(
        data=np.full((height, width), np.nan, dtype=np.float32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="elevation",
    )
    elevation.rio.write_crs(target_crs, inplace=True)
    elevation.rio.write_transform(transform, inplace=True)

    source_elevation = xr.DataArray(
        data=np.full((height, width), 0, dtype=np.uint16),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="source_elevation",
    )
    source_elevation.rio.write_crs(target_crs, inplace=True)
    source_elevation.rio.write_transform(transform, inplace=True)

    # 4. Generate Provider Legend & IDs
    legend = generate_provider_legend(policy)
    provider_to_id = {v: k for k, v in legend.items()}

    # 5. Execution Loop
    for variable in policy.variables:
        if variable.name != "elevation":
            continue

        if not np.isnan(variable.background):
            elevation = elevation.fillna(variable.background)

        for step in variable.steps:
            provider_cls = registry.get_provider_class(step.provider)
            provider_instance = provider_cls()

            try:
                # Fetch using the original EPSG:4326 bbox.
                # The provider handles reprojection if necessary or optimized.
                fetched_data = provider_instance.fetch_layer(
                    bbox,
                    resolution=input_resolution_meters,
                    crs=target_crs,
                    filter=step.filter,
                )
            except (KeyError, ValueError, RuntimeError):
                # Skip provider if data is missing or fetch fails (common for sparse datasets)
                continue

            if fetched_data is None:
                continue

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
            aligned_data = fetched_data.rio.reproject_match(elevation)
            new_data_mask = aligned_data.notnull()

            if not new_data_mask.any():
                continue

            provider_id = provider_to_id.get(step.provider, 0)

            # --- Pairwise Logic ---
            # 1. Initialize mask of pixels to be processed by default rule
            remaining_mask = new_data_mask.copy()

            # Apply Transitions (Specific Overrides)
            for rule in step.transitions:
                target_id = provider_to_id.get(rule.target_provider)
                if target_id is None:
                    continue

                # Identify overlap: Valid New Data AND Underlying Source == TargetID
                transition_mask = new_data_mask & (source_elevation == target_id)

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
            source_elevation = xr.where(new_data_mask, provider_id, source_elevation)  # type: ignore

    # 6. Finalize Dataset
    ds = xr.Dataset(
        {"elevation": elevation, "source_elevation": source_elevation},
        attrs={
            "policy_hash": hash_policy(policy.model_dump()),  # type: ignore
            "policy_legend": str(legend),
            "crs": target_crs,
        },
    )

    # 7. Write to Cache
    if cache_path:
        try:
            logger.info(f"Writing Fused Zarr to {cache_path}")

            # Metadata Attributes
            ds.attrs["created_at"] = datetime.utcnow().isoformat()
            ds.attrs["policy_hash"] = key_hash
            ds.attrs["crs"] = target_crs

            # Chunking is important for Zarr performance
            ds_chunked = ds.chunk({"y": 2048, "x": 2048})

            # Atomic Write Pattern
            tmp_path = cache_path.with_suffix(f".tmp.{key_hash}.zarr")

            try:
                # 1. Write to temp directory
                if tmp_path.exists():
                    import shutil

                    shutil.rmtree(tmp_path)

                ds_chunked.to_zarr(tmp_path, mode="w", consolidated=True)

                # 2. Rename to final path (Atomic)
                if cache_path.exists():
                    import shutil

                    shutil.rmtree(cache_path)

                tmp_path.rename(cache_path)

            except Exception as e:
                logger.warning(f"Failed to write atomic cache {cache_path}: {e}")
                if tmp_path.exists():
                    import shutil

                    shutil.rmtree(tmp_path)

        except Exception as e:
            logger.warning(f"Failed to write cache {cache_path}: {e}")

    return ds
