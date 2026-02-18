"""
Core Runtime for TopoBathySim.

This module executes fusion policies to generate topobathymetric datasets.
"""

from datetime import datetime

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from pyproj import Transformer

from topobathysim.operators.blend import metric_feather, overwrite
from topobathysim.policy.loader import generate_provider_legend, hash_policy, load_policy
from topobathysim.policy.schema import OperatorType
from topobathysim.providers.registry import registry


def run(
    policy_path: str,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
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

    Returns:
        xr.Dataset: The generated dataset containing 'elevation' and 'source_elevation'.

    Note:
        **Edge Case**: If a policy rule specifies `operator: metric_feather` but omits `blend_distance`
        (i.e., it is None), the runtime falls back to `overwrite` (Hard Cut) behavior.
    """
    # 1. Load Policy
    policy = load_policy(policy_path)
    target_crs = policy.crs

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
                )
            except (KeyError, ValueError, RuntimeError):
                # Skip provider if data is missing or fetch fails (common for sparse datasets)
                continue

            if fetched_data is None:
                continue

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
            "policy_hash": hash_policy(policy.model_dump()),
            "policy_legend": str(legend),
            "crs": target_crs,
        },
    )

    return ds
