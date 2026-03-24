"""DEM Viewer API — serves fused zarr cells as mosaiced Float32 binaries for 3D visualization."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dem", tags=["DEM Viewer"])

_DEFAULT_CACHE = "~/.cache/topobathysim/fused_zarr"

# Tolerance for resolution matching: 5% handles latitude-dependent pixel size
# variation in projected CRS (e.g. EPSG:3857 L13 tiles: dx ≈ 14.44-14.46m).
_RES_TOLERANCE = 0.05


def _fused_zarr_root() -> Path:
    cache_dir_str = os.environ.get("TOPOBATHYSIM_CACHE_DIR", _DEFAULT_CACHE)
    if cache_dir_str != _DEFAULT_CACHE and not cache_dir_str.endswith("fused_zarr"):
        return Path(cache_dir_str).expanduser() / "fused_zarr"
    return Path(cache_dir_str).expanduser()


def _read_cell_attrs(cell_path: Path) -> dict[str, Any] | None:
    """Read attrs + coordinate extents from a zarr cell without loading data."""
    try:
        ds = xr.open_dataset(cell_path, engine="zarr", decode_coords="all")
        attrs = dict(ds.attrs)
        # Grab coordinate spacing and extents in native CRS
        x_vals = ds.x.values if "x" in ds.coords else None
        y_vals = ds.y.values if "y" in ds.coords else None
        if x_vals is not None and len(x_vals) > 1:
            attrs["_dx"] = abs(float(x_vals[1] - x_vals[0]))
            attrs["_x_min"] = float(min(x_vals))
            attrs["_x_max"] = float(max(x_vals))
            attrs["_nx"] = len(x_vals)
        if y_vals is not None and len(y_vals) > 1:
            attrs["_dy"] = abs(float(y_vals[1] - y_vals[0]))
            attrs["_y_min"] = float(min(y_vals))
            attrs["_y_max"] = float(max(y_vals))
            attrs["_ny"] = len(y_vals)
        ds.close()
        return attrs
    except Exception:
        return None


def _parse_cell_bbox(attrs: dict[str, Any]) -> list[float] | None:
    cb = attrs.get("cell_bbox")
    if cb is None:
        return None
    if isinstance(cb, str):
        cb = json.loads(cb)
    return list(cb)


def _cluster_cells(
    group: list[tuple[Path, dict[str, Any]]],
    threshold: float,
) -> list[list[tuple[Path, dict[str, Any]]]]:
    """Cluster cells by spatial proximity using single-linkage on bbox centroids.

    Two cells are neighbors if their centroid distance (Chebyshev / L-inf) is
    within `threshold`.  Returns a list of clusters, each a list of (path, attrs).
    """
    if len(group) <= 1:
        return [group]

    # Compute centroids from cell_bbox (geographic degrees)
    centroids: list[tuple[float, float]] = []
    for _, attrs in group:
        bb = _parse_cell_bbox(attrs)
        if bb:
            centroids.append(((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))
        else:
            centroids.append((0.0, 0.0))

    # Union-find for single-linkage clustering
    parent = list(range(len(group)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            dx = abs(centroids[i][0] - centroids[j][0])
            dy = abs(centroids[i][1] - centroids[j][1])
            if max(dx, dy) <= threshold:
                union(i, j)

    # Collect clusters
    clusters_map: dict[int, list[int]] = {}
    for i in range(len(group)):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    return [[group[i] for i in indices] for indices in clusters_map.values()]


@router.get("/datasets")
async def list_datasets(source: str | None = None) -> list[dict[str, Any]]:
    """List fused datasets, grouped by policy hash, resolution, and spatial cluster.

    Args:
        source: Filter by source type ("hydrate" or "tile"). If omitted, returns all.
    """
    root = _fused_zarr_root()
    return _list_datasets_sync(root, source_filter=source)


def _mosaic_to_bin(
    policy_dir: Path,
    dx: float,
    dy: float,
    width: int,
    height: int,
    x_min: float,
    y_max: float,
    cluster_bbox: list[float] | None = None,
) -> np.ndarray:
    """Paint zarr cells into a pre-allocated numpy canvas (north-first, row-major).

    Instead of xr.combine_by_coords (which is slow and fragile with floating-point
    coordinate misalignment), we allocate a single NaN-filled float32 grid and blit
    each cell into the correct position using coordinate → index math.  This is
    O(total_pixels) with zero intermediate copies, and naturally handles overlaps
    (last-writer-wins, same as combine_first).
    """
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    cells_painted = 0

    for cell_path in sorted(policy_dir.glob("*.zarr")):
        try:
            # Quick cluster filter before loading data
            if cluster_bbox is not None:
                attrs = _read_cell_attrs(cell_path)
                if attrs and not _cell_in_cluster(attrs, cluster_bbox):
                    continue

            ds = xr.open_dataset(cell_path, engine="zarr", decode_coords="all")
            x_vals = ds.x.values if "x" in ds.coords else None
            if x_vals is None or len(x_vals) < 2:
                ds.close()
                continue
            cell_dx = abs(float(x_vals[1] - x_vals[0]))
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                ds.close()
                continue

            y_vals = ds.y.values
            elev = ds["elevation"].values.astype(np.float32)
            ds.close()

            # Compute pixel offsets into the canvas
            # Canvas is north-first: row 0 = y_max
            col_start = int(round((float(min(x_vals)) - x_min) / dx))
            row_start = int(round((y_max - float(max(y_vals))) / dy))

            # Ensure elevation is north-first (descending y)
            if len(y_vals) > 1 and y_vals[0] < y_vals[-1]:
                elev = elev[::-1]

            # Clip to canvas bounds (handles halo overlap)
            src_r0 = max(0, -row_start)
            src_c0 = max(0, -col_start)
            dst_r0 = max(0, row_start)
            dst_c0 = max(0, col_start)
            copy_h = min(elev.shape[0] - src_r0, height - dst_r0)
            copy_w = min(elev.shape[1] - src_c0, width - dst_c0)

            if copy_h <= 0 or copy_w <= 0:
                continue

            # Blit: only overwrite NaN pixels (first-valid-wins, like combine_first)
            src_slice = elev[src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w]
            dst_slice = canvas[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w]
            mask = np.isnan(dst_slice) & ~np.isnan(src_slice)
            dst_slice[mask] = src_slice[mask]
            cells_painted += 1

        except Exception as e:
            logger.warning(f"Failed to paint cell {cell_path.name}: {e}")

    logger.info(f"Mosaic complete: {cells_painted} cells -> {width}x{height} canvas")
    return canvas


def _cell_in_cluster(attrs: dict[str, Any], cluster_bbox: list[float] | None) -> bool:
    """Check if a cell's geographic bbox overlaps the cluster bbox."""
    if cluster_bbox is None:
        return True
    bb = _parse_cell_bbox(attrs)
    if bb is None:
        return True  # can't filter, include it
    # Simple overlap test (bboxes are [west, south, east, north])
    return not (
        bb[2] < cluster_bbox[0]
        or bb[0] > cluster_bbox[2]
        or bb[3] < cluster_bbox[1]
        or bb[1] > cluster_bbox[3]
    )


@router.get("/datasets/{dataset_id}/meta")
async def get_dataset_meta(dataset_id: str) -> dict[str, Any]:
    """Return metadata for a fused dataset."""
    root = _fused_zarr_root()
    policy_prefix, dx, dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    cells = sorted(policy_dir.glob("*.zarr"))
    cell_count = 0
    x_min_all = float("inf")
    x_max_all = float("-inf")
    y_min_all = float("inf")
    y_max_all = float("-inf")
    geo_bboxes: list[list[float]] = []
    crs = "EPSG:4326"
    legend = ""

    for cell_path in cells:
        attrs = _read_cell_attrs(cell_path)
        if attrs is None:
            continue
        cell_dx = attrs.get("_dx", 0)
        if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
            continue
        if not _cell_in_cluster(attrs, cluster_bbox):
            continue
        cell_count += 1
        x_min_all = min(x_min_all, attrs.get("_x_min", x_min_all))
        x_max_all = max(x_max_all, attrs.get("_x_max", x_max_all))
        y_min_all = min(y_min_all, attrs.get("_y_min", y_min_all))
        y_max_all = max(y_max_all, attrs.get("_y_max", y_max_all))
        bb = _parse_cell_bbox(attrs)
        if bb:
            geo_bboxes.append(bb)
        crs = attrs.get("crs", crs)
        legend = attrs.get("policy_legend", legend)

    if cell_count == 0:
        raise HTTPException(status_code=404, detail="No cells at this resolution")

    width = max(1, int(round((x_max_all - x_min_all) / dx)) + 1)
    height = max(1, int(round((y_max_all - y_min_all) / dy)) + 1)

    if geo_bboxes:
        bounds = [
            min(b[0] for b in geo_bboxes),
            min(b[1] for b in geo_bboxes),
            max(b[2] for b in geo_bboxes),
            max(b[3] for b in geo_bboxes),
        ]
    else:
        bounds = [x_min_all, y_min_all, x_max_all, y_max_all]

    if bounds and width > 1 and height > 1:
        geo_dx = (bounds[2] - bounds[0]) / (width - 1)
        geo_dy = (bounds[3] - bounds[1]) / (height - 1)
    else:
        geo_dx = dx
        geo_dy = dy

    return {
        "width": width,
        "height": height,
        "bounds": bounds,
        "dx": geo_dx,
        "dy": geo_dy,
        "crs": crs,
        "cell_count": cell_count,
        "policy_legend": legend,
    }


@router.get("/datasets/{dataset_id}/elevation.bin")
async def get_elevation_binary(dataset_id: str) -> Response:
    """Return the mosaiced elevation grid as a raw Float32 binary (north-first).

    On first request, the mosaic is built by painting cells into a numpy canvas
    (no xr.combine_by_coords) and cached to disk as a .bin file.  Subsequent
    requests serve the cached file directly.
    """
    root = _fused_zarr_root()
    policy_prefix, dx, dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    # Check for cached binary (invalidate if cell count changed)
    cache_bin = policy_dir / f"_mosaic_{dataset_id}.bin"
    cache_meta_file = policy_dir / f"_mosaic_{dataset_id}.json"

    cache_valid = False
    if cache_bin.exists() and cache_meta_file.exists():
        try:
            cached = json.loads(cache_meta_file.read_text())
            current_cells = 0
            for cp in policy_dir.glob("*.zarr"):
                a = _read_cell_attrs(cp)
                if not a:
                    continue
                if abs(a.get("_dx", 0) - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                    continue
                if not _cell_in_cluster(a, cluster_bbox):
                    continue
                current_cells += 1
            if cached.get("cell_count") == current_cells:
                cache_valid = True
            else:
                logger.info(
                    f"DEM mosaic cache stale: {cached.get('cell_count')} -> "
                    f"{current_cells} cells, rebuilding"
                )
        except Exception:
            pass

    if cache_valid:
        logger.info(f"DEM mosaic cache hit: {cache_bin.name}")
        content = cache_bin.read_bytes()
    else:
        meta = await get_dataset_meta(dataset_id)
        width = meta["width"]
        height = meta["height"]

        x_min = float("inf")
        y_max = float("-inf")
        for cell_path in sorted(policy_dir.glob("*.zarr")):
            attrs = _read_cell_attrs(cell_path)
            if attrs is None:
                continue
            cell_dx = attrs.get("_dx", 0)
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                continue
            if not _cell_in_cluster(attrs, cluster_bbox):
                continue
            x_min = min(x_min, attrs.get("_x_min", x_min))
            y_max = max(y_max, attrs.get("_y_max", y_max))

        canvas = _mosaic_to_bin(policy_dir, dx, dy, width, height, x_min, y_max, cluster_bbox)
        content = canvas.tobytes()

        try:
            cache_bin.write_bytes(content)
            cache_meta_file.write_text(
                json.dumps(
                    {
                        "width": width,
                        "height": height,
                        "cell_count": meta["cell_count"],
                    }
                )
            )
            logger.info(f"DEM mosaic cached: {cache_bin.name} ({len(content)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to cache mosaic: {e}")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}_elevation.bin"},
    )


@router.delete("/datasets/{dataset_id}/cache")
async def clear_dataset_cache(dataset_id: str) -> dict[str, Any]:
    """Delete the cached mosaic binary so the next load rebuilds from zarr cells."""
    root = _fused_zarr_root()
    policy_prefix, _dx, _dy, _bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    deleted = []
    for suffix in (".bin", ".json"):
        f = policy_dir / f"_mosaic_{dataset_id}{suffix}"
        if f.exists():
            f.unlink()
            deleted.append(f.name)

    logger.info(f"DEM cache cleared for {dataset_id}: {deleted}")
    return {"status": "cleared", "deleted": deleted}


def _parse_dataset_id(root: Path, dataset_id: str) -> tuple[str, float, float, list[float] | None]:
    """Parse dataset_id like '313a6513_15m' into (policy_prefix, dx, dy, cluster_bbox).

    Uses list_datasets() to find the matching dataset with its cluster bbox.
    The bbox is used to filter cells to the correct spatial cluster.
    """
    datasets = _list_datasets_sync(root)

    for ds in datasets:
        if ds["id"] == dataset_id:
            policy_hash_full = ds.get("policy_hash_full", "")
            # Find the policy dir
            for d in root.iterdir():
                if d.is_dir() and d.name.startswith(ds["policy_hash"][:8]):
                    return d.name, ds["dx"], ds["dy"], ds.get("bbox")
            # Try full hash
            for d in root.iterdir():
                if d.is_dir() and policy_hash_full.startswith(d.name):
                    return d.name, ds["dx"], ds["dy"], ds.get("bbox")

    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


def _list_datasets_sync(root: Path, source_filter: str | None = None) -> list[dict[str, Any]]:
    """Synchronous version of list_datasets for internal use."""
    if not root.exists():
        return []

    datasets: list[dict[str, Any]] = []
    for policy_dir in sorted(root.iterdir()):
        if not policy_dir.is_dir():
            continue
        cells = sorted(policy_dir.glob("*.zarr"))
        if not cells:
            continue

        res_groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for cell_path in cells:
            attrs = _read_cell_attrs(cell_path)
            if attrs is None:
                continue
            # Filter by source type if specified.
            # Cells without a source attr predate the source tag — treat as hydrate.
            cell_source = attrs.get("source", "hydrate")
            if source_filter and cell_source != source_filter:
                continue
            dx = attrs.get("_dx", 0)
            dy = attrs.get("_dy", 0)
            # For projected CRS (dx in meters), bucket by integer meter to
            # handle latitude-dependent pixel size variation within a tile tier.
            # E.g. L13 tiles at different latitudes have dx ≈ 14.44-14.46m.
            crs = attrs.get("crs", "EPSG:4326")
            if "4326" not in str(crs) and dx > 1:
                res_key = f"{round(dx)}m_{round(dy)}m"
            else:
                res_key = f"{dx:.8f}_{dy:.8f}"
            res_groups.setdefault(res_key, []).append((cell_path, attrs))

        for _res_key, group in res_groups.items():
            first_attrs = group[0][1]
            # Use directory name as the stable policy identifier, NOT the per-cell
            # policy_hash attr (which includes cell bbox and is unique per cell).
            policy_hash = policy_dir.name
            dx = first_attrs.get("_dx", 0)
            dy = first_attrs.get("_dy", 0)

            grid_cell_size = first_attrs.get("grid_cell_size", 0.05)
            threshold = grid_cell_size * 1.5
            clusters = _cluster_cells(group, threshold)

            for cluster_idx, cluster in enumerate(clusters):
                cluster_bboxes = [b for b in (_parse_cell_bbox(a) for _, a in cluster) if b is not None]
                if not cluster_bboxes:
                    continue

                union_bbox = [
                    min(b[0] for b in cluster_bboxes),
                    min(b[1] for b in cluster_bboxes),
                    max(b[2] for b in cluster_bboxes),
                    max(b[3] for b in cluster_bboxes),
                ]

                crs = cluster[0][1].get("crs", "EPSG:4326")
                if "4326" in str(crs):
                    lat_center = (union_bbox[1] + union_bbox[3]) / 2.0
                    res_m = dx * 111320 * np.cos(np.radians(lat_center))
                else:
                    res_m = dx

                suffix = f"_{cluster_idx}" if len(clusters) > 1 else ""
                dataset_id = f"{policy_hash[:8]}_{res_m:.0f}m{suffix}"

                datasets.append(
                    {
                        "id": dataset_id,
                        "policy_hash": policy_hash[:8],
                        "policy_hash_full": policy_hash,
                        "crs": crs,
                        "cell_count": len(cluster),
                        "resolution_m": round(res_m, 1),
                        "dx": dx,
                        "dy": dy,
                        "bbox": union_bbox,
                        "policy_legend": cluster[0][1].get("policy_legend", ""),
                        "source": cluster[0][1].get("source", "hydrate"),
                    }
                )

    return datasets


def _resolve_policy_dir(root: Path, prefix: str) -> Path:
    """Find a policy directory by prefix match."""
    exact = root / prefix
    if exact.is_dir():
        return exact

    matches = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous prefix '{prefix}' matches {len(matches)} datasets",
        )
    raise HTTPException(status_code=404, detail=f"Dataset '{prefix}' not found")
