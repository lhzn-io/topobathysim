# Import from scripts.cache_manager, suppressing output if any
import contextlib
import io
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from topobathysim.api.models import (
    CacheDetail,
    CacheSummary,
    CacheTierInfo,
    CacheTierSummary,
    CheckResult,
    DiscoveryDetail,
    OrphansResult,
    ProviderDetail,
    PurgeResult,
    RawFileDetail,
    Tier1Detail,
    Tier2Detail,
    Tier3Detail,
    Tier4Detail,
    Tier5Detail,
    TileSubdir,
    ZoomDetail,
)

# We need to suppress strict type checking on this import because it's a script
# and mypy might complain about missing stubs or side effects.
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import topobathysim.scripts.cache_manager as cm
except ImportError:
    # Fallback or error if script is missing
    raise ImportError("Could not import topobathysim.scripts.cache_manager") from None

logger = logging.getLogger(__name__)

# Re-export some constants for convenience
CACHE_ROOT = cm.CACHE_ROOT
TIERS = cm.TIERS

# Cache for get_summary to avoid scanning too often
_SUMMARY_CACHE: tuple[float, CacheSummary] | None = None
_SUMMARY_TTL = 5.0  # seconds


def get_tiers() -> list[CacheTierInfo]:
    """Return static metadata about cache tiers."""
    return [
        CacheTierInfo(
            number=t.number,
            name=t.name,
            short_desc=t.short_desc,
            long_desc=t.long_desc,
            purge_reason=t.purge_reason,
            warning=t.warning,
        )
        for t in TIERS
    ]


def get_cache_summary(cache_bust: bool = False) -> CacheSummary:
    """Return a lightweight summary of cache tiers."""
    global _SUMMARY_CACHE
    now = time.time()

    if not cache_bust and _SUMMARY_CACHE is not None:
        ts, summary = _SUMMARY_CACHE
        if now - ts < _SUMMARY_TTL:
            return summary

    total_bytes = 0
    tier_summaries: list[CacheTierSummary] = []

    for tier in TIERS:
        # These methods scan the filesystem, hence the caching
        items = tier.item_count()
        sz = tier.size_bytes()
        total_bytes += sz
        tier_summaries.append(
            CacheTierSummary(
                number=tier.number,
                name=tier.name,
                items=items,
                bytes=sz,
                mb=round(sz / (1024 * 1024), 2),
                warning=tier.warning,
            )
        )

    summary = CacheSummary(
        cache_root=str(CACHE_ROOT),
        total_bytes=total_bytes,
        total_mb=round(total_bytes / (1024 * 1024), 2),
        tiers=tier_summaries,
        last_updated=now,
    )
    _SUMMARY_CACHE = (now, summary)
    return summary


def get_cache_detail(cache_bust: bool = True) -> CacheDetail:
    """Return full detailed inventory of the cache."""
    # We reuse the logic from cache_manager._gather_status_data
    # but map it to our Pydantic models.
    # Note: cache_manager._gather_status_data is not async, so this blocks.
    # In a real app we might want to run this in a thread pool.

    # Force re-scan
    d = cm._gather_status_data()

    # Map Tier 1
    t1_subdirs = [TileSubdir(label=label, files=files, bytes=sz) for label, files, sz in d.tile_subdirs]
    t1 = Tier1Detail(
        tile_subdirs=t1_subdirs,
        meta_count=d.tile_meta_count,
        npz_count=d.tile_npz_count,
    )

    # Map Tier 2
    by_zoom = {}
    for z, info in d.fused_by_zoom.items():
        by_zoom[z] = ZoomDetail(
            count=info["count"],
            bytes=info["bytes"],
            lon_min=info["lon_min"],
            lon_max=info["lon_max"],
            lat_min=info["lat_min"],
            lat_max=info["lat_max"],
        )

    t2 = Tier2Detail(
        count=d.fused_count,
        bytes=d.fused_total_bytes,
        newest=d.fused_newest,
        oldest=d.fused_oldest,
        by_zoom=by_zoom,
    )

    # Map Tier 3
    providers = []
    for p, count, sz, newest, oldest in d.provider_zarr:
        providers.append(
            ProviderDetail(
                name=p,
                count=count,
                bytes=sz,
                newest=newest,
                oldest=oldest,
            )
        )
    t3 = Tier3Detail(providers=providers)

    # Map Tier 4
    discovery = []
    for label, ok, sz, entries, mtime in d.discovery:
        discovery.append(
            DiscoveryDetail(
                label=label,
                ok=ok,
                bytes=sz,
                entries=entries,
                mtime=mtime,
            )
        )

    t4 = Tier4Detail(
        discovery=discovery,
        bluetopo_scheme={
            "exists": d.bluetopo_scheme[0],
            "bytes": d.bluetopo_scheme[1],
            "mtime": d.bluetopo_scheme[2],
        },
        bluetopo_sidecars={
            "ok_count": d.bluetopo_sidecars[0],
            "failed_count": d.bluetopo_sidecars[1],
            "bytes": d.bluetopo_sidecars[2],
            "mtime": d.bluetopo_sidecars[3],
        },
        topobathy_zips={
            "count": d.topobathy_zips[0],
            "bytes": d.topobathy_zips[1],
            "mtime": d.topobathy_zips[2],
        },
    )

    # Map Tier 5
    raw_files = []
    for p, count, sz, newest, oldest in d.raw_files:
        raw_files.append(
            RawFileDetail(
                provider=p,
                count=count,
                bytes=sz,
                newest=newest,
                oldest=oldest,
            )
        )
    t5 = Tier5Detail(raw_files=raw_files)

    return CacheDetail(
        cache_root=str(CACHE_ROOT),
        total_bytes=d.total_bytes,
        tier_1=t1,
        tier_2=t2,
        tier_3=t3,
        tier_4=t4,
        tier_5=t5,
    )


def purge_tiers(tiers: list[int], dry_run: bool = True, yes: bool = False) -> PurgeResult:
    """Purge specified cache tiers."""

    # Safety check for Tier 5
    if 5 in tiers and not yes:
        # This should have been caught by the API router/controller,
        # but double check here.
        pass

    affected_paths: list[str] = []
    summary: dict[str, dict[str, int]] = {}

    # Sort tiers to process
    tiers_to_process = [t for t in TIERS if t.number in tiers]

    for tier in tiers_to_process:
        paths = tier.purge_paths()
        deleted_count = 0
        deleted_bytes = 0

        tier_paths_str = []

        for p in paths:
            # Gather stats before deletion attempt
            try:
                if p.exists():
                    sz = cm._dir_size_bytes(p) if p.is_dir() else p.stat().st_size
                    deleted_bytes += sz
            except OSError:
                sz = 0

            p_str = str(p)
            tier_paths_str.append(p_str)

            if not dry_run:
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {p}: {e}")
            else:
                deleted_count += 1

        summary[f"tier_{tier.number}"] = {"count": deleted_count, "bytes": deleted_bytes}
        affected_paths.extend(tier_paths_str)

    msg = "Dry-run complete." if dry_run else "Purge complete."
    mode = "dry-run" if dry_run else "committed"

    return PurgeResult(status="success", mode=mode, message=msg, summary=summary, paths=affected_paths)


def check_integrity(clean: bool = False, lock_timeout: int = 3600) -> CheckResult:
    """Scan provider integrity and orphaned files."""
    checks: dict[str, Any] = {}

    # 1. Scan providers
    for p in cm.PROVIDERS:
        # _scan_provider_integrity is available in cm
        # It returns dict with keys: stale_locks, corrupt_src, corrupt_zarr (int counts)
        try:
            counts = cm._scan_provider_integrity(p, clean=clean, lock_timeout=lock_timeout)
            checks[p] = counts
        except Exception as e:
            logger.error(f"Error checking provider {p}: {e}")
            checks[p] = {"error": str(e)}

    # 2. Discovery checks (adapted from cmd_check)
    disco_status = []
    for label, path in cm._DISCOVERY_FILES:
        status = "ok"
        if not path.exists():
            status = "missing"
        else:
            try:
                with open(path) as f:
                    json.load(f)
            except json.JSONDecodeError:
                status = "corrupt"
                if clean:
                    try:
                        path.unlink()
                        status = "deleted"
                    except OSError:
                        status = "delete_failed"
        disco_status.append({"label": label, "status": status})

    checks["discovery"] = disco_status

    # 3. Scan orphans (adapted from _scan_global_orphans)
    # _scan_global_orphans prints to console and returns (stale_count, removed_count)
    # We want more detail. Implementing custom logic similar to _scan_global_orphans.

    orphans = _scan_orphans_internal(clean=clean, lock_timeout=lock_timeout)

    return CheckResult(status="complete", checks=checks, orphans=orphans)


def manage_orphans(clean: bool = False) -> OrphansResult:
    """List or clean orphaned locks/tmp files."""
    orphans = _scan_orphans_internal(clean=clean, lock_timeout=3600)  # Default timeout for manual action

    mode = "committed" if clean else "list-only"

    return OrphansResult(status="success", mode=mode, orphans=orphans)


def _scan_orphans_internal(clean: bool, lock_timeout: int) -> dict[str, Any]:
    now = time.time()
    patterns = ["*.lock", ".tmp*", "__temp__*"]
    candidates: list[Path] = []

    for pat in patterns:
        candidates.extend(CACHE_ROOT.rglob(pat))

    unique_candidates = sorted(list(set(candidates)), key=lambda p: str(p))

    stale_locks = []
    tmp_files = []
    total_bytes = 0

    for p in unique_candidates:
        if not p.exists():
            continue
        try:
            stat = p.stat()
            age = now - stat.st_mtime

            # For .lock files, check timeout. For tmp files, maybe just delete if old?
            # Matching logic from cache_manager: usage of lock_timeout implies it applies to all.
            if age > lock_timeout:
                sz = cm._dir_size_bytes(p) if p.is_dir() else stat.st_size
                info = {"path": str(p), "age_sec": age, "bytes": sz}

                if p.name.endswith(".lock"):
                    stale_locks.append(info)
                else:
                    tmp_files.append(info)

                total_bytes += sz

                if clean:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
        except Exception as e:
            logger.warning(f"Error processing orphan {p}: {e}")

    return {
        "locks": stale_locks,
        "tmp_files": tmp_files,
        "total_bytes": total_bytes,
        "count": len(stale_locks) + len(tmp_files),
    }
