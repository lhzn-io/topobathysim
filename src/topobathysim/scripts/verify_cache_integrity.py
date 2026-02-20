"""
TopoBathySim Cache Verification Tool
====================================

This script scans the local data cache (default: ~/.cache/topobathysim) for integrity issues.
It supports checking for:
- Corrupt source files (BAG, LAZ, TIFF) using native drivers.
- Corrupt Zarr caches (derived products).
- Stale lock files (.lock) left over from crashed processes.
- Abandoned temporary download files.

Usage:
    # 1. Dry Run (Check only)
    python src/topobathysim/scripts/verify_cache_integrity.py --check

    # 2. Cleanup (Delete corrupt/stale files)
    python src/topobathysim/scripts/verify_cache_integrity.py --clean

    # 3. Custom Timeout (e.g. locks considered stale after 2 hours)
    python src/topobathysim/scripts/verify_cache_integrity.py --check --lock-timeout 7200
"""

import argparse
import logging
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import laspy
import rasterio
import xarray as xr
from rasterio.errors import RasterioIOError

# Configure basics
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cache_verify")

CACHE_ROOT = Path("~/.cache/topobathysim").expanduser()

PROVIDERS = ["ncei_bag", "usgs_lidar", "noaa_bluetopo", "noaa_topobathy", "ncei_cudem", "usgs_3dep"]


@contextmanager
def suppress_fd_stderr() -> Iterator[None]:
    """
    Redirects the system level stderr file descriptor to /dev/null.
    """
    try:
        if not hasattr(sys.stderr, "fileno"):
            # Fallback for environments where stderr is not a real file
            yield
            return

        stderr_fd = sys.stderr.fileno()
        # Create a null file descriptor
        with open(os.devnull, "w") as devnull:
            devnull_fd = devnull.fileno()
            # Save the original stderr fd
            saved_stderr_fd = os.dup(stderr_fd)
            try:
                # Redirect stderr to null
                os.dup2(devnull_fd, stderr_fd)
                yield
            finally:
                # Restore stderr
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)
    except Exception:
        # If anything fails (e.g. no permissions, weird environment), just run without suppression
        yield


def verify_raster_integrity(file_path: Path) -> bool:
    """
    Checks if a raster file (BAG, TIFF) is valid using rasterio.
    Reads a small chunk of data to ensure the file isn't just a valid header with missing body.
    """
    try:
        # Using FD redirection to silence stubborn GDAL C-level warnings
        with suppress_fd_stderr(), rasterio.Env(CPL_LOG_ERRORS=False), rasterio.open(file_path) as src:
            # Basic header check - fast failure
            width = src.width
            height = src.height
            _ = src.count

            # Read the last pixel/block to verify file is not truncated
            # (Reading the whole file is too slow for 100s of files)
            # Window(col_off, row_off, width, height)
            if width > 0 and height > 0:
                _ = src.read(1, window=rasterio.windows.Window(width - 1, height - 1, 1, 1))
        return True
    except (RasterioIOError, Exception):
        return False


def verify_laz_integrity(file_path: Path) -> bool:
    """
    Checks if a LAZ file is valid using laspy.
    """
    try:
        with laspy.open(file_path) as fh:
            _ = fh.header
            # Try seeking to end?
        return True
    except Exception:
        return False


def verify_zarr_integrity(zarr_path: Path) -> bool:
    """
    Attempts to open the Zarr dataset. Returns True if valid, False if corrupt.
    """
    try:
        # Try to open with xarray
        # consolidated=True is used in creation, so we expect metadata headers
        with xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", consolidated=False) as da:
            # Just accessing .shape or .attrs is usually enough to verify header integrity
            _ = da.shape
            _ = da.attrs
        return True
    except Exception:
        try:
            # Try opening as Dataset if DataArray failed
            with xr.open_dataset(zarr_path, engine="zarr", chunks="auto", consolidated=False) as ds:
                _ = ds.dims
            return True
        except Exception:
            return False


def scan_provider(provider_name: str, clean: bool = False, lock_timeout: int = 3600) -> None:
    """
    Scans a specific provider directory for:
    1. Orphaned/Stale .lock files
    2. Corrupt Zarr directories
    3. Corrupt Source files (.bag, .tiff, .laz)
    4. Leftover .tmp files
    """
    p_dir = CACHE_ROOT / provider_name
    if not p_dir.exists():
        return

    logger.info(f"--- Scanning {provider_name} ---")

    # 1. Check for .tmp download files or locks
    for t in list(p_dir.glob(".tmp_*")) + list(p_dir.glob("*.lock")):
        # Locks > timeout are stale
        if t.suffix == ".lock":
            age = time.time() - t.stat().st_mtime
            if age > lock_timeout:
                if clean:
                    logger.info(f"Removing stale lock: {t.name}")
                    t.unlink()
                else:
                    logger.warning(f"Found stale lock: {t.name} ({age / 60:.1f} min old)")
        else:
            # Tmp files always suspect if older than timeout
            age = time.time() - t.stat().st_mtime
            if age > lock_timeout:
                if clean:
                    logger.info(f"Removing stuck temp file: {t.name}")
                    if t.is_dir():
                        shutil.rmtree(t)
                    else:
                        t.unlink()
                else:
                    logger.warning(f"Found stuck temp file: {t.name}")

    # 2. Check Source Files
    source_patterns = {
        "ncei_bag": ["*.bag"],
        "noaa_bluetopo": ["*.tiff", "*.tif"],
        "usgs_lidar": ["*.laz"],
        "ncei_cudem": ["*.tif"],
        "noaa_topobathy": ["*.tif", "*.tiff"],
        "usgs_3dep": ["*.tif", "*.tiff"],
    }

    patterns = source_patterns.get(provider_name, [])
    for pat in patterns:
        for f in p_dir.glob(pat):
            is_valid = verify_laz_integrity(f) if f.suffix == ".laz" else verify_raster_integrity(f)

            if not is_valid:
                if clean:
                    logger.error(f"CORRUPT Source File deleted: {f.name}")
                    f.unlink()
                else:
                    logger.error(f"CORRUPT Source File found: {f.name}")

    # 3. Check Zarr directories
    zarr_roots = [p_dir / "zarr", p_dir]

    for z_root in zarr_roots:
        if not z_root.exists():
            continue

        # Look for Zarrs
        zarrs = list(z_root.glob("*.zarr"))
        for z in zarrs:
            if not verify_zarr_integrity(z):
                if clean:
                    logger.error(f"CORRUPT Zarr deleted: {z.name}")
                    shutil.rmtree(z)
                else:
                    logger.error(f"CORRUPT Zarr found: {z.name}")


def main() -> None:
    """
    Main entry point for verifying cache integrity.
    """
    parser = argparse.ArgumentParser(description="Verify and clean TopoBathySim cache.")
    parser.add_argument("--clean", action="store_true", help="Delete corrupt files and stale locks.")
    parser.add_argument("--check", action="store_true", help="Only check integrity (Default).")
    parser.add_argument(
        "--lock-timeout",
        type=int,
        default=3600,
        help="Seconds before considering a lock stale (default: 3600).",
    )
    args = parser.parse_args()

    if args.clean:
        logger.info("Running in CLEAN mode. Deleting corrupt files...")
    else:
        logger.info("Running in CHECK mode. Use --clean to delete files.")

    for p in PROVIDERS:
        scan_provider(p, clean=args.clean, lock_timeout=args.lock_timeout)


if __name__ == "__main__":
    main()
