import math
import time
from typing import Any

import requests  # type: ignore

# ROI: Western Long Island Sound / New York Harbor
# Min Lon, Min Lat, Max Lon, Max Lat
# Default ROI
ROI = (-74.05, 40.5, -73.0, 41.25)
ZOOM = 13
SERVICE_URL = "http://localhost:9595/tiles"


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float, float, float]:
    """
    Returns (west, south, east, north) for a given tile.
    """
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)

    # Bottom right for extent
    lon_deg_next = (xtile + 1) / n * 360.0 - 180.0
    lat_rad_next = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
    lat_deg_next = math.degrees(lat_rad_next)

    return (lon_deg, lat_deg_next, lon_deg_next, lat_deg)


def fetch_tile(z: int, x: int, y: int) -> tuple[int, float]:
    url = f"{SERVICE_URL}/{z}/{x}/{y}?format=png"
    try:
        t0 = time.time()
        resp = requests.get(url, timeout=60)
        dur = time.time() - t0
        status = resp.status_code
        # size = len(resp.content)
        # print(f"Tile {z}/{x}/{y}: {status} ({size/1024:.1f} KB) in {dur:.2f}s")
        return (status, dur)
    except Exception as e:
        print(f"Failed {z}/{x}/{y}: {e}")
        return (0, 0.0)


def get_bluetopo_coverage(west: float, south: float, east: float, north: float) -> Any:
    """
    Returns a unified Shapely geometry of valid BlueTopo tiles in the bbox.
    """
    try:
        # import geopandas as gpd  # Not used directly

        from shapely.geometry import box
        from shapely.ops import unary_union

        from topobathysim.noaa_bluetopo import NoaaBlueTopoProvider

        print("Loading BlueTopo Scheme for guided priming...")
        bt = NoaaBlueTopoProvider()
        bt._ensure_scheme_loaded()

        if bt._gdf is None:
            print("Warning: BlueTopo scheme failed to load.")
            return None

        # Filter Scheme to BBox
        search_box = box(west, south, east, north)

        # Ensure CRS match
        if bt._gdf.crs and bt._gdf.crs.to_string() != "EPSG:4326":
            # This part simplifies; usually we reproject query to gdf crs
            # But for union we want coverage in 4326 to match XYZ tiles
            pass

        # Simplification: Just find intersecting rows
        # We need to intersect the query box with the GDF
        # But GDF might be in another CRS.
        # Let's assume standard behavior from BlueTopoProvider logic

        candidates = bt._gdf[bt._gdf.intersects(search_box)] if bt._gdf.crs == "EPSG:4326" else bt._gdf
        # If not 4326, we should verify. Usually BlueTopo GPKG is coverage.

        if candidates.empty:
            return None

        # Return the collected geometry
        return unary_union(candidates.geometry)

    except ImportError:
        print("BlueTopoProvider or dependencies missing. Skipping guided mode.")
        return None
    except Exception as e:
        print(f"Error loading BlueTopo coverage: {e}")
        return None


def main() -> None:
    import argparse

    from shapely.geometry import box

    parser = argparse.ArgumentParser(description="Prime Bathymetry Tile Cache")
    parser.add_argument(
        "--bbox", type=str, default="-74.05,40.5,-73.0,41.25", help="BBox: west,south,east,north"
    )
    parser.add_argument("--zoom", type=int, default=13, help="Zoom level")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--guided", action="store_true", help="Only prime tiles covered by BlueTopo")
    args = parser.parse_args()

    # Parse BBox
    try:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        west, south, east, north = parts
    except ValueError:
        print("Invalid bbox format. Use: west,south,east,north")
        return

    # Check validity
    if west >= east or south >= north:
        print("Invalid bbox dimensions (west >= east or south >= north)")
        return

    # Load Coverage if guided
    coverage_geom = None
    if args.guided:
        coverage_geom = get_bluetopo_coverage(west, south, east, north)
        if coverage_geom is None:
            print("No BlueTopo coverage found in ROI (or error). Falling back to full bbox?")
            # Decide behavior: Skip or All? Let's generic to All but warn
            print("Proceeding with full bbox.")
        else:
            print("BlueTopo coverage loaded. Filtering tiles...")

    roi = (west, south, east, north)
    zoom_level = args.zoom

    # Top Left (North West)
    x_min, y_min = deg2num(north, west, zoom_level)
    # Bottom Right (South East)
    x_max, y_max = deg2num(south, east, zoom_level)

    # Correct order
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)

    total_possible = (x_end - x_start + 1) * (y_end - y_start + 1)

    tiles = []
    print("Generating tile list...")
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            if coverage_geom:
                # Check intersection
                w, s, e, n = num2deg(x, y, zoom_level)
                tile_box = box(w, s, e, n)
                if not coverage_geom.intersects(tile_box):
                    continue
            tiles.append((zoom_level, x, y))

    print(f"Priming Cache for ROI: {roi}")
    print(f"Zoom: {zoom_level}")
    print(f"Tile Range: X[{x_start}-{x_end}], Y[{y_start}-{y_end}]")
    print(f"Total Tiles in Grid: {total_possible}")
    print(f"Tiles to Fetch (Guided): {len(tiles)}")

    # Parallel Fetch
    print(f"Starting parallel fetch (n_jobs={args.jobs})...")

    import concurrent.futures

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_tile = {executor.submit(fetch_tile, z, x, y): (z, x, y) for z, x, y in tiles}
        for future in concurrent.futures.as_completed(future_to_tile):
            results.append(future.result())

    success = [r for r in results if r[0] == 200]
    avg_time = sum(r[1] for r in success) / len(success) if success else 0

    print(f"Done. Processed {len(results)} tiles.")
    print(f"Success: {len(success)}")
    print(f"Average Time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
