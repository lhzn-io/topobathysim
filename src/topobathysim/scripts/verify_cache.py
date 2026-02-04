import math
import time
from pathlib import Path

import requests  # type: ignore

SERVICE_URL = "http://localhost:9595"
CACHE_DIR = Path("~/.cache/topobathysim/lidar").expanduser()


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def main() -> None:
    # NYC (Battery Park) - Likely to have Lidar
    lat = 40.7032
    lon = -74.0170
    zoom = 15

    x, y = deg2num(lat, lon, zoom)
    print(f"Requesting Tile: Z={zoom} X={x} Y={y} (Lat={lat}, Lon={lon})")

    # URL
    url = f"{SERVICE_URL}/tiles/{zoom}/{x}/{y}.png"

    print(f"Checking Cache Dirs: {CACHE_DIR} and ../land")
    land_cache = CACHE_DIR.parent / "land"

    initial_lidar = len(list(CACHE_DIR.glob("*.laz"))) if CACHE_DIR.exists() else 0
    initial_land = len(list(land_cache.glob("*.tif"))) if land_cache.exists() else 0

    try:
        t0 = time.time()
        print("Sending request (timeout=120s)...")
        resp = requests.get(url, timeout=120)
        dur = time.time() - t0
        print(f"Response Code: {resp.status_code}")
        print(f"Duration: {dur:.2f}s")

        if resp.status_code == 200:
            final_lidar = len(list(CACHE_DIR.glob("*.laz")))
            final_land = len(list(land_cache.glob("*.tif")))

            print(f"Lidar Files: {initial_lidar} -> {final_lidar}")
            print(f"Land Files:  {initial_land} -> {final_land}")

            if final_lidar > initial_lidar:
                print("SUCCESS: New LIDAR file cached!")
            elif final_land > initial_land:
                print("SUCCESS: New LAND file cached (Lidar likely missing/fallback).")
                print(
                    "Note: Lidar missing is expected in some regions, Land caching proves offline capability."
                )
            else:
                print("WARN: No new files cached. Maybe already cached?")
        else:
            print(f"FAIL: Request failed: {resp.text}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
