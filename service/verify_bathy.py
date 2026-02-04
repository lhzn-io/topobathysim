import io

import numpy as np
import requests  # type: ignore


def verify_bathy() -> None:
    # Long Island Sound location (should have bathymetry)
    url = "http://localhost:9595/tiles/13/2409/3086?format=npy&ept_url=https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_NY_LongIsland_Z18_2014_LAS_2015/ept.json"

    print(f"Fetching {url}...")
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"FAILED: Status {r.status_code}")
            print(r.text)
            return

        buf = io.BytesIO(r.content)
        arr = np.load(buf)

        print("SUCCESS: Received NPY tile.")
        print(f"Shape: {arr.shape}")
        print(f"Min: {np.nanmin(arr)}")
        print(f"Max: {np.nanmax(arr)}")
        print(f"Mean: {np.nanmean(arr)}")

        # Check for underwater values
        underwater = arr[arr < 0]
        if len(underwater) > 0:
            print(f"Underwater pixels: {len(underwater)} ({len(underwater) / arr.size * 100:.1f}%)")
            print(f"Deepest point: {np.min(underwater)}")
        else:
            print("WARNING: All values are >= 0! Bathymetry missing?")

        ambiguous = arr[(arr >= 0.0) & (arr < 1.0)]
        print(f"Near-Zero (0-1m) pixels: {len(ambiguous)} (Potential artifacts)")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    verify_bathy()
