import numpy as np


def verify_manager_logic():
    west = -70.1
    south = 40.0
    east = -70.0
    north = 40.1
    target_shape = (10, 10)

    height, width = target_shape
    lon_res = (east - west) / width
    lat_res = (north - south) / height

    print(f"Res: {lon_res}, {lat_res}")

    xs = np.linspace(west + lon_res / 2, east - lon_res / 2, num=width)
    ys = np.linspace(north - lat_res / 2, south + lat_res / 2, num=height)

    print(f"XS Start: {xs[0]:.20f}")
    print(f"XS End: {xs[-1]:.20f}")
    print(f"YS Start: {ys[0]:.20f}")
    print(f"YS End: {ys[-1]:.20f}")

    # Expected
    # West + Res/2 = -70.1 + 0.005 = -70.095
    print(f"Expected XS Start: {-70.095:.20f}")


if __name__ == "__main__":
    verify_manager_logic()
