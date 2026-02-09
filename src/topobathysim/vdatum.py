import logging
from functools import lru_cache
from typing import Any

import requests  # type: ignore

logger = logging.getLogger(__name__)


class VDatumResolver:
    """
    Resolves vertical datum offsets between NAVD88 and Local Mean Sea Level (LMSL)
    using NOAA's VDatum API. Supports GEOID18 for Ellipsoid to NAVD88 conversions.
    """

    VDATUM_API = "https://vdatum.noaa.gov/vdatumweb/api/convert"

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_navd88_to_lmsl_offset(lat: float, lon: float) -> float:
        """
        Calculates the vertical shift to convert NAVD88 elevation to LMSL.
        LMSL = NAVD88 - Offset
        """
        # API requires source/target/region
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,  # Explicitly convert 0
            "s_hframe": "NAD83_2011",
            "s_vframe": "NAVD88",
            "t_hframe": "NAD83_2011",
            "t_vframe": "LMSL",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            response = requests.get(VDatumResolver.VDATUM_API, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "t_z" in data:
                return float(data["t_z"])
            return 0.0

        except Exception as e:
            logger.warning(f"Datum conversion failed for ({lat}, {lon}): {e}")
            return 0.0

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_mllw_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from MLLW to NAVD88.
        NAVD88 = MLLW + Offset
        (Actually VDatum output is Target Height, so if Input=0 MLLW, Output=Z NAVD88)
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "NAD83_2011",
            "s_vframe": "MLLW",
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            response = requests.get(VDatumResolver.VDATUM_API, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "t_z" in data:
                # If s_z=0, t_z is the height of MLLW zero in NAVD88 frame.
                return float(data["t_z"])
            return 0.0

        except Exception as e:
            logger.warning(f"Datum (MLLW->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            return 0.0

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_ellipsoid_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from NAD83(2011) Ellipsoid to NAVD88 using GEOID18.
        NAVD88 = Ellipsoid - GeoidHeight
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "NAD83_2011",
            "s_vframe": "NAD83_2011",  # Ellipsoid
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "geoid": "GEOID18",  # Explicitly request GEOID18
            "result_format": "json",
        }

        try:
            response = requests.get(VDatumResolver.VDATUM_API, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "t_z" in data:
                return float(data["t_z"])
            return 0.0

        except Exception as e:
            logger.warning(f"Datum (Ellipsoid->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            return 0.0

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_egm2008_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from EGM2008 to NAVD88.
        NAVD88 = EGM2008 + Offset (t_z)
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "WGS84_G1674",  # EGM2008 implies WGS84 usually
            "s_vframe": "EGM2008",
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            response = requests.get(VDatumResolver.VDATUM_API, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "t_z" in data:
                return float(data["t_z"])
            return 0.0

        except Exception as e:
            logger.warning(f"Datum (EGM2008->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            return 0.0
