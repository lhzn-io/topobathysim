from enum import Enum
from typing import ClassVar

import numpy as np
import xarray as xr


class QualityClass(str, Enum):
    DIRECT = "Direct Measurement"
    INDIRECT = "Indirect/Predicted"
    UNKNOWN = "Unknown/Interpolated"


class TIDClassifier:
    """
    Classifies GEBCO Type Identifier (TID) codes into semantic quality classes.
    Based on GEBCO 2025 TID conventions.
    """

    # Direct measurements: Single beam, Multibeam, Seismic, Lidar, Optical, etc.
    DIRECT_TIDS: ClassVar[set[int]] = {10, 11, 12, 13, 14, 15, 16, 17}

    # Indirect methods: Satellite/Gravity Predicted, Interpolated, Contours, etc.
    INDIRECT_TIDS: ClassVar[set[int]] = {40, 41, 42, 43, 44, 45, 46, 47}

    # Unknown, Pre-generated, or Steering points
    UNKNOWN_TIDS: ClassVar[set[int]] = {0, 70, 71, 72}

    @classmethod
    def classify(cls, tid: int) -> QualityClass:
        if tid in cls.DIRECT_TIDS:
            return QualityClass.DIRECT
        elif tid in cls.INDIRECT_TIDS:
            return QualityClass.INDIRECT
        else:
            return QualityClass.UNKNOWN

    @classmethod
    def get_color(cls, quality: QualityClass) -> str:
        if quality == QualityClass.DIRECT:
            return "green"
        elif quality == QualityClass.INDIRECT:
            return "orange"
        else:
            return "grey"


def source_report(tid_data: xr.DataArray | np.ndarray) -> dict[str, float]:
    """
    Generates a percentage report of data sources for the given TID array.

    Args:
        tid_data: xarray.DataArray or numpy array containing GEBCO TID values.

    Returns:
        Dictionary mapping QualityClass names to percentage coverage (0-100).
    """
    data = tid_data.values.flatten() if isinstance(tid_data, xr.DataArray) else np.asarray(tid_data).flatten()

    total_pixels = len(data)
    if total_pixels == 0:
        return {
            QualityClass.DIRECT.value: 0.0,
            QualityClass.INDIRECT.value: 0.0,
            QualityClass.UNKNOWN.value: 0.0,
        }

    # Vectorized counting
    # We can use numpy's isin or logical checks, but a simple loop or mask is clear for now
    # Optimization: Use searchsorted or histograms if performance is critical for massive arrays

    # Direct
    direct_mask = np.isin(data, list(TIDClassifier.DIRECT_TIDS))
    direct_count = np.count_nonzero(direct_mask)

    # Indirect
    indirect_mask = np.isin(data, list(TIDClassifier.INDIRECT_TIDS))
    indirect_count = np.count_nonzero(indirect_mask)

    # Unknown/Rest
    # Alternatively explicit check:
    # unknown_mask = np.isin(data, list(TIDClassifier.UNKNOWN_TIDS))
    # unknown_count = np.count_nonzero(unknown_mask)

    # Handle any unclassified TIDs as unknown for safety
    final_unknown = total_pixels - direct_count - indirect_count

    return {
        QualityClass.DIRECT.value: round((direct_count / total_pixels) * 100, 2),
        QualityClass.INDIRECT.value: round((indirect_count / total_pixels) * 100, 2),
        QualityClass.UNKNOWN.value: round((final_unknown / total_pixels) * 100, 2),
    }
