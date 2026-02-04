Methodology
===========

TopoBathySim employs a multi-tiered data handling strategy to create seamless topobathymetric Digital Elevation Models (DEMs).

Data Hierarchy
--------------

The system prioritizes high-resolution local data over lower-resolution global basemaps.

Bathymetry (Water)
~~~~~~~~~~~~~~~~~~

1.  **Tier 1: NOAA BlueTopo™**
    *   **Resolution**: Variable (High), approx. 4-8m.
    *   **Coverage**: US Coastal Waters.
    *   **Priority**: **Highest**. If a BlueTopo tile covers the requested region, it is used exclusively for the bathymetric component.

2.  **Tier 2: GEBCO 2025**
    *   **Resolution**: 15 arc-seconds (approx. 450m).
    *   **Coverage**: Global.
    *   **Priority**: **Fallback**. Used when BlueTopo is unavailable (e.g., international waters, open ocean).

Topography (Land & Coast)
~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Tier 1: Coastal Lidar (USGS 3DEP / NOAA)**
    *   **Resolution**: 1-3m (Entwine Point Tiles / COPC).
    *   **Coverage**: Narrow coastal ribbons (US).
    *   **Priority**: **Anchor**. This is the highest fidelity dataset and serves as the reference for fusion.

2.  **Tier 2: Land Topography (USGS 3DEP / NASADEM)**
    *   **Resolution**: 10m (US), 30m (Global).
    *   **Coverage**: Continental.
    *   **Priority**: **Fill**. Used to extend coverage inland behind the narrow Lidar strip.

Fusion Logic
------------

The ``FusionEngine`` merges these disparate sources using two primary strategies defined by the nature of the data overlap.

Logistic Overlap Fusion
~~~~~~~~~~~~~~~~~~~~~~~

When datasets overlap significantly (e.g., Topobathymetric Lidar extending into the water column), we use a logistic weighting function to transition between sources based on elevation.

**Formula:**

.. math::

   w = \frac{1}{1 + e^{-k(z - z_{center})}}

Where:

*   :math:`z` is the elevation from the priority dataset (Lidar).
*   :math:`z_{center}` is the transition midpoint (default 0.0 for NAVD88/LMSL).
*   :math:`k` is the steepness of the transition.

**Result:**

*   :math:`z \gg 0`: Weight approaches 1.0 (Lidar Dominant).
*   :math:`z \ll 0`: Weight approaches 0.0 (Bathy Dominant).
*   :math:`z \approx 0`: Smooth blend.

Seamline Blending (USGS-Inspired)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When datasets do not overlap, or have a gap ("The Void"), we use a spatial blending approach inspired by the **USGS Adaptive Topobathymetric Fusion** methodology.

**Reference:**

    Danielson, J.J., and Poppenga, S.K., 2023, Adaptive Topobathymetric Fusion Software: U.S. Geological Survey software release, https://doi.org/10.5066/P9POPM96.

**Algorithm:**

The USGS approach mitigates seamless artifacts by projecting a "Weight Surface" outwards from the priority dataset (Data A / Lidar) into the secondary dataset (Data B / Bathy).

1.  **Identify Seam**: We generate a binary mask where Lidar data is valid.
2.  **Distance Transform**: We calculate the Euclidean Distance (:math:`d`) from the nearest valid Lidar pixel into the void using ``scipy.ndimage.distance_transform_edt``.
3.  **Proxy Generation**: We use the indices from the distance transform to identify the "Nearest Lidar Pixel" for every point in the transition zone. This creates a "Lidar Proxy" layer that extrapolates land values outwards (Nearest Neighbor).
4.  **Weight Calculation**:

    .. math::

       W_{lidar} = 1.0 - \frac{d}{d_{max}}

    Where :math:`d_{max}` is the designated blending distance (e.g., 20 meters). Weights are clipped to :math:`[0, 1]`.

5.  **Fusion**:

    .. math::

       Z_{final} = (W_{lidar} \cdot Z_{proxy}) + ((1 - W_{lidar}) \cdot Z_{bathy})

**Visual Result:**

Instead of a vertical cliff at the exact pixel where Lidar ends, the land "fades out" linearly over 20 meters, blending smoothly into the underlying bathymetry. This eliminates step-interaction artifacts in simulation physics engines.

.. note::
   The Lidar dataset is treated as the "truth" in overlapping regions due to its superior precision.

Corner Cases & Fallbacks
------------------------

Global / Missing Lidar
~~~~~~~~~~~~~~~~~~~~~~
In regions where Coastal Lidar is unavailable (e.g., International waters, Remote islands):
*   **Behavior**: The system renders the Bathymetry source (GEBCO) directly.
*   **Result**: The coastline will be defined by the resolution of the bathymetry source (450m for GEBCO). Land will be filled by NASADEM (if enabled) or remain flat/zero.

Missing BlueTopo
~~~~~~~~~~~~~~~~
*   **Behavior**: Fallback to GEBCO 2025.
*   **Result**: Lower resolution bathymetry seamlessly accepted by the fusion engine.

Offline Mode
~~~~~~~~~~~~
*   **Behavior**: If ``OFFLINE_MODE=1``, the system skips all network requests.
*   **Result**:
    *   If a tile is **Cached**: It renders normally.
    *   If **All Missing**: The service may return an empty/flat grid or an error depending on the extent of missing data.

Data Access Technologies
------------------------

Lidar via COPC & Hybrid Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TopoBathySim leverages **Cloud Optimized Point Clouds (COPC)** to provide instant access to massive Lidar datasets while building a robust offline cache.

*   **Technology**: COPC (an LAZ 1.4 extension) allows ``pdal`` to stream specific byte-ranges from a remote file, fetching only the points needed for the requested tile.
*   **Format**: ``.copc.laz`` files hosted by USGS/NOAA on AWS/Azure.
*   **Strategy**: "Stream First, Cache Background".
    1.  **Request**: User requests a tile.
    2.  **Streaming**: The system immediately streams the necessary points from the remote COPC URL to generate the tile.
    3.  **Background Cache**: Simultaneously, a background thread downloads the *entire* source file to the local cache.
    4.  **Future Access**: Subsequent requests for the same area use the local file instantly (Offline capable).

.. code-block:: python

    # This happens automatically within LidarProvider
    provider = LidarProvider()
    da = provider.fetch_lidar_from_stac(
        bounds=(-73.74, 40.87, -73.72, 40.89),
        force_cache=True  # Spawns background download
    )
