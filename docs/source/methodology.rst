Methodology
===========

TopoBathySim employs a strict, multi-tiered "Priority Overwrite" strategy to create seamless topobathymetric Digital Elevation Models (DEMs).

Data Hierarchy
--------------

The system processes data layers in specific order, where higher-tier (lower index) sources overwrite lower-tier sources.

**Tier 0 (Absolute Priority): NOAA BAG (Survey Data)**
    *   **Source**: NOAA NCEI Bathymetric Attributed Grids (BAG).
    *   **Resolution**: Survey Native (approx. 0.5m - 4m).
    *   **Logic**: The system performs a spatial bounding-box search for high-resolution survey files (e.g., *H13385*). If found, these are merged and burned into the grid.
    *   **Role**: Definitive "Truth" for surveyed bathymetry.

**Tier 1 (High-Res Coastal/Land Fusion): Lidar & Topobathy**
    *   **Source**: USGS 3DEP (Airborne Lidar) and NOAA NGS (Topobathymetic Lidar).
    *   **Resolution**: 1m - 3m.
    *   **Logic**: **Conditional Upland Fusion**.
        *   The system aligns the Topobathymetic grid to the Airborne Lidar grid.
        *   **Upland Rule**: Where Airborne Lidar elevation > 1.0m (approx. MSL + 1m), the system prioritizes **Airborne Lidar**. This strictly masks out water noise (specular returns) from land scanners.
        *   **Intertidal/Water Rule**: Where Airborne Lidar <= 1.0m (or missing), the system prioritizes **Topobathy Lidar**, which captures shallow water bottoms that airborne sensors miss.
    *   **Role**: Essential for defining the precise land-water interface and shallow subtidal zones.

**Tier 2: NOAA BlueTopo™**
    *   **Source**: NOAA Office of Coast Survey.
    *   **Resolution**: ~6-8m.
    *   **Role**: Primary high-resolution bathymetry covering US coastal waters. Prioritized over CUDEM to prevent older models from obscuring modern consolidated bathymetry.

**Tier 3: NOAA CUDEM**
    *   **Source**: NCEI Continuously Updated Digital Elevation Model (1/9 Arc-Second).
    *   **Resolution**: ~3m.
    *   **Role**: Secondary gap filler for coastal zones where BlueTopo coverage is incomplete or where very high-resolution gap filling is needed.

**Tier 4: Global Context (GEBCO 2025)**
    *   **Source**: General Bathymetric Chart of the Oceans.
    *   **Resolution**: ~450m (Global).
    *   **Role**: Universal fallback. Ensures that no query returns "empty" space, providing a base layer for the entire planet.

Fusion Logic
------------

The ``FusionEngine`` (orchestrated by ``BathyManager``) follows a "Composition" strategy rather than simple blending.

1.  **Canvas Creation**:
    A "Base Canvas" is initialized for the requested tile extent (bounds) and target resolution (e.g., 512x512).

2.  **Layer Accumulation**:
    The system queries all providers (BAG, Lidar, CUDEM, BlueTopo, GEBCO) in parallel availability checks.

3.  **Sequential Composition**:
    Layers are applied to the canvas in reverse order of priority (lowest to highest), or using `combine_first` logic where higher priority layers fill gaps.

    *   **Step 1**: The Global (GEBCO) or Regional (BlueTopo) layer establishes the baseline.
    *   **Step 2**: The **Tier 1 Fused Lidar** layer is superimposed. This layer itself is a composite result of the "Conditional Upland Fusion" described above.
    *   **Step 3**: Any **Tier 0 (BAG)** data is burned in on top of all other layers. This ensures that if a ship physically surveyed an area, that measurement overrides any interpolation or older model.

4.  **CRS Unification**:
    All layers are dynamically reprojected to EPSG:4326 (WGS84) and aligned to the pixel grid of the target canvas using Bilinear resampling.

Corner Cases
------------

**Missing Lidar**
    If Tier 1 data is missing, the system falls back seamlessly to Tier 2 (CUDEM) or Tier 3 (BlueTopo) for that specific pixel area.

**Deep Water**
    In deep water zones where Lidar/Topobathy do not exit, the system relies on BAGs (if available) or BlueTopo/GEBCO.


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

Visualization & Debugging Tools
-----------------------------

TopoBathySim includes comprehensive visualization tools to validate fusion logic and data provenance.

**Source Map ("Truth" Mode)**
    Use the `style=source` parameter or select "Source Map (Debug)" in the viewer to enable provenance visualization.
    This mode replaces elevation colors with categorical identifiers to show exactly which dataset contributed to each pixel.

    *   **Red**: Tier 0 - NOAA BAG (Bathymetric Attributed Grids, Survey Data)
    *   **Orange**: Tier 1 - Fused High-Res Lidar (Airborne + Topobathy)
    *   **Green**: Raw Airborne Lidar (USGS 3DEP)
    *   **Lime**: Raw Topobathy Lidar (NOAA NGS)
    *   **Cyan**: Tier 2 - NOAA BlueTopo™
    *   **Blue**: Tier 3 - NOAA CUDEM
    *   **Brown**: Land DEM (USGS/NASADEM)
    *   **Gray**: Tier 4 - GEBCO 2025 (Global Context)

**Contour Overlays**
    Use the `style=contours` parameter or the "Show Contours" checkbox in the viewer.
    This renders dynamically generated vector-like contour lines (2m intervals) on top of the map.
    Useful for:

    *   Verifying identifying seams/steps between datasets.
    *   checking vertical datum alignment (NAVD88 vs MLLW).
    *   Visualizing slope continuity across fusion boundaries.
