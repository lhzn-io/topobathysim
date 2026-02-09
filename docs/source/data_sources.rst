Data Sources & Acknowledgements
=============================

**TopoBathySim** relies on a federation of open-access geospatial datasets. We gratefully acknowledge the following agencies and specific surveys used for development, validation, and simulation.

The datasets are listed below in inverse order of priority (from global context to high-resolution survey truth).

Global Coverage (Tier 4)
------------------------

General Bathymetric Chart of the Oceans (GEBCO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **GEBCO_2025 Grid**

    *   **Description**: A continuous terrain model for ocean and land, providing global context at 15 arc-second intervals.
    *   **Citation**: GEBCO Compilation Group (2025) GEBCO 2025 Grid (doi:10.5285/1c44ce9e-0a53-431f-9780-3843464522c5).
    *   **Access**: `GEBCO Grid Download <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_

Continental / Land (Tier 3.5)
-----------------------------

U.S. Geological Survey (USGS) - 3DEP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **3D Elevation Program (3DEP)**

    *   **Description**: The source for high-quality topographic data of the conterminous United States, Hawaii, and U.S. territories.
    *   **Citation**: U.S. Geological Survey, 2017, 3D Elevation Program (3DEP): U.S. Geological Survey Fact Sheet 2017–3081.
    *   **Access**: `3DEP Homepage <https://www.usgs.gov/core-science-systems/ngp/3dep>`_

Regional Bathymetry (Tier 3)
----------------------------

NOAA NCEI - CUDEM
~~~~~~~~~~~~~~~~~

*   **Continuously Updated Digital Elevation Model (CUDEM)**

    *   **Description**: Ninth Arc-Second Resolution Bathymetric-Topographic Tiles. Used as a critical gap-filler for coastal inundation zones.
    *   **Citation**: Cooperative Institute for Research in Environmental Sciences (CIRES) at the University of Colorado, Boulder. 2014. Continuously Updated Digital Elevation Model (CUDEM) - Ninth Arc-Second Resolution Bathymetric-Topographic Tiles. NOAA National Centers for Environmental Information.
    *   **Access**: `Dataset Homepage <https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:999919/html>`_ | `Bulk Access <https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/index.html>`_

Regional High-Resolution (Tier 2)
---------------------------------

NOAA Office of Coast Survey - BlueTopo™
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **BlueTopo™**

    *   **Description**: A compilation of the best available public bathymetric data of U.S. waters, merged into a simplified product.
    *   **Citation**: NOAA Office of Coast Survey. BlueTopo™.
    *   **Access**: `BlueTopo Portal <https://nauticalcharts.noaa.gov/data/bluetopo.html>`_

Local High-Resolution Lidar (Tier 1)
------------------------------------

NOAA National Geodetic Survey (NGS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Topobathymetric Lidar**

    *   **Description**: High-resolution nearshore bathymetry captured via green-wavelength Lidar.
    *   **Citation**: National Geodetic Survey, 2026: 2023 NOAA NGS Lidar DEM. NOAA National Centers for Environmental Information.
    *   **Example (Long Island Sound)**: `Methodology Report (PDF) <https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/10273/supplemental/NOAA_NY_Long_Island_Topobathymetric_Lidar_Imagery_Report_Final.pdf>`_
    *   **Access**: `NOAA Digital Coast <https://coast.noaa.gov/digitalcoast/>`_

Survey "Truth" Data (Tier 0)
----------------------------

NOAA NCEI - Bathymetric Attributed Grid (BAG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Bathymetric Attributed Grids (BAG)**

    *   **Description**: The definitive format for multi-beam sonar surveys. When found, these overwrite all other sources.
    *   **Citation**: NOAA National Centers for Environmental Information. "Bathymetric Attributed Grid (BAG) File Format."
    *   **Access**: `NCEI Bathymetry Data Viewer <https://www.ncei.noaa.gov/maps/bathymetry/>`_
