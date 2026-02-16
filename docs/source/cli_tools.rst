CLI Tools
=========

TopoBathySim includes several command-line utilities for managing data caches, verifying integrity, and running the tile server.

Cache Verification & Maintenance
--------------------------------

Since TopoBathySim aggressively caches gigabytes of survey data (BAG, Lidar, COGs), use the integrity tool to manage storage health. This script identifies and (optionally) removes corrupt downloads or stale lock files from crashed processes.

.. code-block:: bash

   # 1. Audit Cache (Dry Run)
   # Checks ZIPs, BAGs, LAZ files, and Zarr stores for corruption
   micromamba run -n topobathysim python src/topobathysim/scripts/verify_cache_integrity.py --check

   # 2. Repair Cache (Clean Mode)
   # Deletes corrupt source files and releases locks older than the timeout
   micromamba run -n topobathysim python src/topobathysim/scripts/verify_cache_integrity.py --clean

   # 3. Custom Timeout
   # Consider locks stale after 2 minutes (120s) instead of the default 1 hour
   micromamba run -n topobathysim python src/topobathysim/scripts/verify_cache_integrity.py --check --lock-timeout 120

Key Features:
^^^^^^^^^^^^^

- **Source Validation**: Uses native drivers (`rasterio`, `laspy`) to inspect file headers for standard geospatial formats (`.bag`, `.tiff`, `.laz`).
- **Zarr Integrity**: verifies the consistency of derived Zarr cache directories.
- **Stale Lock Cleanup**: Identifies `.lock` files abandoned by interrupted processes.
- **Benign Warning Suppression**: Automatically filters harmless C-level warnings (e.g., NCEI BAG metadata precision issues) to reduce noise.

Server Management
-----------------

To run the tile server for local development or production usage:

.. code-block:: bash

   # Start the server on port 9595 with 8 worker processes
   micromamba run -n topobathysim python service/run_server.py --host 0.0.0.0 --port 9595 --workers 8

Arguments:
^^^^^^^^^^

- ``--host``: Bind address (default: 127.0.0.1)
- ``--port``: Port to listen on (default: 9595)
- ``--workers``: Number of worker processes (default: 4)
- ``--debug``: Enable debug logging (default: 0)
