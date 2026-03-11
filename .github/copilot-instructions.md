# TopoBathySim Copilot Instructions

## 🧠 Project Architecture & Core Concepts

- **Mission**: High-fidelity topobathymetric fusion for digital twins/robotics, merging disparate sources into unified tiles.
- **Fusion Hierarchy (Strict Priority)**:
  1. **Tier 0 (Absolute)**: NOAA BAG (Raw Survey, high precision).
  2. **Tier 1 (Fusion)**: Airborne Lidar (>0m) + Greenwave Topobathy (Intertidal).
  3. **Tier 2 (Coastal)**: NOAA CUDEM (Gap-fill).
  4. **Tier 3 (Regional)**: NOAA BlueTopo.
  5. **Tier 4 (Global Fallback)**: GEBCO 2025.
- **Microservice**: `service/topobathyserve` is a FastAPI app serving tiles (`/tiles/{z}/{x}/{y}`) and raw NumPy buffers.
- **Data Flow**: `Runtime` (in `src/topobathysim/runtime.py`) orchestrates data fetching and fusion based on YAML policies. It uses `Policy` objects to define fusion steps.

## 🛠️ Tech Stack & Conventions

- **Language**: Python 3.10+ (Strict type hinting required).
- **Core Libs**: `xarray`/`rioxarray` (raster manip), `numpy` (math), `pdal` (point clouds), `bmi-topography` (standard fetching).
- **Environment**: **Micromamba** is preferred due to complex binary deps (`gdal`, `pdal`).
  - *Warning*: `bmi-topography` version is pinned (~0.9.0); handle upgrades with caution.
  - **CRITICAL**: Always use `micromamba run -n topobathysim <command>` or `micromamba activate topobathysim` before running any Python code (including pytest/pip).
- **Linting/Formatting**: `ruff` for linting/formatting, `mypy` for static analysis (configured in `pyproject.toml`).
  - **CRITICAL**: Avoid `E501` line-length violations. Keep lines under 110 characters (or as configured in `pyproject.toml`). Break long strings, complex imports, and nested calls into multiple lines.
- **Path Handling**: Use `pathlib.Path` over `os.path`.

### 1. Code Generation & Refactoring

- **Type Safety**: ALWAYS add type hints. Run `mypy` after significant changes.
- **Path Handling**: Use `pathlib.Path` over `os.path`.
- **Fusion Logic**: When modifying fusion logic (`fusion.py`), respect the **Source ID** constants defined in `manager.py`.
- **Refactoring**: If moving files, use `git mv`. Note recent refactor of `service` module -> `topobathyserve`.

### 2. Service & API

- **FastAPI**: The service is located in `service/topobathyserve/main.py`.
- **Endpoints**:
  - `GET /tiles/{z}/{x}/{y}.png` (Visual debug)
  - `GET /tiles/{z}/{x}/{y}?format=npy` (Simulation data)
  - `GET /source_map/{z}/{x}/{y}.png` (Provenance debugging)

### 3. Testing & Verification

- **Framework**: `pytest`.
- **Markers**: Use `@pytest.mark.integration` for tests hitting external APIs (USGS/NOAA).
- **Running Tests**: `pytest` or `hatch run test`.
- **Debugging**:
  - Use the built-in viewer: `python service/run_server.py`, then open `http://localhost:9595/viewer`.
  - Check `~/.cache/topobathysim` for cached COGs/LAZ files if data looks weird.

### 4. Common Tasks & Commands

- **Start Server**: `python service/run_server.py`
- **Lint/Format**: `pre-commit run --all-files` or `ruff check .`
- **Clean Cache**: `rm -rf ~/.cache/topobathysim` (Useful when debugging bad downloads).

## ⚠️ Gotchas & Edge Cases

- **Vertical Datums**: Data comes in varying datums. We strive for **LMSL** or **NAVD88**. Transformations happen via VDatum logic (`vdatum.py`).
- **Offline Mode**: logic handles `offline_mode=True`. Ensure new providers respect this flag (fail gracefully or use cache).
- **Geometry**: Bathymetry is negative down? No, usually typical DEMs are **positive up** (Elevation). standard is `z > 0` = Land, `z < 0` = Water (in most contexts, but check `fusion.py` for specific normalization).
