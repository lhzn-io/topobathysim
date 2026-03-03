# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "TopoBathySim"
copyright = "2025, WHOI Marine Predators Group"
author = "Daniel Fry (Ported from Camrin Braun)"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["_extra"]

autodoc_mock_imports = [
    "osgeo",
    "xarray",
    "rioxarray",
    "numpy",
    "scipy",
    "pyproj",
    "affine",
    "rasterio",
    "h5py",
    "zarr",
    "fsspec",
    "s3fs",
    "geopandas",
    "pdal",
    "pydap",
    "matplotlib",
    "planetary_computer",
    "pystac_client",
    "requests",
    "httpx",
    "filelock",
    "pydantic",
    "uvicorn",
    "fastapi",
    "dotenv",
    "questionary",
    "rich",
    "lazrs",
    "yaml",
]
