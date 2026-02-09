import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before imports to silence bmi_topography warnings
# Find .env in project root (assuming we are in src/topobathysim)
# Path(__file__) = src/topobathysim/__init__.py
# .parent = src/topobathysim
# .parent.parent = src
# .parent.parent.parent = topobathysim (root)
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
    os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

from .gebco_2025 import GEBCO2025Provider
from .manager import BathyManager
from .ncei_cudem import CUDEMProvider
from .noaa_bluetopo import NoaaBlueTopoProvider
from .quality import TIDClassifier, source_report

# Set up NullHandler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BathyManager",
    "CUDEMProvider",
    "GEBCO2025Provider",
    "NoaaBlueTopoProvider",
    "TIDClassifier",
    "source_report",
]
