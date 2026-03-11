import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env variables
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from .providers.registry import registry  # noqa: E402
from .quality import TIDClassifier, source_report  # noqa: E402

# Set up NullHandler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "TIDClassifier",
    "registry",
    "source_report",
]
