import os
from pathlib import Path

def get_cache_root() -> Path:
    """Resolve cache root from environment or fallback to user cache."""
    return Path(os.getenv("TOPOBATHYSIM_CACHE_DIR", "~/.cache/topobathysim")).expanduser()

CACHE_ROOT = get_cache_root()
