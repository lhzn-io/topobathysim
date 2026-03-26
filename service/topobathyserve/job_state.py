"""
Persistent job state for hydration tasks.

Each job writes its state to a JSON file at:
    ~/.cache/topobathysim/hydration_jobs/{job_id}.json

Design constraints:
- Single writer (the hydration subprocess) per job file.
- Multiple readers (GET endpoint, WebSocket handler) may read concurrently.
- Writes are atomic: write to .tmp then os.replace() (POSIX-atomic on same filesystem).
- Readers never get a partial/corrupt file — they either see the old or new version.
- Dead subprocess detection: reader checks if the recorded PID is still alive.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from topobathysim.config import get_cache_root


def _ensure_jobs_dir() -> Path:
    jobs_dir = get_cache_root() / "hydration_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def job_path(job_id: str) -> Path:
    return _ensure_jobs_dir() / f"{job_id}.json"


def write_state(job_id: str, state: dict[str, Any]) -> None:
    """Atomically write job state to disk."""
    path = job_path(job_id)
    tmp = path.with_suffix(".json.tmp")
    data = json.dumps(state, default=str)
    tmp.write_text(data)
    os.replace(str(tmp), str(path))  # atomic on POSIX


def read_state(job_id: str) -> dict[str, Any] | None:
    """Read job state from disk, detecting dead subprocesses."""
    path = job_path(job_id)
    if not path.exists():
        return None
    try:
        state: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # If status is running/pending, check if the subprocess is still alive
    if state.get("status") in ("running", "pending"):
        pid = state.get("pid")
        if pid and not _pid_alive(pid):
            state["status"] = "failed"
            state["error"] = f"Worker process {pid} died (likely OOM)"
            # Update the file so subsequent reads don't re-check
            write_state(job_id, state)

    return state


def list_jobs(max_age_hours: float = 24) -> list[dict[str, Any]]:
    """List recent jobs, pruning expired ones."""
    _ensure_jobs_dir()
    now = datetime.now(timezone.utc)
    jobs = []
    for f in _ensure_jobs_dir().glob("*.json"):

        try:
            state = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Prune old completed/failed jobs
        submitted = state.get("submitted_at", "")
        try:
            submitted_dt = datetime.fromisoformat(submitted)
            if submitted_dt.tzinfo is None:
                submitted_dt = submitted_dt.replace(tzinfo=timezone.utc)
            age_hours = (now - submitted_dt).total_seconds() / 3600
            if age_hours > max_age_hours and state.get("status") in ("completed", "failed"):
                p.unlink(missing_ok=True)
                continue
        except (ValueError, TypeError):
            pass
        jobs.append(state)
    return jobs


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running. Safe, no signals sent."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive
        return True
