"""Make the service package importable for the test session.

Modules under service/topobathyserve use absolute imports of the form
`from topobathyserve.models import ...`, which assume service/ is on the
path (true under run_server.py and in the container, where WORKDIR is
service/). Tests import `service.topobathyserve.main` from the repo root,
so both roots are needed. Adding service/ here lets `python -m pytest tests`
run from a fresh shell without a PYTHONPATH export.
"""

import sys
from pathlib import Path

_SERVICE = str(Path(__file__).resolve().parent.parent / "service")
if _SERVICE not in sys.path:
    sys.path.insert(0, _SERVICE)
