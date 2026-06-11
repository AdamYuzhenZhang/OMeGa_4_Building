#!/usr/bin/env python3
"""Run the OMeGa-4-Building in-training remesh extension."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.runner import main  # noqa: E402


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(REPO_ROOT / "configs" / "local" / "remesh_no_app_from_warmup_10000.json")])
    raise SystemExit(main("extension"))
