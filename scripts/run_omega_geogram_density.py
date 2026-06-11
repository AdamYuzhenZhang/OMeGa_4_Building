#!/usr/bin/env python3
"""Export OMeGa Phase 3 weights as a Geogram/Vorpalite density sidecar."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.remesh.geogram_density import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
