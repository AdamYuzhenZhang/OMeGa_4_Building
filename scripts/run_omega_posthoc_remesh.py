#!/usr/bin/env python3
"""Run post-hoc structure-aware remeshing on an OMeGa result directory."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.remesh.posthoc import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

