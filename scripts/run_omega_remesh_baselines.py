#!/usr/bin/env python3
"""Run mesh-only remeshing baselines for comparison with view-informed remesh."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.remesh.baselines import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
