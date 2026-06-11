#!/usr/bin/env python3
"""Export OMeGa view-normal tangent-flow constraints for Instant Meshes."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.remesh.instant_field import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

