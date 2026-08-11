#!/usr/bin/env python3
"""Run the OMeGa adapter for released 3D Gaussian Flats."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_local.reconstruction.backends.gaussian_flats.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
