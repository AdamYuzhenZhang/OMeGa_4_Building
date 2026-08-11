#!/usr/bin/env python3
"""Run Gaussian Grouping directly on persistent-region masks."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_local.reconstruction.backends.static_semantic_3dgs.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], method="gaussian_grouping"))
