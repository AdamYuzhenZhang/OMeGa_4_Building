#!/usr/bin/env python3
"""Reserved entry point for future OMeGa-4-Building extension runs.

This script intentionally exists before the extension logic returns. Baseline
runs should use ``run_omega_baseline.py``; once a research component is added,
extension configs with ``"kind": "extension"`` should be launched here.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main("extension"))

