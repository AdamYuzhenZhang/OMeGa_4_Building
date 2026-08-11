#!/usr/bin/env python3
"""Run the complete released Segment then Splat paper pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_local.reconstruction.pipelines.segment_then_splat_official.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
