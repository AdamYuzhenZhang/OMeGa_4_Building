#!/usr/bin/env python3
"""Run the 6500-step OMeGa-4-Building depth-regularized extension config."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.runner import main  # noqa: E402


if __name__ == "__main__":
    from_warmup = False
    if "--from-warmup" in sys.argv:
        from_warmup = True
        sys.argv.remove("--from-warmup")
    if "--config" not in sys.argv:
        config_name = "depth_regularized_from_warmup_6500.json" if from_warmup else "depth_regularized_6500.json"
        sys.argv.extend(["--config", str(REPO_ROOT / "configs" / "local" / config_name)])
    raise SystemExit(main("extension"))
