#!/usr/bin/env python3
"""Run the shared 0..2999 OMeGa warmup used before A/B branches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.runner import main  # noqa: E402


def _extract_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mesh-preview-frame",
        type=int,
        default=None,
        help="Parser-frame index used for warmup mesh preview plots. Default comes from the JSON config/trainer.",
    )
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    wrapper_args, remaining = _extract_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(REPO_ROOT / "configs" / "local" / "warmup_2999.json")])
    if wrapper_args.mesh_preview_frame is not None:
        sys.argv.extend(["--set", f"args.mesh_preview_frame={int(wrapper_args.mesh_preview_frame)}"])
    raise SystemExit(main("warmup"))
