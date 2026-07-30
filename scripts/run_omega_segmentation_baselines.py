#!/usr/bin/env python3
"""Run segmentation baselines for OMeGa outputs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _extract_baseline(argv: list[str]) -> tuple[str, list[str]]:
    baseline = "view_proposals"
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--baseline":
            if index + 1 >= len(argv):
                raise SystemExit("--baseline requires a value")
            baseline = argv[index + 1].strip().lower()
            index += 2
            continue
        if arg.startswith("--baseline="):
            baseline = arg.split("=", 1)[1].strip().lower()
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return baseline, cleaned


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    baseline, cleaned = _extract_baseline(raw)
    if baseline in {"sam2object", "sam2_object"}:
        from omega_local.segmentation.sam2object_baseline import main as sam2object_main

        return sam2object_main(cleaned)
    if baseline in {"view_proposals", "view_proposal", "proposals", "sam2_proposals", "sam2_auto_proposals"}:
        from omega_local.segmentation.view_proposals_baseline import main as view_proposals_main

        return view_proposals_main(cleaned)
    if baseline in {"sai3d", "sai_3d"}:
        from omega_local.segmentation.sai3d_baseline import main as sai3d_main

        return sai3d_main(cleaned)
    raise SystemExit(f"Unknown segmentation baseline: {baseline}")


if __name__ == "__main__":
    raise SystemExit(main())
