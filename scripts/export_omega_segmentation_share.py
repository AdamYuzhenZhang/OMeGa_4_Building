#!/usr/bin/env python3
"""Export an interactive persistent-region propagation as a share package."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from omega_local.segmentation.interactive.share_export import main as export_main

    return export_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
