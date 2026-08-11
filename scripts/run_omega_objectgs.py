#!/usr/bin/env python3
"""Thin launcher for joint ObjectGS reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from omega_local.reconstruction.backends.objectgs.cli import main

    return main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
