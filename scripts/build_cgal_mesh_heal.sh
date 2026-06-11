#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/tools/cgal_mesh_heal.cpp"
OUT="$REPO_ROOT/tools/cgal_mesh_heal"

g++ -std=c++17 -O3 "$SRC" -o "$OUT" -lgmp -lmpfr
echo "$OUT"
