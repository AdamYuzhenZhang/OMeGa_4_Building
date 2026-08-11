#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMEGA_BUILDING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$OMEGA_BUILDING_ROOT/../.." && pwd)}"
SOURCE_VENV="${SOURCE_VENV:-$PROJECT_ROOT/.venv}"
TARGET_VENV="${TARGET_VENV:-$PROJECT_ROOT/.venv_gaussian_flats}"
GAUSSIAN_FLATS_ROOT="${GAUSSIAN_FLATS_ROOT:-$PROJECT_ROOT/third_party/3dgs-flats}"
TORCH_KDTREE_ROOT="${TORCH_KDTREE_ROOT:-$PROJECT_ROOT/third_party/torch_kdtree}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_HOME="${CUDA_HOME:-/usr}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -x "$SOURCE_VENV/bin/python" ]]; then
  echo "Project environment is missing: $SOURCE_VENV" >&2
  exit 1
fi
if [[ ! -f "$GAUSSIAN_FLATS_ROOT/train_planar.py" ]]; then
  echo "3D Gaussian Flats source is missing: $GAUSSIAN_FLATS_ROOT" >&2
  exit 1
fi
if [[ ! -f "$TORCH_KDTREE_ROOT/setup.py" ]]; then
  echo "torch-kdtree source is missing: $TORCH_KDTREE_ROOT" >&2
  exit 1
fi

if [[ ! -x "$TARGET_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TARGET_VENV"
fi

# Reuse the project's RTX 5090-compatible PyTorch/CUDA stack while keeping all
# Gaussian Flats native extensions isolated in their own environment.
SOURCE_SITE="$("$SOURCE_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
TARGET_SITE="$("$TARGET_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$SOURCE_SITE" > "$TARGET_SITE/digitaltwin_project_venv.pth"

build_extension() {
  local source_dir="$1"
  env \
    CUDA_HOME="$CUDA_HOME" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
    "$TARGET_VENV/bin/pip" install \
      -e "$source_dir" \
      --no-build-isolation \
      --no-deps
}

build_extension "$TORCH_KDTREE_ROOT"
build_extension "$GAUSSIAN_FLATS_ROOT/submodules/diff-gaussian-rasterization"
build_extension "$GAUSSIAN_FLATS_ROOT/submodules/simple-knn"

PYTHONPATH="$GAUSSIAN_FLATS_ROOT:$GAUSSIAN_FLATS_ROOT/submodules/diff-gaussian-rasterization:$GAUSSIAN_FLATS_ROOT/submodules/simple-knn:$TORCH_KDTREE_ROOT:$OMEGA_BUILDING_ROOT" \
  "$TARGET_VENV/bin/python" - <<'PY'
import torch
import torch_kdtree
import diff_gaussian_rasterization._C
import simple_knn._C

print("3D Gaussian Flats native imports: ready")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA available now: {torch.cuda.is_available()}")
print(f"torch-kdtree: {torch_kdtree.__file__}")
PY

echo "3D Gaussian Flats Python: $TARGET_VENV/bin/python"
