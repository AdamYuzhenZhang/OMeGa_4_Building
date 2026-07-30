#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMEGA_BUILDING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$OMEGA_BUILDING_ROOT/../.." && pwd)}"
SOURCE_VENV="${SOURCE_VENV:-$PROJECT_ROOT/.venv}"
TARGET_VENV="${TARGET_VENV:-$PROJECT_ROOT/.venv_split_splat}"
SPLIT_SPLAT_ROOT="${SPLIT_SPLAT_ROOT:-$PROJECT_ROOT/third_party/Split_and_Splat}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_HOME="${CUDA_HOME:-/usr}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -x "$SOURCE_VENV/bin/python" ]]; then
  echo "Source project environment is missing: $SOURCE_VENV" >&2
  exit 1
fi
if [[ ! -d "$SPLIT_SPLAT_ROOT" ]]; then
  echo "Split&Splat source is missing: $SPLIT_SPLAT_ROOT" >&2
  exit 1
fi

if [[ ! -x "$TARGET_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TARGET_VENV"
fi

# The released Split&Splat propagation code uses pycolmap's 3.11 pose API.
# Install the repository's exact pin locally so the project environment linked
# below cannot silently supply an incompatible newer release.
"$TARGET_VENV/bin/pip" install --no-deps "pycolmap==3.11.1"

SOURCE_SITE="$("$SOURCE_VENV/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
TARGET_SITE="$("$TARGET_VENV/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$SOURCE_SITE" > "$TARGET_SITE/digitaltwin_project_venv.pth"

build_extension() {
  local source_dir="$1"
  env \
    CUDA_HOME="$CUDA_HOME" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    "$TARGET_VENV/bin/pip" install \
      -e "$source_dir" \
      --no-build-isolation \
      --no-deps
}

build_extension "$SPLIT_SPLAT_ROOT/point_projection"
build_extension "$SPLIT_SPLAT_ROOT/submodules/diff-gaussian-rasterization"
build_extension "$SPLIT_SPLAT_ROOT/submodules/simple-knn"
build_extension "$SPLIT_SPLAT_ROOT/submodules/fused-ssim"

CHECKPOINT="$SPLIT_SPLAT_ROOT/checkpoints/sam2.1_hiera_large.pt"
SHARED_CHECKPOINT="$PROJECT_ROOT/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  if [[ ! -f "$SHARED_CHECKPOINT" ]]; then
    echo "Shared SAM2 large checkpoint is missing: $SHARED_CHECKPOINT" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$CHECKPOINT")"
  if [[ -L "$CHECKPOINT" ]]; then
    rm "$CHECKPOINT"
  fi
  ln -s "$SHARED_CHECKPOINT" "$CHECKPOINT"
fi

SPLIT_PYTHONPATH="$PROJECT_ROOT/third_party/sam2"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT/utils"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT/point_projection"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT/submodules/diff-gaussian-rasterization"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT/submodules/simple-knn"
SPLIT_PYTHONPATH="$SPLIT_PYTHONPATH:$SPLIT_SPLAT_ROOT/submodules/fused-ssim"

PYTHONPATH="$SPLIT_PYTHONPATH" "$TARGET_VENV/bin/python" - <<'PY'
import torch
import pycolmap
import point_projection_cuda
import diff_gaussian_rasterization._C
import simple_knn._C
import fused_ssim_cuda

print("Split&Splat native imports: ready")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"pycolmap: {pycolmap.__version__}")
print(f"CUDA available now: {torch.cuda.is_available()}")
PY

echo "Split&Splat Python: $TARGET_VENV/bin/python"
