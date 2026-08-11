#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMEGA_BUILDING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$OMEGA_BUILDING_ROOT/../.." && pwd)}"
SOURCE_VENV="${SOURCE_VENV:-$PROJECT_ROOT/.venv}"
TARGET_VENV="${TARGET_VENV:-$PROJECT_ROOT/.venv_objectgs}"
OBJECTGS_ROOT="${OBJECTGS_ROOT:-$PROJECT_ROOT/third_party/ObjectGS}"
GSPLAT_OBJECT_ROOT="${GSPLAT_OBJECT_ROOT:-$PROJECT_ROOT/third_party/gsplat-object}"
GLM_ROOT="${GLM_ROOT:-$PROJECT_ROOT/third_party/glm}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_HOME="${CUDA_HOME:-/usr}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
MAX_JOBS="${MAX_JOBS:-4}"

for required in \
  "$SOURCE_VENV/bin/python" \
  "$OBJECTGS_ROOT/train.py" \
  "$GSPLAT_OBJECT_ROOT/setup.py" \
  "$GLM_ROOT/glm/glm.hpp"; do
  if [[ ! -e "$required" ]]; then
    echo "Required ObjectGS dependency is missing: $required" >&2
    exit 1
  fi
done

if [[ ! -x "$TARGET_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TARGET_VENV"
fi

SOURCE_SITE="$("$SOURCE_VENV/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
TARGET_SITE="$("$TARGET_VENV/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$SOURCE_SITE" > "$TARGET_SITE/digitaltwin_project_venv.pth"

env \
  CPATH="$GLM_ROOT${CPATH:+:$CPATH}" \
  CUDA_HOME="$CUDA_HOME" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  MAX_JOBS="$MAX_JOBS" \
  "$TARGET_VENV/bin/pip" install \
    -e "$GSPLAT_OBJECT_ROOT" \
    --no-build-isolation \
    --no-deps

COMPAT_ROOT="$OMEGA_BUILDING_ROOT/omega_local/reconstruction/backends/objectgs/compat"
cd "$OBJECTGS_ROOT"
PYTHONPATH="$GSPLAT_OBJECT_ROOT:$COMPAT_ROOT:$OBJECTGS_ROOT" \
  "$TARGET_VENV/bin/python" - <<'PY'
from pathlib import Path

import gsplat
import torch
from torch_scatter import scatter_max

gsplat_path = Path(gsplat.__file__).resolve()
if "gsplat-object" not in str(gsplat_path):
    raise RuntimeError(f"Wrong gsplat package loaded: {gsplat_path}")
source = torch.tensor([[1.0], [3.0]], device="cuda")
index = torch.tensor([[0], [0]], device="cuda")
reduced, _ = scatter_max(source, index, dim=0)
assert reduced.item() == 3.0

print("ObjectGS runtime: ready")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA capability: {torch.cuda.get_device_capability()}")
print(f"Customized gsplat: {gsplat_path}")
PY

echo "ObjectGS Python: $TARGET_VENV/bin/python"
