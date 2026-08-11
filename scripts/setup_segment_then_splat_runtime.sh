#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yz2332/projects/digitalTwin}"
STS_ROOT="${STS_ROOT:-$PROJECT_ROOT/third_party/Segment-then-Splat}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
EXT_DIR="${EXT_DIR:-$PROJECT_ROOT/.native_extensions/segment_then_splat}"
BUILD_DIR="${EXT_DIR}.build"
SOURCE_DIR="${EXT_DIR}.sources"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python does not exist: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$STS_ROOT/train.py" ]]; then
  echo "Segment-then-Splat source does not exist: $STS_ROOT" >&2
  exit 1
fi

if ! "$PYTHON" -c "import loguru" 2>/dev/null; then
  "$PYTHON" -m pip install "loguru>=0.7,<1"
fi

git -C "$STS_ROOT" submodule update --init --recursive
rm -rf "$BUILD_DIR" "$SOURCE_DIR"
mkdir -p "$BUILD_DIR" "$SOURCE_DIR"
cp -a \
  "$STS_ROOT/submodules/simple-knn" \
  "$SOURCE_DIR/simple-knn"
cp -a \
  "$STS_ROOT/submodules/depth-diff-gaussian-rasterization" \
  "$SOURCE_DIR/depth-diff-gaussian-rasterization"

# CUDA 12.0 cannot emit sm_120 directly. PTX compiled for compute 9.0 is
# forward-JITed by the RTX 5090 driver, matching the other local 3DGS builds.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-8}"

"$PYTHON" -m pip install \
  --no-build-isolation \
  --no-deps \
  --target "$BUILD_DIR" \
  "$SOURCE_DIR/simple-knn"
"$PYTHON" -m pip install \
  --no-build-isolation \
  --no-deps \
  --target "$BUILD_DIR" \
  "$SOURCE_DIR/depth-diff-gaussian-rasterization"

PYTHONPATH="$BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c \
  "import inspect, diff_gaussian_rasterization as d, simple_knn; signature=str(inspect.signature(d.GaussianRasterizer.forward)); assert 'means2D_densify' in signature, signature; print(signature)"

rm -rf "$SOURCE_DIR"
rm -rf "$EXT_DIR"
mv "$BUILD_DIR" "$EXT_DIR"
echo "Segment-then-Splat native extensions: $EXT_DIR"
