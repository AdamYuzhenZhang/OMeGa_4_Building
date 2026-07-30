#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
VIEWER_ROOT="${SUPERSPLAT_VIEWER_ROOT:-$PROJECT_ROOT/third_party/supersplat-viewer}"
NODE_VERSION="${NODE_VERSION:-v22.23.1}"
TOOLCHAIN_ROOT="${NODE_TOOLCHAIN_ROOT:-$PROJECT_ROOT/.toolchains}"
NODE_ROOT="$TOOLCHAIN_ROOT/node-$NODE_VERSION-linux-x64"

if [[ ! -f "$VIEWER_ROOT/package.json" ]]; then
  echo "SuperSplat viewer repo not found: $VIEWER_ROOT" >&2
  exit 1
fi

if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  NODE_BIN="$(command -v node)"
  NPM_BIN="$(command -v npm)"
else
  mkdir -p "$TOOLCHAIN_ROOT"
  if [[ ! -x "$NODE_ROOT/bin/node" ]]; then
    ARCHIVE="$TOOLCHAIN_ROOT/node-$NODE_VERSION-linux-x64.tar.xz"
    echo "[viewer] Downloading Node.js $NODE_VERSION"
    curl -fL \
      "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" \
      -o "$ARCHIVE"
    tar -xJf "$ARCHIVE" -C "$TOOLCHAIN_ROOT"
    rm -f "$ARCHIVE"
  fi
  NODE_BIN="$NODE_ROOT/bin/node"
  NPM_BIN="$NODE_ROOT/bin/npm"
  export PATH="$NODE_ROOT/bin:$PATH"
fi

echo "[viewer] Node: $("$NODE_BIN" --version)"
VIEWER_INDEX="$VIEWER_ROOT/src/index.ts"
VIEWER_TYPES="$VIEWER_ROOT/src/types.ts"
if grep -Fq "export { loadGsplat, main };" "$VIEWER_INDEX" && \
   grep -Fq "entityName?: string;" "$VIEWER_TYPES"; then
  echo "[viewer] Multipart loader patch already applied"
elif grep -Fq "export { main };" "$VIEWER_INDEX" && \
     grep -Fq "const entity = new Entity('gsplat');" "$VIEWER_INDEX"; then
  echo "[viewer] Applying multipart loader patch"
  perl -0pi -e "s/new Entity\('gsplat'\)/new Entity(config.entityName ?? 'gsplat')/; s/export \{ main \};/export { loadGsplat, main };/" "$VIEWER_INDEX"
  perl -0pi -e 's/contentUrl\?: string;\n/contentUrl?: string;\n    entityName?: string;\n/' "$VIEWER_TYPES"
else
  echo "SuperSplat source is incompatible with the multipart loader adapter" >&2
  exit 1
fi

echo "[viewer] Installing locked SuperSplat dependencies"
(
  cd "$VIEWER_ROOT"
  "$NPM_BIN" ci
  echo "[viewer] Building static viewer"
  "$NPM_BIN" run build
)

test -f "$VIEWER_ROOT/public/index.html"
echo "[viewer] Ready: $VIEWER_ROOT/public/index.html"
