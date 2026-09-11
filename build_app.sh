#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"
export PYINSTALLER_CONFIG_DIR="$PROJECT_DIR/build/pyinstaller-cache"
export MPLCONFIGDIR="$PROJECT_DIR/build/matplotlib-cache"
export CLANG_MODULE_CACHE_PATH="$PROJECT_DIR/build/swift-module-cache"

if [ ! -x "$PYTHON" ]; then
    echo "Missing .venv. Follow the README setup steps first." >&2
    exit 1
fi

if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "Missing PyInstaller. Run: .venv/bin/python -m pip install -r requirements-build.txt" >&2
    exit 1
fi

if ! command -v swiftc >/dev/null 2>&1; then
    echo "Missing swiftc. Install Apple's Command Line Tools first." >&2
    exit 1
fi

# Build the native menu-bar icon, status popup, and red alert. It communicates
# with the detector through two local pipes and has no runtime dependency.
mkdir -p "$PROJECT_DIR/build/helper-dist" "$CLANG_MODULE_CACHE_PATH"
swiftc \
    -O \
    -swift-version 5 \
    -framework AppKit \
    "$PROJECT_DIR/status_helper.swift" \
    -o "$PROJECT_DIR/build/helper-dist/HairGuardStatus"
codesign --force --sign - "$PROJECT_DIR/build/helper-dist/HairGuardStatus"

"$PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$PROJECT_DIR/dist" \
    --workpath "$PROJECT_DIR/build/hairguard-work" \
    "$PROJECT_DIR/HairGuard.spec"

# Apply a free ad-hoc signature. It is not Apple notarization, but it makes the
# local bundle internally consistent on Apple Silicon Macs.
codesign --force --deep --sign - "$PROJECT_DIR/dist/HairGuard.app"

ARCHIVE="$PROJECT_DIR/dist/HairGuard.app.zip"
rm -f "$ARCHIVE"
ditto -c -k --keepParent "$PROJECT_DIR/dist/HairGuard.app" "$ARCHIVE"

echo "Built: $PROJECT_DIR/dist/HairGuard.app"
echo "Share: $ARCHIVE"
