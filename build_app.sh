#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"
export PYINSTALLER_CONFIG_DIR="$PROJECT_DIR/build/pyinstaller-cache"

if [ ! -x "$PYTHON" ]; then
    echo "Missing .venv. Follow the README setup steps first." >&2
    exit 1
fi

if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "Missing PyInstaller. Run: .venv/bin/python -m pip install -r requirements-build.txt" >&2
    exit 1
fi

# Build the small Tk overlay independently so showing STOP does not load a
# second copy of MediaPipe, OpenCV, or the pose model.
"$PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    --onefile \
    --name HairGuardOverlay \
    --distpath "$PROJECT_DIR/build/helper-dist" \
    --workpath "$PROJECT_DIR/build/helper-work" \
    --specpath "$PROJECT_DIR/build/helper-spec" \
    "$PROJECT_DIR/overlay.py"

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
