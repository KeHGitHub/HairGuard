from pathlib import Path
from importlib.util import find_spec


project_root = Path(SPECPATH)
status_helper = project_root / "build" / "helper-dist" / "HairGuardStatus"
mediapipe_c_spec = find_spec("mediapipe.tasks.c")

if mediapipe_c_spec is None or mediapipe_c_spec.origin is None:
    raise SystemExit("Could not locate mediapipe.tasks.c in the build environment.")

mediapipe_c_library = Path(mediapipe_c_spec.origin).parent / "libmediapipe.dylib"

if not status_helper.exists():
    raise SystemExit(
        "Missing build/helper-dist/HairGuardStatus. Run ./build_app.sh instead."
    )

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[
        (str(status_helper), "."),
        (str(mediapipe_c_library), "mediapipe/tasks/c"),
    ],
    datas=[(str(project_root / "pose_landmarker_lite.task"), ".")],
    hiddenimports=["mediapipe.tasks.c"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HairGuard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HairGuard",
)
app = BUNDLE(
    coll,
    name="HairGuard.app",
    icon=None,
    bundle_identifier="com.hairguard.prototype",
    info_plist={
        "CFBundleDisplayName": "HairGuard",
        "LSUIElement": True,
        "NSCameraUsageDescription": (
            "HairGuard uses the camera locally to detect hand-to-head scratching."
        ),
        "NSHighResolutionCapable": True,
    },
)
