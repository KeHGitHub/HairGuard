# HairGuard

HairGuard is a minimal macOS prototype that watches the built-in camera for
repeated hand-to-head scratching.

HairGuard currently:

1. Captures the Mac camera with OpenCV.
2. detects pose landmarks locally with MediaPipe.
3. estimates and draws a head/hair region.
4. rejects brief or stationary touches.
5. prints `SCRATCH` after repeated wrist movement near the head.
6. applies a cooldown to prevent repeated triggers.
7. supports a debug test mode and a quiet background mode.
8. shows a full-screen `STOP` warning in background mode.
9. estimates pulse locally from subtle colour changes in two cheek regions.

The warning does not lock or freeze macOS. It remains visible until the detected
hand leaves the head region or the user presses any key.

## Requirements

- macOS with a built-in or default camera
- Python 3.10 (tested with Python 3.10.8)
- No cloud service or backend

The lightweight pose model is included as `pose_landmarker_lite.task`, so pose
inference runs locally and does not need a model download at runtime.

## Setup

Open Terminal and run:

```bash
cd /Users/keh/Desktop/HairGuard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Camera permission

The first run may prompt for camera access. Choose **Allow**.

If access was previously denied, open:

**System Settings → Privacy & Security → Camera**

Enable access for the application launching Python, normally Terminal, iTerm,
or Codex. Restart that application if macOS does not apply the change
immediately.

## Run in test mode

With the virtual environment active:

```bash
python main.py --mode test
```

Keep your head, shoulders, and hands visible. The preview displays:

- the relevant pose landmarks;
- a thin ellipse for the raw head estimate;
- the translucent, asymmetric region actually used for contact detection;
- the detector state;
- whether either hand is near the head;
- accumulated movement and direction reversals;
- the two cheek regions sampled for rPPG and the current pulse status; and
- a visible scratch event indicator.

When a scratch is detected, the terminal prints:

```text
SCRATCH
```

Press `q` while the preview window is focused to quit. `Ctrl-C` in the terminal
also stops the program.

## Run in background mode

```bash
python main.py --mode background
```

Background mode keeps camera inference running without an OpenCV preview. When
a scratch is detected, it prints `SCRATCH` and displays a full-screen `STOP`
warning. The overlay runs in a separate process, so the camera and detector
continue processing instead of freezing while the warning is visible.

- Move the detected hand outside the head region to close the warning.
- Press any key while the warning is focused to dismiss it.
- Press `Ctrl-C` in the launching terminal at any time to stop HairGuard.

The process still runs in the terminal; “background” means the camera preview is
hidden. HairGuard intentionally does not install a daemon or login item.

## Build a macOS app

The local beta can be packaged as a double-clickable Apple Silicon macOS app.
The app contains Python, MediaPipe, OpenCV, the pose model, and a small native
menu-bar and alert helper. Your friends do not need Python and do not receive
separate source files.

Install the build tool once:

```bash
.venv/bin/python -m pip install -r requirements-build.txt
```

Then build:

```bash
./build_app.sh
```

The build produces:

```text
dist/HairGuard.app       Double-clickable application
dist/HairGuard.app.zip   Archive to send to friends
```

Double-clicking `HairGuard.app` starts background mode. It runs without a
camera preview and displays the full-screen `STOP` warning when scratching is
detected. The source command `python main.py` still defaults to test mode for
development.

The running app appears as a waveform icon in the macOS menu bar, without a
Dock icon or camera window. Click it to see:

- the current pulse estimate;
- the scratch detector state (`IDLE`, `POSSIBLE_SCRATCH`, `SCRATCHING`, or
  `COOLDOWN`); and
- **Quit HairGuard**.

Quitting closes any active warning overlay and releases the camera cleanly.
The pulse row displays a filtered rolling average. It shows `acquiring`,
`stabilizing`, `low light`, or `low signal` when there is not enough reliable
camera information.

## Pulse estimation (rPPG)

HairGuard samples average RGB values from two small cheek regions and retains
only a rolling colour-signal window. It does not retain camera frames. A simple
chrominance calculation reduces shared lighting changes, and a frequency
analysis estimates the strongest pulse between 45 and 180 BPM. HairGuard then
keeps a short history of those estimates, removes isolated outliers, weights
the reliable estimates, and smoothly updates the displayed average. A brief
signal-quality dip holds the last stable result instead of making the value
jump or disappear immediately.

For a cleaner reading, face the camera, remain reasonably still, and use steady
front lighting. Motion, automatic camera exposure, shadows, makeup, and changing
screen brightness can disturb the estimate. This is an experimental wellness
signal and must not be used for diagnosis or medical decisions.

There is one primary rPPG setting in `rppg.py`:
`ANALYSIS_WINDOW_SECONDS`. A longer window produces a steadier but slower
reading. The default is 10 seconds.

This beta uses a free ad-hoc signature rather than a paid Apple Developer ID.
After unzipping it, another user may need to Control-click `HairGuard.app`,
choose **Open**, and confirm **Open** on the first launch. They must also allow
camera access when macOS asks. Build separate versions if Intel Mac support is
needed; this configuration targets Apple Silicon (`arm64`).

## Detector states

```text
IDLE → POSSIBLE_SCRATCH → SCRATCHING → COOLDOWN → IDLE
```

The detector uses the wrist, thumb, index, and pinky points to decide whether a
hand touches the estimated head region. It then follows the wrist as the more
stable motion signal. A trigger requires a minimum contact time, enough total
movement, and several direction changes. Camera and UI code live in `main.py`;
the reusable heuristic is isolated in `scratch_detector.py`.

## Tuning

The tuning constants are near the top of `scratch_detector.py`.

If detection is too sensitive:

- decrease `HEAD_HORIZONTAL_SCALE` to narrow the region horizontally;
- decrease `HEAD_UPPER_SCALE` to shorten the hair region above the face;
- decrease `HEAD_LOWER_SCALE` to reject more cheek/chin contact;
- increase `MIN_CONTACT_SECONDS`;
- increase `MIN_DIRECTION_REVERSALS`;
- increase `MIN_TOTAL_TRAVEL` or `MIN_AXIS_RANGE`; or
- increase `MIN_MOTION_STEP` to ignore more landmark jitter.

If real scratches are missed, adjust those values in the opposite direction.
`HEAD_REGION_HOLD_SECONDS` controls how long the last reliable head position is
used when looking down or when the hand briefly hides facial landmarks. If hand
landmarks disappear during contact, slightly lower `MIN_LANDMARK_VISIBILITY`.

Hand points turn red when they are inside the actual detection boundary. This
makes it clear which points can currently start or continue a scratch candidate.

Increase `COOLDOWN_SECONDS` if separate alerts occur too close together.

Change one value at a time while watching `MOVE` and `REVERSALS` in the preview.

## Files

```text
HairGuard/
├── main.py                       Camera, modes, preview, and STOP overlay
├── overlay.py                    Non-blocking full-screen warning helper
├── rppg.py                       Local camera-based pulse estimator
├── status_helper.swift           Native macOS status and alert helper
├── scratch_detector.py           Portable scratching state machine
├── tests/test_rppg.py            Synthetic pulse-estimation checks
├── pose_landmarker_lite.task     Local MediaPipe pose model
├── requirements.txt              Pinned Python dependencies
├── requirements-build.txt        App packaging dependency
├── HairGuard.spec                macOS application bundle configuration
├── build_app.sh                  Reproducible local app build
└── README.md                     This guide
```

## Current verification status

- Python dependencies install without conflicts.
- The pose model initializes and returns 33 landmarks on a person image.
- Head-region estimation works with real pose landmarks.
- Synthetic checks confirm that stationary and brief touches do not trigger.
- Repeated back-and-forth movement produces one event followed by cooldown.
- Live test-mode detection has been confirmed during tuning.
- The full-screen overlay has been verified to appear and close automatically.
