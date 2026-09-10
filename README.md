# HairGuard

HairGuard is a minimal macOS prototype that watches the built-in camera for
repeated hand-to-head scratching.

The current milestone:

1. Captures the Mac camera with OpenCV.
2. detects pose landmarks locally with MediaPipe.
3. estimates and draws a head/hair region.
4. rejects brief or stationary touches.
5. prints `SCRATCH` after repeated wrist movement near the head.
6. applies a cooldown to prevent repeated triggers.

The screen-blocking overlay is intentionally **not implemented yet**. It should
only be added after the detector has been tested and tuned with the live camera.

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

## Run

With the virtual environment active:

```bash
python main.py
```

Keep your head, shoulders, and hands visible. The preview displays:

- the relevant pose landmarks;
- a thin ellipse for the raw head estimate;
- the translucent, asymmetric region actually used for contact detection;
- the detector state;
- whether either hand is near the head;
- accumulated movement and direction reversals; and
- a visible scratch event indicator.

When a scratch is detected, the terminal prints:

```text
SCRATCH
```

Press `q` while the preview window is focused to quit. `Ctrl-C` in the terminal
also stops the program.

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
├── main.py                       Camera, pose inference, and preview
├── scratch_detector.py           Portable scratching state machine
├── pose_landmarker_lite.task     Local MediaPipe pose model
├── requirements.txt              Pinned Python dependencies
└── README.md                     This guide
```

## Current verification status

- Python dependencies install without conflicts.
- The pose model initializes and returns 33 landmarks on a person image.
- Head-region estimation works with real pose landmarks.
- Synthetic checks confirm that stationary and brief touches do not trigger.
- Repeated back-and-forth movement produces one event followed by cooldown.
- Live camera verification still requires macOS camera permission on the host.
