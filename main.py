"""HairGuard milestone one: local webcam pose tracking and scratch detection."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from scratch_detector import DetectorResult, Landmark, ScratchDetector


def bundled_path(filename: str) -> Path:
    """Locate a resource in source runs or inside a PyInstaller bundle."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / filename
    return Path(__file__).resolve().with_name(filename)


# Official MediaPipe lite model, stored locally so runtime inference is offline:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
MODEL_PATH = bundled_path("pose_landmarker_lite.task")
OVERLAY_PATH = bundled_path("overlay.py")
OVERLAY_HELPER_PATH = bundled_path("HairGuardOverlay")
WINDOW_NAME = "HairGuard - press q to quit"
OVERLAY_EXIT_GRACE_SECONDS = 0.25

LANDMARK_INDEX = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
}

POINT_COLORS = {
    "nose": (0, 220, 255),
    "left_ear": (0, 220, 255),
    "right_ear": (0, 220, 255),
    "left_shoulder": (255, 180, 0),
    "right_shoulder": (255, 180, 0),
    "left_wrist": (60, 255, 60),
    "right_wrist": (60, 255, 60),
    "left_pinky": (60, 255, 60),
    "right_pinky": (60, 255, 60),
    "left_index": (60, 255, 60),
    "right_index": (60, 255, 60),
    "left_thumb": (60, 255, 60),
    "right_thumb": (60, 255, 60),
}


def open_camera() -> cv2.VideoCapture:
    """Open the built-in/default camera, preferring macOS AVFoundation."""
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open camera 0. Grant camera access to Terminal in "
            "System Settings > Privacy & Security > Camera, then try again."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def select_landmarks(pose: Optional[Iterable[object]]) -> Dict[str, Landmark]:
    if pose is None:
        return {}
    points = list(pose)
    selected: Dict[str, Landmark] = {}
    for name, index in LANDMARK_INDEX.items():
        point = points[index]
        visibility = float(getattr(point, "visibility", 1.0) or 0.0)
        presence = float(getattr(point, "presence", 1.0) or 0.0)
        selected[name] = Landmark(
            x=float(getattr(point, "x")),
            y=float(getattr(point, "y")),
            visibility=min(visibility, presence),
        )
    return selected


def pixel(point: Landmark, width: int, height: int) -> Tuple[int, int]:
    return int(point.x * width), int(point.y * height)


def draw_debug(
    frame: np.ndarray,
    landmarks: Mapping[str, Landmark],
    result: DetectorResult,
) -> None:
    height, width = frame.shape[:2]

    if result.head_region is not None:
        region = result.head_region
        center = (int(region.center_x * width), int(region.center_y * height))
        base_axes = (
            max(1, int(region.radius_x * width)),
            max(1, int(region.radius_y * height)),
        )
        detection_radius_x = max(1, int(region.detection_radius_x * width))
        upper_axes = (
            detection_radius_x,
            max(1, int(region.detection_radius_y_upper * height)),
        )
        lower_axes = (
            detection_radius_x,
            max(1, int(region.detection_radius_y_lower * height)),
        )

        # Join two half-ellipses to draw the exact asymmetric contains() boundary.
        lower_arc = cv2.ellipse2Poly(center, lower_axes, 0, 0, 180, 4)
        upper_arc = cv2.ellipse2Poly(center, upper_axes, 0, 180, 360, 4)
        detection_contour = np.vstack((lower_arc, upper_arc))
        tint = frame.copy()
        cv2.fillPoly(tint, [detection_contour], (255, 80, 200))
        cv2.addWeighted(tint, 0.13, frame, 0.87, 0, frame)
        cv2.polylines(frame, [detection_contour], True, (255, 80, 200), 2)

        # The thin inner ellipse shows the raw head estimate before weighting.
        cv2.ellipse(frame, center, base_axes, 0, 0, 360, (170, 90, 150), 1)
        cv2.putText(
            frame,
            "ACTUAL SCRATCH DETECTION REGION",
            (
                max(8, center[0] - detection_radius_x),
                max(18, center[1] - upper_axes[1] - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 80, 200),
            1,
            cv2.LINE_AA,
        )

    for first, second in (
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_pinky"),
        ("left_wrist", "left_thumb"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_pinky"),
        ("right_wrist", "right_thumb"),
    ):
        if first in landmarks and second in landmarks:
            cv2.line(
                frame,
                pixel(landmarks[first], width, height),
                pixel(landmarks[second], width, height),
                (170, 170, 170),
                2,
            )

    for name, point in landmarks.items():
        if point.visibility < 0.5:
            continue
        location = pixel(point, width, height)
        hand_point = name.endswith(("wrist", "pinky", "index", "thumb"))
        inside = bool(
            hand_point
            and result.head_region is not None
            and result.head_region.contains(point)
        )
        color = (0, 0, 255) if inside else POINT_COLORS[name]
        cv2.circle(frame, location, 9 if inside else 7, color, -1)
        label = name.replace("left_", "L ").replace("right_", "R ")
        cv2.putText(
            frame,
            label,
            (location[0] + 8, location[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (10, 10), (390, 142), (20, 20, 20), -1)
    lines = (
        f"STATE: {result.state}",
        f"LEFT NEAR HEAD:  {'YES' if result.left_near_head else 'no'}",
        f"RIGHT NEAR HEAD: {'YES' if result.right_near_head else 'no'}",
        f"ACTIVE: {result.active_hand or '-'}   MOVE: {result.movement:.3f}",
        f"REVERSALS: {result.reversals}",
    )
    for row, text in enumerate(lines):
        color = (50, 80, 255) if "YES" in text else (235, 235, 235)
        cv2.putText(
            frame,
            text,
            (22, 35 + row * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            1,
            cv2.LINE_AA,
        )

    if result.state == "SCRATCHING":
        text_size = cv2.getTextSize("SCRATCH", cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
        origin = ((width - text_size[0]) // 2, height // 2)
        cv2.putText(
            frame,
            "SCRATCH",
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (20, 20, 255),
            4,
            cv2.LINE_AA,
        )


def launch_stop_overlay() -> subprocess.Popen:
    """Start the overlay separately so its event loop cannot pause detection."""
    if getattr(sys, "frozen", False):
        if not OVERLAY_HELPER_PATH.exists():
            raise RuntimeError(f"Missing overlay helper: {OVERLAY_HELPER_PATH}")
        command = [str(OVERLAY_HELPER_PATH)]
    else:
        if not OVERLAY_PATH.exists():
            raise RuntimeError(f"Missing overlay helper: {OVERLAY_PATH}")
        command = [sys.executable, str(OVERLAY_PATH)]
    try:
        return subprocess.Popen(command)
    except OSError as error:
        raise RuntimeError(f"Could not launch the STOP overlay: {error}") from error


def close_overlay(process: subprocess.Popen) -> None:
    """Close a running overlay helper and reap its process."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local hand-to-hair scratch detector")
    parser.add_argument(
        "--mode",
        choices=("test", "background"),
        default="test",
        help="test shows the debug preview; background hides it and shows STOP alerts",
    )
    return parser.parse_args()


def run(mode: str = "test") -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing pose model: {MODEL_PATH}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = ScratchDetector()
    camera = open_camera()
    started = time.monotonic()
    previous_timestamp_ms = -1
    overlay_process: Optional[subprocess.Popen] = None
    overlay_hand: Optional[str] = None
    hand_left_region_at: Optional[float] = None

    if mode == "test":
        print("HairGuard test mode. Press q in the preview window to quit.", flush=True)
    else:
        print("HairGuard background mode. Press Ctrl-C in this terminal to quit.", flush=True)
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                if overlay_process is not None:
                    overlay_status = overlay_process.poll()
                    if overlay_status is not None:
                        if overlay_status != 0:
                            print(
                                f"HairGuard warning: overlay exited with status {overlay_status}",
                                file=sys.stderr,
                            )
                        overlay_process = None
                        overlay_hand = None
                        hand_left_region_at = None

                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")

                # A mirrored preview behaves like a normal selfie camera.
                frame = cv2.flip(frame, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp = time.monotonic()
                timestamp_ms = max(previous_timestamp_ms + 1, int((timestamp - started) * 1000))
                previous_timestamp_ms = timestamp_ms
                pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                landmarks = select_landmarks(pose)
                result = detector.update(landmarks, timestamp)

                if overlay_process is not None and overlay_hand is not None:
                    active_hand_near = (
                        result.left_near_head
                        if overlay_hand == "left"
                        else result.right_near_head
                    )
                    if active_hand_near:
                        hand_left_region_at = None
                    elif hand_left_region_at is None:
                        hand_left_region_at = timestamp
                    elif timestamp - hand_left_region_at >= OVERLAY_EXIT_GRACE_SECONDS:
                        close_overlay(overlay_process)
                        overlay_process = None
                        overlay_hand = None
                        hand_left_region_at = None

                if result.scratch:
                    print("SCRATCH", flush=True)
                    if mode == "background" and overlay_process is None:
                        overlay_process = launch_stop_overlay()
                        overlay_hand = result.active_hand
                        hand_left_region_at = None

                if mode == "test":
                    draw_debug(frame, landmarks, result)
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if overlay_process is not None and overlay_process.poll() is None:
            close_overlay(overlay_process)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(parse_args().mode)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(f"HairGuard error: {error}", file=sys.stderr)
        raise SystemExit(1)
