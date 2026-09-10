"""Portable hand-to-head scratching heuristic.

This module knows nothing about cameras, OpenCV, MediaPipe, or macOS. Coordinates
are normalized to the image: x=0..1 from left to right and y=0..1 top to bottom.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Deque, Mapping, Optional, Tuple


# ---- Manual tuning knobs -------------------------------------------------
# The contact boundary is deliberately asymmetric: hair extends farther above
# the face, while the lower edge stays tight to avoid ordinary chin touches.
HEAD_HORIZONTAL_SCALE = 2.0
HEAD_UPPER_SCALE = 1.65
HEAD_LOWER_SCALE = 0.85
# Reuse the last reliable head region during brief face/ear occlusion.
HEAD_REGION_HOLD_SECONDS = 0.75
# Increase these three values to reduce false positives; decrease them if a
# real scratch is not detected.
MIN_CONTACT_SECONDS = 1
MIN_DIRECTION_REVERSALS = 0
MIN_TOTAL_TRAVEL = 0.055
# Smaller supporting movements are treated as pose-estimation jitter.
MIN_MOTION_STEP = 0.0025
MIN_AXIS_RANGE = 0.012
MOTION_WINDOW_SECONDS = 1.20
# Time from a trigger until another scratch can trigger.
COOLDOWN_SECONDS = 2.50

MIN_LANDMARK_VISIBILITY = 0.50
CONTACT_LOSS_GRACE_SECONDS = 0.12
SCRATCHING_STATE_SECONDS = 0.30


class State(str, Enum):
    IDLE = "IDLE"
    POSSIBLE_SCRATCH = "POSSIBLE_SCRATCH"
    SCRATCHING = "SCRATCHING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class HeadRegion:
    """Asymmetric head-contact region in normalized image coordinates."""

    center_x: float
    center_y: float
    radius_x: float
    radius_y: float

    @property
    def detection_radius_x(self) -> float:
        return self.radius_x * HEAD_HORIZONTAL_SCALE

    @property
    def detection_radius_y_upper(self) -> float:
        return self.radius_y * HEAD_UPPER_SCALE

    @property
    def detection_radius_y_lower(self) -> float:
        return self.radius_y * HEAD_LOWER_SCALE

    def score(self, point: Landmark) -> float:
        """Asymmetric elliptical distance; a score <= 1 is detectable contact."""
        radius_y = (
            self.detection_radius_y_upper
            if point.y < self.center_y
            else self.detection_radius_y_lower
        )
        dx = (point.x - self.center_x) / self.detection_radius_x
        dy = (point.y - self.center_y) / radius_y
        return dx * dx + dy * dy

    def contains(self, point: Landmark) -> bool:
        return self.score(point) <= 1.0


@dataclass(frozen=True)
class DetectorResult:
    scratch: bool
    state: str
    head_region: Optional[HeadRegion]
    left_near_head: bool
    right_near_head: bool
    active_hand: Optional[str]
    movement: float
    reversals: int


def _visible(point: Optional[Landmark]) -> bool:
    return point is not None and point.visibility >= MIN_LANDMARK_VISIBILITY


def estimate_head_region(landmarks: Mapping[str, Landmark]) -> Optional[HeadRegion]:
    """Estimate a deliberately generous head/hair ellipse from pose landmarks."""
    nose_point = landmarks.get("nose")
    nose = nose_point if _visible(nose_point) else None

    ears = [
        point
        for name in ("left_ear", "right_ear")
        if _visible(point := landmarks.get(name))
    ]
    shoulders = [
        point
        for name in ("left_shoulder", "right_shoulder")
        if _visible(point := landmarks.get(name))
    ]
    if nose is None and len(ears) < 2:
        return None

    center_x = (
        sum(point.x for point in ears) / len(ears)
        if len(ears) == 2
        else nose.x  # type: ignore[union-attr]
    )
    face_y = nose.y if nose is not None else sum(point.y for point in ears) / len(ears)
    ear_y = sum(point.y for point in ears) / len(ears) if ears else face_y

    if len(ears) == 2:
        ear_span = hypot(ears[0].x - ears[1].x, ears[0].y - ears[1].y)
    elif len(ears) == 1 and nose is not None:
        ear_span = 2.0 * hypot(ears[0].x - nose.x, ears[0].y - nose.y)
    else:
        ear_span = 0.0

    if len(shoulders) == 2:
        shoulder_span = hypot(
            shoulders[0].x - shoulders[1].x,
            shoulders[0].y - shoulders[1].y,
        )
    else:
        shoulder_span = 0.0

    shoulder_y = sum(point.y for point in shoulders) / len(shoulders) if shoulders else face_y
    face_to_shoulders = max(0.0, shoulder_y - face_y)

    radius_x = max(0.060, ear_span * 0.85, shoulder_span * 0.22)
    radius_y = max(0.080, radius_x * 1.25, face_to_shoulders * 0.55)
    center_y = (face_y + ear_y) * 0.5 - radius_y * 0.5
    return HeadRegion(center_x, center_y, radius_x, radius_y)


class ScratchDetector:

    def __init__(self) -> None:
        self.state = State.IDLE
        self._active_hand: Optional[str] = None
        self._contact_started = 0.0
        self._last_near = 0.0
        self._scratch_started = 0.0
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._last_head_region: Optional[HeadRegion] = None
        self._last_head_region_time = 0.0

    def update(self, landmarks: Mapping[str, Landmark], timestamp: float) -> DetectorResult:
        fresh_region = estimate_head_region(landmarks)
        if fresh_region is not None:
            self._last_head_region = fresh_region
            self._last_head_region_time = timestamp
        region = fresh_region
        if (
            region is None
            and self._last_head_region is not None
            and timestamp - self._last_head_region_time <= HEAD_REGION_HOLD_SECONDS
        ):
            region = self._last_head_region
        hands = {
            side: [
                point
                for part in ("wrist", "pinky", "index", "thumb")
                if _visible(point := landmarks.get(f"{side}_{part}"))
            ]
            for side in ("left", "right")
        }
        near = {
            side: bool(region and any(region.contains(point) for point in points))
            for side, points in hands.items()
        }
        motion_points = {
            side: self._motion_point(landmarks, side, points)
            for side, points in hands.items()
        }

        if self.state == State.SCRATCHING:
            if timestamp - self._scratch_started >= SCRATCHING_STATE_SECONDS:
                self.state = State.COOLDOWN
            return self._result(False, region, near)

        if self.state == State.COOLDOWN:
            if timestamp - self._scratch_started >= COOLDOWN_SECONDS:
                self._reset()
            return self._result(False, region, near)

        if self.state == State.IDLE:
            candidates = [side for side in ("left", "right") if near[side]]
            if candidates and region is not None:
                # If both hands are near, follow whichever is closer to the ellipse center.
                side = min(
                    candidates,
                    key=lambda name: min(region.score(point) for point in hands[name]),
                )
                point = motion_points[side]
                assert point is not None
                self.state = State.POSSIBLE_SCRATCH
                self._active_hand = side
                self._contact_started = timestamp
                self._last_near = timestamp
                self._history.clear()
                self._history.append((timestamp, point.x, point.y))
            return self._result(False, region, near)

        # POSSIBLE_SCRATCH
        side = self._active_hand
        point = motion_points.get(side) if side else None
        if side and near[side] and point is not None:
            self._last_near = timestamp
            self._history.append((timestamp, point.x, point.y))
            self._trim_history(timestamp)
        elif timestamp - self._last_near > CONTACT_LOSS_GRACE_SECONDS:
            self._reset()
            return self._result(False, region, near)

        movement, reversals, axis_range = self._motion_metrics()
        long_enough = timestamp - self._contact_started >= MIN_CONTACT_SECONDS
        oscillating = (
            movement >= MIN_TOTAL_TRAVEL
            and reversals >= MIN_DIRECTION_REVERSALS
            and axis_range >= MIN_AXIS_RANGE
        )
        if long_enough and oscillating:
            self.state = State.SCRATCHING
            self._scratch_started = timestamp
            return self._result(True, region, near, movement, reversals)

        return self._result(False, region, near, movement, reversals)

    @staticmethod
    def _motion_point(
        landmarks: Mapping[str, Landmark],
        side: str,
        visible_hand_points: list[Landmark],
    ) -> Optional[Landmark]:
        """Prefer the stable wrist; otherwise average the visible hand points."""
        wrist = landmarks.get(f"{side}_wrist")
        if _visible(wrist):
            return wrist
        if not visible_hand_points:
            return None
        return Landmark(
            x=sum(point.x for point in visible_hand_points) / len(visible_hand_points),
            y=sum(point.y for point in visible_hand_points) / len(visible_hand_points),
            visibility=min(point.visibility for point in visible_hand_points),
        )

    def _trim_history(self, timestamp: float) -> None:
        cutoff = timestamp - MOTION_WINDOW_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _motion_metrics(self) -> Tuple[float, int, float]:
        if len(self._history) < 2:
            return 0.0, 0, 0.0

        points = list(self._history)
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        use_x = max(xs) - min(xs) >= max(ys) - min(ys)
        values = xs if use_x else ys
        axis_range = max(values) - min(values)

        movement = 0.0
        reversals = 0
        previous_direction = 0
        for previous, current in zip(values, values[1:]):
            delta = current - previous
            if abs(delta) < MIN_MOTION_STEP:
                continue
            movement += abs(delta)
            direction = 1 if delta > 0 else -1
            if previous_direction and direction != previous_direction:
                reversals += 1
            previous_direction = direction
        return movement, reversals, axis_range

    def _reset(self) -> None:
        self.state = State.IDLE
        self._active_hand = None
        self._history.clear()

    def _result(
        self,
        scratch: bool,
        region: Optional[HeadRegion],
        near: Mapping[str, bool],
        movement: Optional[float] = None,
        reversals: Optional[int] = None,
    ) -> DetectorResult:
        if movement is None or reversals is None:
            movement, reversals, _ = self._motion_metrics()
        return DetectorResult(
            scratch=scratch,
            state=self.state.value,
            head_region=region,
            left_near_head=near["left"],
            right_near_head=near["right"],
            active_hand=self._active_hand,
            movement=movement,
            reversals=reversals,
        )
