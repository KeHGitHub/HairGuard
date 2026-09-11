"""Minimal camera-based pulse estimation for HairGuard.

This is a wellness signal, not a medical measurement. It stores only rolling
colour averages from two cheek regions; camera frames are never retained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping, Optional, Tuple

import numpy as np

from scratch_detector import Landmark


# The single user-facing tradeoff: longer windows are steadier but react slower.
ANALYSIS_WINDOW_SECONDS = 10.0

# Internal stability policy. These deliberately are not exposed as tuning knobs:
# the app keeps a short history of FFT estimates, rejects isolated peaks, and
# briefly holds the last reliable average when signal quality dips.
_MINIMUM_AVERAGE_SAMPLES = 3
_LOW_SIGNAL_HOLD_SECONDS = 2.5

PixelRegion = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RPPGResult:
    bpm: Optional[float]
    status: str
    regions: Tuple[PixelRegion, ...] = ()


class RPPGMonitor:
    """Estimate pulse from subtle RGB changes in two facial skin regions."""

    def __init__(self, window_seconds: float = ANALYSIS_WINDOW_SECONDS) -> None:
        self.window_seconds = max(6.0, float(window_seconds))
        self._samples: Deque[Tuple[float, float, float, float]] = deque()
        self._bpm_candidates: Deque[Tuple[float, float, float]] = deque()
        self._last_estimate_at = float("-inf")
        self._last_good_estimate_at = float("-inf")
        self._smoothed_bpm: Optional[float] = None
        self._last_result = RPPGResult(None, "NO FACE")

    def update(
        self,
        frame: np.ndarray,
        landmarks: Mapping[str, Landmark],
        timestamp: float,
    ) -> RPPGResult:
        regions = self._cheek_regions(frame.shape, landmarks)
        self._discard_old_samples(timestamp)
        if not regions:
            self._reset_signal()
            self._last_result = RPPGResult(None, "NO FACE")
            return self._last_result

        colour = self._mean_bgr(frame, regions)
        if colour is None:
            self._reset_signal()
            self._last_result = RPPGResult(None, "LOW LIGHT", regions)
            return self._last_result

        blue, green, red = colour
        self._samples.append((timestamp, red, green, blue))
        self._discard_old_samples(timestamp)

        minimum_seconds = min(6.0, self.window_seconds * 0.65)
        duration = self._sample_duration()
        if duration < minimum_seconds:
            remaining = max(1, int(np.ceil(minimum_seconds - duration)))
            self._last_result = RPPGResult(
                None,
                f"ACQUIRING {remaining}s",
                regions,
            )
            return self._last_result

        # FFT work twice per second is enough; keep sampling colour every frame.
        if timestamp - self._last_estimate_at < 0.5:
            return RPPGResult(
                self._last_result.bpm,
                self._last_result.status,
                regions,
            )

        self._last_estimate_at = timestamp
        candidate, quality = self._estimate_bpm()
        if candidate is None or quality < 4.0:
            if (
                self._smoothed_bpm is not None
                and timestamp - self._last_good_estimate_at <= _LOW_SIGNAL_HOLD_SECONDS
            ):
                self._last_result = RPPGResult(
                    self._smoothed_bpm,
                    f"{self._smoothed_bpm:.0f} BPM AVG",
                    regions,
                )
            else:
                self._last_result = RPPGResult(None, "LOW SIGNAL", regions)
            return self._last_result

        averaged_bpm = self._average_bpm_candidate(candidate, quality, timestamp)
        if averaged_bpm is None:
            self._last_result = RPPGResult(None, "STABILIZING", regions)
            return self._last_result

        self._last_result = RPPGResult(
            averaged_bpm,
            f"{averaged_bpm:.0f} BPM AVG",
            regions,
        )
        return self._last_result

    @staticmethod
    def _cheek_regions(
        frame_shape: Tuple[int, ...],
        landmarks: Mapping[str, Landmark],
    ) -> Tuple[PixelRegion, ...]:
        required = ("nose", "left_eye", "right_eye")
        if any(
            name not in landmarks or landmarks[name].visibility < 0.5
            for name in required
        ):
            return ()

        height, width = frame_shape[:2]
        nose = landmarks["nose"]
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        nose_x, nose_y = nose.x * width, nose.y * height
        eye_y = (left_eye.y + right_eye.y) * 0.5 * height
        eye_span = abs(left_eye.x - right_eye.x) * width

        ear_span = 0.0
        if all(
            name in landmarks and landmarks[name].visibility >= 0.5
            for name in ("left_ear", "right_ear")
        ):
            ear_span = abs(
                landmarks["left_ear"].x - landmarks["right_ear"].x
            ) * width

        face_width = max(eye_span * 2.2, ear_span * 0.9)
        if face_width < 24:
            return ()

        y0 = eye_y + max(2.0, (nose_y - eye_y) * 0.18)
        y1 = nose_y + face_width * 0.10
        cheek_boxes = (
            (
                nose_x - face_width * 0.34,
                y0,
                nose_x - face_width * 0.08,
                y1,
            ),
            (
                nose_x + face_width * 0.08,
                y0,
                nose_x + face_width * 0.34,
                y1,
            ),
        )

        regions = []
        for x0, top, x1, bottom in cheek_boxes:
            clipped = (
                max(0, min(width - 1, int(x0))),
                max(0, min(height - 1, int(top))),
                max(1, min(width, int(x1))),
                max(1, min(height, int(bottom))),
            )
            if clipped[2] - clipped[0] >= 4 and clipped[3] - clipped[1] >= 4:
                regions.append(clipped)
        return tuple(regions)

    @staticmethod
    def _mean_bgr(
        frame: np.ndarray,
        regions: Tuple[PixelRegion, ...],
    ) -> Optional[Tuple[float, float, float]]:
        channel_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        for x0, y0, x1, y1 in regions:
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            channel_sum += patch.sum(axis=(0, 1), dtype=np.float64)
            pixel_count += patch.shape[0] * patch.shape[1]
        if pixel_count == 0:
            return None

        mean = channel_sum / pixel_count
        brightness = float(mean.mean())
        if brightness < 20.0 or brightness > 245.0:
            return None
        return float(mean[0]), float(mean[1]), float(mean[2])

    def _discard_old_samples(self, timestamp: float) -> None:
        cutoff = timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _reset_signal(self) -> None:
        self._samples.clear()
        self._bpm_candidates.clear()
        self._smoothed_bpm = None
        self._last_estimate_at = float("-inf")
        self._last_good_estimate_at = float("-inf")

    def _average_bpm_candidate(
        self,
        candidate: float,
        quality: float,
        timestamp: float,
    ) -> Optional[float]:
        """Return a stable rolling BPM average, ignoring isolated FFT peaks."""
        self._bpm_candidates.append((timestamp, candidate, quality))
        cutoff = timestamp - max(3.0, self.window_seconds * 0.5)
        while self._bpm_candidates and self._bpm_candidates[0][0] < cutoff:
            self._bpm_candidates.popleft()

        if len(self._bpm_candidates) < _MINIMUM_AVERAGE_SAMPLES:
            return None

        values = np.asarray(
            [sample[1] for sample in self._bpm_candidates],
            dtype=np.float64,
        )
        qualities = np.asarray(
            [sample[2] for sample in self._bpm_candidates],
            dtype=np.float64,
        )
        median = float(np.median(values))
        median_deviation = float(np.median(np.abs(values - median)))
        # A minimum 6 BPM band handles normal FFT-bin wobble. MAD expands the
        # band when the recent estimates genuinely have more variation.
        inlier_limit = max(6.0, 3.0 * 1.4826 * median_deviation)
        inliers = np.abs(values - median) <= inlier_limit
        if int(np.count_nonzero(inliers)) < _MINIMUM_AVERAGE_SAMPLES:
            return None

        # Quality helps reliable peaks count more, but the cap prevents a single
        # suspiciously sharp peak from dominating the displayed average.
        weights = np.clip(qualities[inliers], 1.0, 20.0)
        target = float(np.average(values[inliers], weights=weights))

        if self._smoothed_bpm is None:
            self._smoothed_bpm = target
        else:
            # Limit abrupt movement, then ease toward the robust rolling target.
            target = float(
                np.clip(
                    target,
                    self._smoothed_bpm - 8.0,
                    self._smoothed_bpm + 8.0,
                )
            )
            self._smoothed_bpm = 0.80 * self._smoothed_bpm + 0.20 * target

        self._last_good_estimate_at = timestamp
        return self._smoothed_bpm

    def _sample_duration(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def _estimate_bpm(self) -> Tuple[Optional[float], float]:
        if len(self._samples) < 48:
            return None, 0.0

        samples = np.asarray(self._samples, dtype=np.float64)
        times = samples[:, 0]
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return None, 0.0
        sample_rate = 1.0 / float(np.median(intervals))
        if sample_rate < 8.0:
            return None, 0.0

        uniform_times = np.arange(times[0], times[-1], 1.0 / sample_rate)
        if uniform_times.size < 48:
            return None, 0.0

        rgb = np.column_stack(
            [np.interp(uniform_times, times, samples[:, channel]) for channel in range(1, 4)]
        )
        means = rgb.mean(axis=0)
        if np.any(means <= 0):
            return None, 0.0
        normalized = rgb / means - 1.0

        # POS-style chrominance combination suppresses shared lighting changes.
        x_signal = normalized[:, 1] - normalized[:, 2]
        y_signal = normalized[:, 1] + normalized[:, 2] - 2.0 * normalized[:, 0]
        y_std = float(y_signal.std())
        if y_std < 1e-8:
            return None, 0.0
        pulse = x_signal + (float(x_signal.std()) / y_std) * y_signal

        positions = np.linspace(-1.0, 1.0, pulse.size)
        trend = np.polyval(np.polyfit(positions, pulse, 1), positions)
        pulse = (pulse - trend) * np.hanning(pulse.size)
        if float(np.std(pulse)) < 1e-8:
            return None, 0.0

        frequencies = np.fft.rfftfreq(pulse.size, d=1.0 / sample_rate)
        power = np.abs(np.fft.rfft(pulse)) ** 2
        band_indices = np.flatnonzero((frequencies >= 0.75) & (frequencies <= 3.0))
        if band_indices.size < 3:
            return None, 0.0

        band_power = power[band_indices]
        local_peak = int(np.argmax(band_power))
        peak_index = int(band_indices[local_peak])
        peak_frequency = float(frequencies[peak_index])

        if 0 < local_peak < band_power.size - 1:
            left, center, right = band_power[local_peak - 1 : local_peak + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                offset = 0.5 * (left - right) / denominator
                frequency_step = frequencies[1] - frequencies[0]
                peak_frequency += float(np.clip(offset, -0.5, 0.5) * frequency_step)

        noise_floor = float(np.median(band_power)) + 1e-12
        quality = float(band_power[local_peak] / noise_floor)
        return peak_frequency * 60.0, quality
