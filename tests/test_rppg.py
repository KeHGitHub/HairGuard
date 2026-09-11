import math
import unittest

import numpy as np

from rppg import RPPGMonitor
from scratch_detector import Landmark


LANDMARKS = {
    "nose": Landmark(0.50, 0.48),
    "left_eye": Landmark(0.43, 0.40),
    "right_eye": Landmark(0.57, 0.40),
    "left_ear": Landmark(0.32, 0.42),
    "right_ear": Landmark(0.68, 0.42),
}


class RPPGMonitorTests(unittest.TestCase):
    def test_estimates_synthetic_72_bpm_signal(self) -> None:
        monitor = RPPGMonitor()
        result = None
        frames_per_second = 30
        frequency_hz = 72.0 / 60.0

        for index in range(frames_per_second * 12):
            timestamp = index / frames_per_second
            wave = math.sin(2.0 * math.pi * frequency_hz * timestamp)
            frame = np.empty((200, 200, 3), dtype=np.uint8)
            frame[:, :, 0] = round(90 + wave)
            frame[:, :, 1] = round(120 + 5 * wave)
            frame[:, :, 2] = round(145 + 2 * wave)
            result = monitor.update(frame, LANDMARKS, timestamp)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.bpm)
        self.assertAlmostEqual(result.bpm, 72.0, delta=4.0)
        self.assertIn("BPM AVG", result.status)

    def test_rolling_average_rejects_an_isolated_bpm_spike(self) -> None:
        monitor = RPPGMonitor()
        readings = [71.0, 72.0, 73.0, 145.0, 72.0, 71.0]
        averaged = None

        for index, reading in enumerate(readings):
            averaged = monitor._average_bpm_candidate(
                reading,
                quality=8.0,
                timestamp=index * 0.5,
            )

        self.assertIsNotNone(averaged)
        self.assertAlmostEqual(averaged, 72.0, delta=2.0)

    def test_rolling_average_waits_for_three_estimates(self) -> None:
        monitor = RPPGMonitor()

        first = monitor._average_bpm_candidate(70.0, 8.0, 0.0)
        second = monitor._average_bpm_candidate(71.0, 8.0, 0.5)
        third = monitor._average_bpm_candidate(72.0, 8.0, 1.0)

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)

    def test_requires_visible_face_landmarks(self) -> None:
        monitor = RPPGMonitor()
        frame = np.full((200, 200, 3), 120, dtype=np.uint8)

        result = monitor.update(frame, {}, 0.0)

        self.assertEqual(result.status, "NO FACE")
        self.assertIsNone(result.bpm)
        self.assertEqual(result.regions, ())


if __name__ == "__main__":
    unittest.main()
