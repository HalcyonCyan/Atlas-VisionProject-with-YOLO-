"""
DistanceEstimator - Monocular distance estimation via known object sizes.
Uses the pinhole camera model:  distance = (real_height * focal_px) / bbox_height
Focal length is calibrated from a reference measurement or set as default.
"""

import math
from typing import List, Tuple, Optional, Dict

# ── Known real-world heights (cm) for common COCO classes ────────────────────
KNOWN_HEIGHTS_CM: Dict[str, float] = {
    'person':       170.0,
    'bicycle':       95.0,
    'car':          145.0,
    'motorcycle':    110.0,
    'bus':          300.0,
    'truck':        250.0,
    'cat':           25.0,
    'dog':           50.0,
    'horse':        160.0,
    'elephant':     250.0,
    'bottle':        25.0,
    'cup':           10.0,
    'laptop':        25.0,
    'cell phone':    15.0,
    'keyboard':       4.0,
    'mouse':          4.0,
    'chair':          90.0,
    'couch':          85.0,
    'tv':             50.0,
    'suitcase':       60.0,
    'book':           25.0,
    'clock':          30.0,
    'vase':           30.0,
    'toothbrush':     18.0,
    'fire hydrant':   60.0,
    'stop sign':      75.0,
    'traffic light': 100.0,
    'bird':           20.0,
    'airplane':     1200.0,
    'boat':          150.0,
}

# Default "unknown object" height
DEFAULT_HEIGHT_CM = 50.0

# Default focal length (pixels) for a 1280x720 camera at ~60° FOV
# f = (width / 2) / tan(H_FOV/2)
DEFAULT_FOCAL_PX = 902.0


class DistanceEstimator:
    """
    Single-camera distance estimator.

    Parameters
    ----------
    focal_px : float
        Camera focal length in pixels. Override with calibrate() or set directly.
    frame_w, frame_h : int
        Frame resolution for FOV calculation.
    """

    def __init__(
        self,
        focal_px: float = DEFAULT_FOCAL_PX,
        frame_w: int = 1280,
        frame_h: int = 720,
    ):
        self.focal_px = focal_px
        self.frame_w = frame_w
        self.frame_h = frame_h

    # ── Calibration ───────────────────────────────────────────────────────────
    def calibrate(
        self,
        class_name: str,
        bbox_height_px: float,
        known_distance_cm: float,
    ):
        """
        Calibrate focal length from a single reference measurement.
        Place object at known_distance_cm and measure its bbox height.
        """
        real_h = KNOWN_HEIGHTS_CM.get(class_name.lower(), DEFAULT_HEIGHT_CM)
        self.focal_px = (bbox_height_px * known_distance_cm) / real_h

    # ── Main estimate ─────────────────────────────────────────────────────────
    def estimate(
        self,
        bbox: List[float],
        class_name: str = '',
        real_height_cm: Optional[float] = None,
    ) -> float:
        """
        Estimate distance in centimetres.

        Parameters
        ----------
        bbox        : [x1, y1, x2, y2]
        class_name  : COCO class name string
        real_height_cm : override known height table
        """
        x1, y1, x2, y2 = bbox[:4]
        bbox_h = max(abs(y2 - y1), 1)

        if real_height_cm is None:
            real_height_cm = KNOWN_HEIGHTS_CM.get(
                class_name.lower(), DEFAULT_HEIGHT_CM)

        distance = (real_height_cm * self.focal_px) / bbox_h
        return round(distance, 1)

    # ── Batch estimate ────────────────────────────────────────────────────────
    def estimate_batch(self, detections: list) -> list:
        for d in detections:
            d['distance_cm'] = self.estimate(d['bbox'], d.get('class_name', ''))
        return detections

    # ── Zone classification ───────────────────────────────────────────────────
    @staticmethod
    def zone(distance_cm: float) -> str:
        """Return 'close', 'medium', 'far', or 'very_far'."""
        if distance_cm < 100:
            return 'close'
        elif distance_cm < 300:
            return 'medium'
        elif distance_cm < 800:
            return 'far'
        return 'very_far'

    # ── Horizontal FOV ────────────────────────────────────────────────────────
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.frame_w / (2 * self.focal_px)))

    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.frame_h / (2 * self.focal_px)))
