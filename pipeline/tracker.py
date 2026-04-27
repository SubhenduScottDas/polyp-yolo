"""tracker.py — SORT (Simple Online and Realtime Tracking) implementation.

Self-contained SORT tracker built on top of:
  - filterpy  : Kalman filter
  - scipy     : Hungarian algorithm (linear_sum_assignment)
  - numpy     : IoU computations

References:
  Bewley et al. "Simple Online and Realtime Tracking" (ICIP 2016)
  https://arxiv.org/abs/1602.00763

Public API::
    from pipeline.tracker import SORTTracker

    tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
    tracked = tracker.update(detections)   # List[Detection] → List[Detection]

Each returned Detection has ``track_id`` set.  Bounding boxes are the
Kalman-smoothed positions (slightly refined vs. raw detector output).
The original detector confidence is preserved on each Detection.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List

from pipeline.detector import Detection


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman measurement [cx, cy, s, r].

    s = area (scale),  r = width / height (aspect ratio, kept constant).
    Shape of returned array: (4, 1).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # area
    r = w / float(h)   # aspect ratio
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, s, r, vcx, vcy, vs] back to [x1,y1,x2,y2].

    Returns shape (4,).
    """
    s = max(x[2, 0], 1e-6)   # guard: area must be positive
    r = max(x[3, 0], 1e-6)   # guard: aspect ratio must be positive
    w = np.sqrt(s * r)
    h = s / w
    cx, cy = x[0, 0], x[1, 0]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# IoU matrix
# ──────────────────────────────────────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bboxes.

    Args:
        a: shape (N, 4)  — [x1, y1, x2, y2]
        b: shape (M, 4)  — [x1, y1, x2, y2]

    Returns:
        iou_matrix: shape (N, M)
    """
    a = np.expand_dims(a, 1)   # (N,1,4)
    b = np.expand_dims(b, 0)   # (1,M,4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union_area = area_a + area_b - inter_area + 1e-6

    return inter_area / union_area


def _hungarian_match(iou_matrix: np.ndarray, threshold: float):
    """Run Hungarian algorithm and filter matches below *threshold*.

    Returns:
        matched    : list of (track_idx, det_idx) pairs
        unmatched_t: set of unmatched track indices
        unmatched_d: set of unmatched detection indices
    """
    n_tracks, n_dets = iou_matrix.shape

    if n_dets == 0:
        return [], set(range(n_tracks)), set()
    if n_tracks == 0:
        return [], set(), set(range(n_dets))

    # linear_sum_assignment minimises cost → maximise IoU via negation
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched, unmatched_t, unmatched_d = [], set(), set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matched.append((r, c))
        else:
            unmatched_t.add(r)
            unmatched_d.add(c)

    unmatched_t |= set(range(n_tracks)) - {r for r, _ in matched} - unmatched_t
    unmatched_d |= set(range(n_dets))  - {c for _, c in matched} - unmatched_d

    return matched, unmatched_t, unmatched_d


# ──────────────────────────────────────────────────────────────────────────────
# Single-object Kalman tracker
# ──────────────────────────────────────────────────────────────────────────────

class _KalmanBoxTracker:
    """Kalman filter tracker for a single bounding box.

    State vector  x = [cx, cy, s, r, vcx, vcy, vs]  (7-D)
    Measurement   z = [cx, cy, s, r]                 (4-D)

    - s  : area (scale)
    - r  : aspect ratio (assumed near-constant)
    - vcx, vcy, vs: respective velocities
    """

    _count = 0   # class-level counter for unique IDs

    def __init__(self, bbox: np.ndarray, confidence: float) -> None:
        _KalmanBoxTracker._count += 1
        self.track_id: int = _KalmanBoxTracker._count
        self.confidence: float = confidence  # most recent matched confidence
        self.hits: int = 1       # consecutive matched frames
        self.age: int = 0        # frames since last match
        self.time_since_update: int = 0

        kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix F (constant velocity model)
        kf.F = np.eye(7, dtype=np.float32)
        kf.F[0, 4] = 1.0  # cx  += vcx
        kf.F[1, 5] = 1.0  # cy  += vcy
        kf.F[2, 6] = 1.0  # s   += vs

        # Measurement function H: observe [cx, cy, s, r] from state
        kf.H = np.zeros((4, 7), dtype=np.float32)
        kf.H[:4, :4] = np.eye(4)

        # Measurement noise R — higher for s and r (noisier)
        kf.R = np.eye(4, dtype=np.float32)
        kf.R[2, 2] *= 10.0
        kf.R[3, 3] *= 10.0

        # Initial state covariance P — large for velocities (unknown)
        kf.P = np.eye(7, dtype=np.float32)
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        # Process noise Q — small for velocities
        kf.Q = np.eye(7, dtype=np.float32)
        kf.Q[4:, 4:] *= 0.01
        kf.Q[6, 6] *= 0.01

        # Initialise state from first detection
        kf.x[:4] = _bbox_to_z(bbox)

        self._kf = kf

    def predict(self) -> np.ndarray:
        """Advance Kalman state by one step; return predicted [x1,y1,x2,y2]."""
        # Guard: prevent negative area
        if self._kf.x[2, 0] + self._kf.x[6, 0] <= 0:
            self._kf.x[6, 0] = 0.0
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1
        return _z_to_bbox(self._kf.x)

    def update(self, bbox: np.ndarray, confidence: float) -> None:
        """Update Kalman state with a new matched detection."""
        self._kf.update(_bbox_to_z(bbox))
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        """Return current estimated [x1, y1, x2, y2]."""
        return _z_to_bbox(self._kf.x)


# ──────────────────────────────────────────────────────────────────────────────
# Public SORT Tracker
# ──────────────────────────────────────────────────────────────────────────────

class SORTTracker:
    """Multi-object SORT tracker wrapping the pipeline ``Detection`` type.

    Args:
        max_age:       Frames a track may go unmatched before deletion.
                       Set to 1 for strict SORT; raise to 3–5 to bridge
                       short gaps in polyp detection.
        min_hits:      Minimum consecutive matched frames before a track is
                       reported.  Set to 1 so every detection is reflected
                       immediately (important for clinical sensitivity).
        iou_threshold: Minimum IoU required to match a track to a detection.

    Usage::
        tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
        tracked = tracker.update(detections)    # called once per frame
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_KalmanBoxTracker] = []
        self._frame_count: int = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracks (use between unrelated video sequences)."""
        self._tracks = []
        self._frame_count = 0
        _KalmanBoxTracker._count = 0

    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame: "np.ndarray | None" = None,  # accepted for interface parity with DeepSortTracker; unused by SORT
    ) -> List[Detection]:
        """Process one frame of detections and return tracked detections.

        Algorithm (per frame):
        1. Predict each existing track's new position via Kalman filter.
        2. Build IoU matrix between predicted tracks and new detections.
        3. Solve assignment via Hungarian algorithm.
        4. Update matched tracks; start new tracks for unmatched detections.
        5. Delete tracks that have been unmatched too long (> max_age).
        6. Return confirmed tracks (hits >= min_hits) as Detection objects
           with ``track_id`` filled in.

        Args:
            detections: List of :class:`~pipeline.detector.Detection` objects
                        from the current frame (``track_id`` must be None).
            frame:      Ignored by SORT (no appearance model); accepted so
                        both SORT and DeepSORT trackers share the same call
                        signature ``update(detections, frame)``.

        Returns:
            List of :class:`~pipeline.detector.Detection` objects — one per
            *active confirmed track* — with:
            - ``track_id`` set to the SORT-assigned integer ID
            - Bounding box refined by the Kalman filter
            - ``confidence`` copied from the last matched detector output
        """
        self._frame_count += 1

        # ── 1. Predict all tracks ──────────────────────────────────────
        predicted_bboxes = np.array(
            [t.predict() for t in self._tracks], dtype=np.float32
        )  # shape (T, 4)

        # ── 2. Convert current detections to numpy ─────────────────────
        if detections:
            det_bboxes = np.array(
                [[d.x1, d.y1, d.x2, d.y2] for d in detections],
                dtype=np.float32,
            )  # shape (D, 4)
        else:
            det_bboxes = np.empty((0, 4), dtype=np.float32)

        # ── 3. Hungarian matching ──────────────────────────────────────
        if len(self._tracks) > 0 and len(detections) > 0:
            iou_mat = _iou_batch(predicted_bboxes, det_bboxes)  # (T, D)
            matched, unmatched_t, unmatched_d = _hungarian_match(
                iou_mat, self.iou_threshold
            )
        else:
            matched = []
            unmatched_t = set(range(len(self._tracks)))
            unmatched_d = set(range(len(detections)))

        # ── 4a. Update matched tracks ──────────────────────────────────
        for t_idx, d_idx in matched:
            self._tracks[t_idx].update(det_bboxes[d_idx], detections[d_idx].confidence)

        # ── 4b. Start new tracks for unmatched detections ─────────────
        for d_idx in unmatched_d:
            d = detections[d_idx]
            self._tracks.append(
                _KalmanBoxTracker(det_bboxes[d_idx], d.confidence)
            )

        # ── 5. Remove stale tracks ─────────────────────────────────────
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # ── 6. Build output: confirmed tracks only ─────────────────────
        output: List[Detection] = []
        for track in self._tracks:
            if track.hits < self.min_hits and self._frame_count > self.min_hits:
                continue   # not yet confirmed

            bbox = track.get_state()  # Kalman-refined [x1,y1,x2,y2]
            output.append(
                Detection(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                    confidence=track.confidence,
                    class_id=0,
                    class_name="polyp",
                    track_id=track.track_id,
                )
            )

        return output
