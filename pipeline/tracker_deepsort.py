"""tracker_deepsort.py — DeepSORT tracking wrapper for the CADe pipeline.

Wraps the ``deep-sort-realtime`` library to expose an interface that is
**drop-in compatible** with :class:`pipeline.tracker.SORTTracker`:

    update(detections, frame) → List[Detection]

Each returned :class:`~pipeline.detector.Detection` has its ``track_id``
field set to the DeepSORT-assigned integer ID.  Only *confirmed* tracks
(those that survived the ``n_init`` probation period) are returned.

DeepSORT uses a MobileNet appearance embedder to build a Re-ID feature
vector for each detected crop.  This feature is used in addition to IoU
to associate detections across frames, making it more robust to occlusion
and brief disappearances than IoU-only SORT.

Usage::
    from pipeline.tracker_deepsort import DeepSortTracker
    from pipeline.detector import Detector

    tracker = DeepSortTracker(max_age=5, n_init=2, max_cosine_distance=0.4)
    det     = Detector("models/.../best.pt", conf=0.25)

    # inside frame loop:
    detections = det.detect(frame)
    tracked    = tracker.update(detections, frame)   # List[Detection]
    for d in tracked:
        print(d.track_id, d.bbox, d.confidence)

Dependencies::
    pip install deep-sort-realtime
"""

from __future__ import annotations

import numpy as np
from typing import List

from deep_sort_realtime.deepsort_tracker import DeepSort

from pipeline.detector import Detection


# ---------------------------------------------------------------------------
# Crop extraction helper
# ---------------------------------------------------------------------------

def extract_crops(frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
    """Extract image patches (crops) from *frame* for each detection bbox.

    DeepSORT's embedder requires one RGB crop per detection.  Crops are
    clamped to the frame boundaries so invalid boxes never cause an out-of-
    bounds slice.

    Args:
        frame:      BGR image as returned by ``cv2.VideoCapture.read()``,
                    shape (H, W, 3), dtype uint8.
        detections: List of :class:`~pipeline.detector.Detection` objects
                    whose ``.x1 / .y1 / .x2 / .y2`` give pixel coordinates.

    Returns:
        List of BGR uint8 crops, one per detection, in the same order.
        Each crop has shape (h', w', 3) and positive area.
    """
    h, w = frame.shape[:2]
    crops: List[np.ndarray] = []
    for d in detections:
        x1 = max(0, int(d.x1))
        y1 = max(0, int(d.y1))
        x2 = min(w, int(d.x2))
        y2 = min(h, int(d.y2))
        # Guard against degenerate boxes (zero area after clamping)
        if x2 <= x1 or y2 <= y1:
            # Fall back to a 1×1 patch at the clamped origin
            x2 = x1 + 1
            y2 = y1 + 1
        crops.append(frame[y1:y2, x1:x2])
    return crops


# ---------------------------------------------------------------------------
# DeepSortTracker
# ---------------------------------------------------------------------------

class DeepSortTracker:
    """DeepSORT tracker with the same interface as :class:`pipeline.tracker.SORTTracker`.

    Parameters
    ----------
    max_age:
        Maximum number of frames a track may go un-matched before it is
        deleted.  Equivalent to SORT's ``max_age``.
    n_init:
        Number of consecutive detections required before a track is
        *confirmed* and returned to the caller.  Tentative tracks are
        invisible outside this class.
    max_cosine_distance:
        Gating threshold on the cosine similarity between the stored
        appearance embedding and a new candidate crop feature.  Lower
        values enforce stricter appearance matching.
    embedder_gpu:
        If ``True``, run the MobileNet embedder on GPU (faster).
        Set to ``False`` on CPU-only machines.
    """

    def __init__(
        self,
        max_age: int = 5,
        n_init: int = 2,
        max_cosine_distance: float = 0.4,
        embedder_gpu: bool = False,
    ) -> None:
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
            half=False,          # FP32 for CPU compatibility
            bgr=True,            # OpenCV frames are BGR
            embedder_gpu=embedder_gpu,
        )

    # ------------------------------------------------------------------
    # Public interface — matches SORTTracker.update(detections, frame)
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[Detection]:
        """Run DeepSORT on one frame.

        Args:
            detections: Raw YOLO detections from :class:`~pipeline.detector.Detector`.
                        ``track_id`` fields are expected to be ``None`` on entry.
            frame:      The **original** BGR frame (before any annotation).
                        Used by the MobileNet embedder to extract appearance
                        features from each detection crop.

        Returns:
            List of :class:`~pipeline.detector.Detection` objects, one per
            **confirmed** track.  Each object carries:
              - the Kalman-predicted bounding box from DeepSORT
              - the raw detector confidence of the most recent matched detection
              - ``track_id`` set to the DeepSORT track integer identifier

            Callers receive an empty list when no confirmed tracks exist
            (e.g., during the first ``n_init`` frames for new tracks).
        """
        if len(detections) == 0 or frame is None:
            # Still tick the tracker so stale tracks age out correctly
            self._tracker.update_tracks([], frame=frame)
            return []

        # ---- convert Detection → deep-sort-realtime input format ----------
        # Expected: List[([left, top, width, height], confidence, class_id)]
        raw: list = []
        for d in detections:
            left   = d.x1
            top    = d.y1
            width  = d.x2 - d.x1
            height = d.y2 - d.y1
            raw.append(([left, top, width, height], d.confidence, d.class_id))

        # ---- run DeepSORT (embedder crops from frame automatically) --------
        tracks = self._tracker.update_tracks(raw, frame=frame)

        # ---- convert confirmed tracks back to Detection objects ------------
        tracked: List[Detection] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            # to_ltrb() → [x1, y1, x2, y2]  (Kalman-smoothed position)
            x1, y1, x2, y2 = track.to_ltrb()

            # get_det_conf() returns the confidence of the last matched det
            conf = track.get_det_conf()
            if conf is None:
                conf = 0.0   # track was recovered: use 0.0 as sentinel

            tracked.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=0,
                    class_name="polyp",
                    track_id=int(track.track_id),
                )
            )

        return tracked
