"""temporal_buffer.py — Sliding-window history buffer per track ID.

Step 3: for every active track, store the last N frames of bounding
boxes, raw confidence scores, and frame indices.  This history is the
foundation for Step 4 (confidence smoothing) and Step 5 (missing
detection recovery).

Public API::
    from pipeline.temporal_buffer import TemporalBuffer

    buffer = TemporalBuffer(window=5)
    detections = buffer.update(detections, frame_idx)
    # Each Detection now has .smoothed_conf populated (Step 4 uses this)
    # Buffer also exposes buffer.get(track_id) for Step 5
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from pipeline.detector import Detection


# ---------------------------------------------------------------------------
# One frame's worth of data stored per track entry
# ---------------------------------------------------------------------------

@dataclass
class FrameSnapshot:
    """A single frame's observation for one track.

    Attributes:
        frame_idx:  Absolute frame number in the video.
        bbox:       (x1, y1, x2, y2) as reported by the tracker.
        conf:       Raw detector confidence for this frame.
    """
    frame_idx: int
    bbox: tuple[float, float, float, float]
    conf: float


# ---------------------------------------------------------------------------
# TemporalBuffer
# ---------------------------------------------------------------------------

class TemporalBuffer:
    """Sliding-window buffer storing per-track history.

    Tracker-agnostic: works identically whether the upstream tracker is
    SORT (``tracker.py``) or DeepSORT (``tracker_deepsort.py``).  The
    buffer only consumes the ``track_id`` field that both trackers set on
    each :class:`~pipeline.detector.Detection` — it has no dependency on
    which tracker generated those IDs.

    For each ``track_id`` seen a :class:`~collections.deque` of
    :class:`FrameSnapshot` objects is maintained.  The deque is capped at
    ``window`` entries; the oldest entry is automatically discarded when a
    new one is appended.

    After :meth:`update` the ``smoothed_conf`` field of every returned
    :class:`~pipeline.detector.Detection` is populated with the moving
    average confidence over the window — satisfying Step 4.

    Args:
        window: Number of past frames to keep per track (3 or 5 recommended).
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        # track_id → deque of FrameSnapshot (oldest-first, newest at right)
        self._history: Dict[int, Deque[FrameSnapshot]] = {}

    # ------------------------------------------------------------------
    # Main per-frame call
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Detection], frame_idx: int
    ) -> List[Detection]:
        """Record the current frame's detections and fill in smoothed confidence.

        For each detection (which must already have ``track_id`` set by the
        upstream tracker — either SORT or DeepSORT):
        1. Append a :class:`FrameSnapshot` to that track's history deque.
        2. Compute ``smoothed_conf`` = mean of all confidence values in the
           window so far.
        3. Set ``detection.smoothed_conf`` in-place.

        Args:
            detections: Tracked detections from the current frame
                        (``track_id`` must be set by SORT or DeepSORT;
                        ``smoothed_conf`` will be filled in by this method).
            frame_idx:  Current frame number (1-based).

        Returns:
            The same list with ``smoothed_conf`` populated on each item.
        """
        for d in detections:
            if d.track_id is None:
                continue  # untracked detection — skip

            # ── 1. Ensure a history deque exists for this track ────────
            if d.track_id not in self._history:
                self._history[d.track_id] = deque(maxlen=self.window)

            # ── 2. Append current snapshot ─────────────────────────────
            self._history[d.track_id].append(
                FrameSnapshot(
                    frame_idx=frame_idx,
                    bbox=d.bbox,
                    conf=d.confidence,
                )
            )

            # ── 3. Compute moving-average confidence (Step 4) ──────────
            history = self._history[d.track_id]
            d.smoothed_conf = float(np.mean([s.conf for s in history]))

        return detections

    # ------------------------------------------------------------------
    # Accessors used by Step 5 (missing detection recovery)
    # ------------------------------------------------------------------

    def get(self, track_id: int) -> Deque[FrameSnapshot]:
        """Return the history deque for *track_id* (empty deque if unknown)."""
        return self._history.get(track_id, deque(maxlen=self.window))

    def last_snapshot(self, track_id: int) -> Optional[FrameSnapshot]:
        """Return the most recent snapshot for *track_id*, or None."""
        history = self._history.get(track_id)
        if not history:
            return None
        return history[-1]

    def active_track_ids(self) -> List[int]:
        """Return all track IDs that have at least one recorded snapshot."""
        return list(self._history.keys())

    def purge(self, track_id: int) -> None:
        """Remove all history for *track_id* (called when a track is deleted)."""
        self._history.pop(track_id, None)

    # ------------------------------------------------------------------
    # Step 5: Missing detection recovery
    # ------------------------------------------------------------------

    def recover(
        self,
        current_track_ids: set[int],
        frame_idx: int,
        gap_tolerance: int = 3,
        conf_decay: float = 0.8,
    ) -> List[Detection]:
        """Generate recovered detections for recently-active but absent tracks.

        Called once per frame *after* :meth:`update`.  For every ``track_id``
        that has history in the buffer but is **not** in *current_track_ids*
        (i.e. the tracker produced no output for it this frame), a synthetic
        Detection is generated from the last known state and marked
        ``recovered=True``.

        **Confidence decay** — the longer the gap, the less certain we are:

        .. math::

            \\text{recovered\\_conf} = \\text{last\\_conf}
                \\times \\text{conf\\_decay}^{\\text{frames\\_since\\_seen}}

        With ``conf_decay=0.8``: after 1 missed frame → 80 % of last confidence,
        after 2 → 64 %, after 3 → 51 %.  This makes recovered boxes visually
        fade as the gap grows, which is clinically intuitive.

        Args:
            current_track_ids: Set of ``track_id`` values present in the
                               tracker's output for the current frame.
            frame_idx:         Current frame number (1-based).
            gap_tolerance:     Maximum frames to bridge.  Should be ≥ the
                               upstream tracker's ``max_age`` (SORT uses 3,
                               DeepSORT uses 5) so the buffer never tries to
                               recover a track the tracker has already
                               confirmed is lost.  Defaults to 3 (conservative).
            conf_decay:        Multiplicative confidence reduction per missed
                               frame (default 0.8 = 20 % decay per frame).

        Returns:
            List of recovered :class:`~pipeline.detector.Detection` objects
            (possibly empty).  Each has ``recovered=True`` and both
            ``confidence`` (decayed raw) and ``smoothed_conf`` (decayed mean)
            set.
        """
        recovered: List[Detection] = []

        for track_id, history in self._history.items():
            if track_id in current_track_ids:
                continue  # track is active this frame — nothing to recover

            if not history:
                continue

            last = history[-1]
            frames_since_seen = frame_idx - last.frame_idx

            # Only bridge gaps within tolerance (1 … gap_tolerance frames)
            if frames_since_seen < 1 or frames_since_seen > gap_tolerance:
                continue

            # ── Decay confidence proportionally to gap length ──────────
            decay = conf_decay ** frames_since_seen
            raw_conf    = last.conf * decay
            # Historical mean also decayed — gives a conservative estimate
            hist_mean   = float(np.mean([s.conf for s in history]))
            smooth_conf = hist_mean * decay

            x1, y1, x2, y2 = last.bbox
            recovered.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=raw_conf,
                    class_id=0,
                    class_name="polyp",
                    track_id=track_id,
                    smoothed_conf=smooth_conf,
                    recovered=True,
                )
            )

        return recovered
