"""detector.py — YOLO inference wrapper.

Provides a clean ``detect(frame) → List[Detection]`` interface that
decouples raw YOLO calls from the rest of the pipeline (tracker,
temporal buffer, visualisation).

Usage::
    from pipeline.detector import Detector

    det = Detector("models/polyp_yolov8n/weights/best.pt", conf=0.25)
    detections = det.detect(frame)      # List[Detection]
    for d in detections:
        print(d.bbox, d.confidence)     # (x1,y1,x2,y2), float
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single bounding-box detection for one frame.

    Attributes:
        x1, y1, x2, y2: Absolute pixel coordinates (top-left / bottom-right).
        confidence:      Raw model confidence in [0, 1].
        class_id:        Integer class index (0 = polyp).
        class_name:      Human-readable class label.
        track_id:        Assigned by the tracker; None until tracked.
        smoothed_conf:   Set by TemporalBuffer; None until smoothed.
        recovered:       True if this detection was interpolated / recovered.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = field(default=None, compare=False)
    smoothed_conf: Optional[float] = field(default=None, compare=False)
    recovered: bool = field(default=False, compare=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def display_conf(self) -> float:
        """Return smoothed confidence if available, otherwise raw confidence."""
        return self.smoothed_conf if self.smoothed_conf is not None else self.confidence

    def to_sort_row(self) -> np.ndarray:
        """Return shape-(5,) array [x1, y1, x2, y2, conf] expected by SORT."""
        return np.array(
            [self.x1, self.y1, self.x2, self.y2, self.confidence],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        tid = f" track={self.track_id}" if self.track_id is not None else ""
        rec = " [RECOVERED]" if self.recovered else ""
        return (
            f"Detection({self.class_name} conf={self.confidence:.2f}"
            f" bbox=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f})"
            f"{tid}{rec})"
        )


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """Thin wrapper around an Ultralytics YOLO model.

    The only public method is :meth:`detect`, which accepts a BGR numpy
    frame (as returned by ``cv2.VideoCapture.read``) and returns a list
    of :class:`Detection` objects.

    Args:
        weights:  Path to YOLO ``.pt`` weights file.
        conf:     Minimum detection confidence threshold (default 0.25).
        imgsz:    Inference image size in pixels (default 640).
        device:   Torch device string, e.g. ``"cpu"``, ``"cuda:0"``, or
                  ``""`` to let Ultralytics choose automatically.
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "",
    ) -> None:
        self.conf = conf
        self.imgsz = imgsz
        self._model = YOLO(weights)
        self._names: dict[int, str] = (
            self._model.names
            if hasattr(self._model, "names")
            else {0: "polyp"}
        )
        # Warm-up: avoids first-frame latency spike (single blank frame)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy,
            conf=conf,
            imgsz=imgsz,
            device=device if device else None,
            save=False,
            verbose=False,
        )

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame: H×W×3 uint8 numpy array in BGR colour order.

        Returns:
            A (possibly empty) list of :class:`Detection` objects.
            Each object carries raw coordinates, raw confidence, and
            class information.  ``track_id`` and ``smoothed_conf`` are
            left as ``None`` — they are filled by the tracker and buffer.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )
        return self._parse(results[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, result) -> List[Detection]:
        """Convert a single Ultralytics result into a list of Detections."""
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)
        scores = result.boxes.conf.cpu().numpy()   # (N,)
        classes = result.boxes.cls.cpu().numpy()   # (N,)

        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
            cid = int(cls)
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=cid,
                    class_name=self._names.get(cid, str(cid)),
                )
            )
        return detections
