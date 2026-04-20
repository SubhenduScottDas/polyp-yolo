"""pipeline — Temporal CADe pipeline for colonoscopy polyp detection.

Public exports
--------------
Detector      : YOLO inference wrapper (Step 1)
Detection     : per-frame bounding-box dataclass
SORTTracker   : SORT multi-object tracker (Step 2)
TemporalBuffer: sliding-window history + smoothing + recovery (Steps 3–5)
FrameSnapshot : single-frame observation stored in the buffer
"""
from .detector import Detector, Detection
from .tracker import SORTTracker
from .temporal_buffer import TemporalBuffer, FrameSnapshot

__all__ = ["Detector", "Detection", "SORTTracker", "TemporalBuffer", "FrameSnapshot"]
