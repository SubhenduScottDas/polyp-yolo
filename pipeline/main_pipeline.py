"""main_pipeline.py — Temporal CADe pipeline (Steps 1–7 complete).

End-to-end pipeline for temporally consistent colonoscopy polyp
detection, assembling the four pipeline modules:

    Step 1 – YOLO inference wrapper        (``detector.py``)
    Step 2 – SORT multi-object tracker     (``tracker.py``)
    Step 3 – Per-track sliding window      (``temporal_buffer.py``)
    Step 4 – Moving-average conf smoothing (inside ``buffer.update``)
    Step 5 – Missing detection recovery    (``buffer.recover``)
    Step 6 – Annotated video with HUD overlays
    Step 7 – Modular structure + BONUS: continuity & flicker metrics

Usage::
    python pipeline/main_pipeline.py \
        --weights models/polyp_yolov8n/weights/best.pt \
        --video   data/test-set/videos/sample.mp4 \
        --out     results/out_temporal.mp4 \
        --csv     results/out_temporal.csv \
        --conf    0.25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.detector import Detector, Detection
from pipeline.tracker import SORTTracker
from pipeline.temporal_buffer import TemporalBuffer


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    frame_idx: int | None = None,
) -> np.ndarray:
    """Annotate *frame* with bounding boxes, labels, track IDs, and HUD overlays.

    For each detection draws:
    - A coloured bounding box: **green** for a live detection, **yellow**
      for a recovered one (``[R]`` appended to label).
    - A filled label tag showing ``#track_id  class  conf``; black text
      on a coloured background for legibility on any video content.

    Fixed HUD overlays:
    - Frame counter in the top-right corner.
    - Colour legend in the bottom-left corner.

    Colour scheme (BGR):
    - ``(0, 255,   0)`` green  — live / tracked detection
    - ``(0, 255, 255)`` yellow — recovered / interpolated detection

    Args:
        frame:      BGR image array from ``cv2.VideoCapture.read()``.
        detections: Detections to annotate (may be empty).
        frame_idx:  Optional 1-based frame index shown in the HUD.

    Returns:
        A copy of *frame* with all annotations applied.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.50
    THICKNESS = 1

    for d in detections:
        # Yellow for recovered; green for normal tracked detection
        colour = (0, 255, 255) if d.recovered else (0, 255, 0)
        x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # ── Text label: "#id  class  conf  [R]" ──────────────────────
        parts: list[str] = []
        if d.track_id is not None:
            parts.append(f"#{d.track_id}")
        parts.append(d.class_name)
        parts.append(f"{d.display_conf:.2f}")
        if d.recovered:
            parts.append("[R]")
        label = " ".join(parts)

        # ── Filled label tag (black text on coloured background) ──────
        pad = 3
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        # Place tag just above the bounding box; clamp to frame boundaries
        tag_y2 = max(y1 - 2, th + baseline + pad * 2)
        tag_y1 = tag_y2 - th - baseline - pad * 2
        tag_x1 = max(x1, 0)
        tag_x2 = min(tag_x1 + tw + pad * 2, w_img)
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2),
                      colour, cv2.FILLED)
        cv2.putText(
            annotated, label,
            (tag_x1 + pad, tag_y2 - baseline - pad),
            FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA,
        )

    # ── Frame counter (top-right) ────────────────────────────────────
    if frame_idx is not None:
        txt = f"Frame {frame_idx}"
        (cw, ch), _ = cv2.getTextSize(txt, FONT, 0.45, 1)
        cx1 = w_img - cw - 12
        cy2 = ch + 10
        cv2.rectangle(annotated, (cx1 - 4, 4), (w_img - 6, cy2 + 4),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, txt, (cx1, cy2),
                    FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Colour legend (bottom-left) ──────────────────────────────────
    cv2.rectangle(annotated, (4, h_img - 42), (110, h_img - 2),
                  (0, 0, 0), cv2.FILLED)
    cv2.circle(annotated,  (14, h_img - 28), 5, (0, 255, 0),   cv2.FILLED)
    cv2.putText(annotated, "detected",  (24, h_img - 24),
                FONT, 0.38, (0, 255, 0),   1, cv2.LINE_AA)
    cv2.circle(annotated,  (14, h_img - 12), 5, (0, 255, 255), cv2.FILLED)
    cv2.putText(annotated, "recovered", (24, h_img - 8),
                FONT, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def run_pipeline(
    weights: str,
    video_path: str,
    out_video: str,
    out_csv: str | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    skip: int = 1,
) -> None:
    """Process *video_path* frame-by-frame using the temporal pipeline.

    Args:
        weights:    Path to YOLO ``.pt`` weights.
        video_path: Input video file.
        out_video:  Destination for the annotated video.
        out_csv:    Optional path for per-frame detection CSV.
        conf:       Confidence threshold forwarded to the detector.
        imgsz:      Inference image size.
        skip:       Process every Nth frame (1 = every frame).
    """
    # --- Step 1: initialise detector --------------------------------
    print(f"[pipeline] Loading detector from: {weights}")
    detector = Detector(weights=weights, conf=conf, imgsz=imgsz)

    # --- Step 2: initialise SORT tracker ----------------------------
    # max_age=3  : keep a track alive for up to 3 unmatched frames
    #              (bridges brief missed detections without being too
    #               aggressive — tuned for 25-30 fps colonoscopy video)
    # min_hits=1 : report every matched track immediately — clinical
    #              sensitivity is more important than zero false tracks
    # iou_threshold=0.3 : standard SORT default; polyps are mostly
    #              stationary so even 0.3 IoU overlap is reliable
    tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
    print("[pipeline] SORT tracker initialised  "
          "(max_age=3, min_hits=1, iou_threshold=0.3)")

    # --- Step 3: initialise temporal buffer ------------------------
    # window=5: store last 5 frames per track for smoothing & recovery
    # chosen over 3 because colonoscopy videos have frequent short gaps;
    # 5 frames gives a wider baseline for the moving average (Step 4)
    # without introducing noticeable lag in confidence response
    buffer = TemporalBuffer(window=5)
    print("[pipeline] Temporal buffer initialised  (window=5)")

    # --- Video I/O --------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / max(1, skip), (width, height))

    csv_file = None
    csv_writer_obj = None
    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer_obj = csv.writer(csv_file)
        csv_writer_obj.writerow(
            ["frame", "track_id", "class_id", "class_name",
             "conf_raw", "conf_smooth", "x1", "y1", "x2", "y2", "recovered"]
        )

    # --- Main loop --------------------------------------------------
    frame_idx = 0
    written = 0

    # --- BONUS: metrics counters ------------------------------------
    frames_with_detection: int = 0   # frames with ≥1 detection (live or recovered)
    frames_recovered_only: int = 0   # frames where ONLY recovered detections shown
    flicker_count: int = 0           # per-track conf jumps above threshold
    _FLICKER_THRESHOLD: float = 0.15
    _prev_raw_conf: dict[int, float] = {}

    print(f"[pipeline] Processing {total} frames  (skip={skip}) …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if (frame_idx - 1) % skip != 0:
            continue

        # ---- Step 1: detect ----------------------------------------
        detections: list[Detection] = detector.detect(frame)

        # ---- Step 2: track (SORT) ----------------------------------
        # tracker.update() returns Kalman-smoothed bboxes with track_id
        detections = tracker.update(detections)

        # ---- Step 3: push into temporal buffer (also fills smoothed_conf) ----
        detections = buffer.update(detections, frame_idx)

        # ---- Step 4: confidence smoothing (done inside buffer.update) ------
        # smoothed_conf = mean(last window confs) is now set on every Detection.
        # Detection.display_conf returns smoothed_conf automatically.

        # ---- Step 5: missing detection recovery ----------------------------
        # Find track_ids that have recent history but no output this frame,
        # then synthesise a held bbox with decayed confidence, marked [R].
        current_ids = {d.track_id for d in detections if d.track_id is not None}
        recovered   = buffer.recover(current_ids, frame_idx)
        detections  = detections + recovered

        # ---- BONUS: metrics ------------------------------------------
        if detections:
            frames_with_detection += 1
        real_dets = [d for d in detections if not d.recovered]
        if not real_dets and recovered:
            frames_recovered_only += 1
        for d in real_dets:
            if d.track_id is not None:
                prev = _prev_raw_conf.get(d.track_id)
                if prev is not None and abs(d.confidence - prev) > _FLICKER_THRESHOLD:
                    flicker_count += 1
                _prev_raw_conf[d.track_id] = d.confidence

        # ---- Step 6: visualise & write frame -------------------------
        annotated = draw_detections(frame, detections, frame_idx=frame_idx)
        writer.write(annotated)
        written += 1

        # ---- CSV logging -------------------------------------------
        if csv_writer_obj:
            if not detections:
                csv_writer_obj.writerow(
                    [written, None, None, None, None, None,
                     None, None, None, None, False]
                )
            for d in detections:
                csv_writer_obj.writerow([
                    written,
                    d.track_id,
                    d.class_id,
                    d.class_name,
                    f"{d.confidence:.4f}",
                    f"{d.smoothed_conf:.4f}" if d.smoothed_conf is not None else "",
                    f"{d.x1:.1f}", f"{d.y1:.1f}",
                    f"{d.x2:.1f}", f"{d.y2:.1f}",
                    d.recovered,
                ])

        if written % 50 == 0:
            print(f"  … frame {written} / ~{total // max(1, skip)}", end="\r")

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    # ── BONUS: print metrics ─────────────────────────────────────────
    if written > 0:
        continuity = frames_with_detection / written
        print(f"\n[metrics] Detection continuity rate : {continuity:.1%}  "
              f"({frames_with_detection}/{written} frames)")
        print(f"[metrics] Recovery bridges used     : {frames_recovered_only} frame(s)")
        print(f"[metrics] Confidence flicker events : {flicker_count}  "
              f"(|Δconf| > {_FLICKER_THRESHOLD:.2f} threshold)")

    print(f"\n[pipeline] Done.  Written {written} frames → {out_video}")
    if out_csv:
        print(f"[pipeline] CSV saved   → {out_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal polyp detection pipeline (CADe)"
    )
    parser.add_argument("--weights", required=True, help="YOLO .pt weights")
    parser.add_argument("--video",   required=True, help="Input video path")
    parser.add_argument("--out",     default=None,
                        help="Output video path (default: results/phase2/<video_stem>_tracked.mp4)")
    parser.add_argument("--csv",     default=None,
                        help="CSV path (default: results/phase2/<video_stem>_tracked.csv)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--skip",    type=int, default=1,
                        help="Process every Nth frame")
    args = parser.parse_args()

    stem = Path(args.video).stem
    out_video = args.out  or f"results/phase2/{stem}_tracked.mp4"
    out_csv   = args.csv or f"results/phase2/{stem}_tracked.csv"

    run_pipeline(
        weights=args.weights,
        video_path=args.video,
        out_video=out_video,
        out_csv=out_csv,
        conf=args.conf,
        imgsz=args.imgsz,
        skip=args.skip,
    )
