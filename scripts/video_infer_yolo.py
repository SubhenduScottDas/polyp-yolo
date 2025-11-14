"""Run a YOLO model on a video, draw bounding boxes and optionally save detections to CSV.

Usage example:
    python scripts/video_infer_yolo.py \
      --weights runs/detect/train/weights/best.pt \
      --video data/videos/sample.mp4 \
      --out results/video_out.mp4 \
      --csv results/detections.csv --conf 0.25 --skip 1
"""
from ultralytics import YOLO
import cv2
import argparse
import os
import csv
from pathlib import Path


def draw_boxes(img, boxes, scores, classes, names):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names.get(int(c), str(int(c)))} {s:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def video_infer(weights, video_path, out_video, out_csv=None, conf=0.25, imgsz=640, skip=1):
    model = YOLO(weights)
    names = model.names if hasattr(model, 'names') else {0: 'polyp'}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = Path(out_video).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), fourcc, fps / max(1, skip), (width, height))

    csv_file = None
    csv_writer = None
    if out_csv:
        csv_file = open(out_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'class_id', 'class_name', 'conf', 'x1', 'y1', 'x2', 'y2'])

    frame_idx = 0
    written_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % skip != 0:
            continue

        # run inference
        res = model.predict(source=frame, conf=conf, imgsz=imgsz, save=False, verbose=False)
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            out_frame = frame
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            out_frame = draw_boxes(frame.copy(), boxes, scores, classes, names)
            if csv_writer:
                for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
                    csv_writer.writerow([written_frames, int(c), names.get(int(c), str(int(c))), float(s), float(x1), float(y1), float(x2), float(y2)])

        writer.write(out_frame)
        written_frames += 1

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    print(f"Done. Output video: {out_video}")
    if out_csv:
        print(f"Detections CSV: {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='YOLO weights (.pt)')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--out', default='results/video_out.mp4', help='Output video path')
    parser.add_argument('--csv', default=None, help='Optional CSV path to save detections')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame (speedup)')
    args = parser.parse_args()
    video_infer(args.weights, args.video, args.out, args.csv, args.conf, args.imgsz, args.skip)
