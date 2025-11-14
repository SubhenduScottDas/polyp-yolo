## Project layout

```
polyp-yolo/
├─ data/                       # put raw dataset here (images + masks)
│  ├─ kvasir/                  # original downloaded dataset
│  │  ├─ images/
│  │  └─ masks/
│  └─ processed/               # YOLO-formatted split created by scripts
│     ├─ images/train
│     ├─ images/val
│     ├─ labels/train
│     └─ labels/val
├─ models/                     # training outputs (Ultralytics project)
├─ results/                    # metrics, plots, predictions
├─ scripts/                    # helper scripts (convert, split, infer)
│  ├─ convert_masks_to_yolo.py
│  ├─ split_train_val.py
│  └─ infer_and_viz.py
├─ configs/
│  └─ kvasir.yaml
├─ notebooks/
├─ environment.yml             # optional conda env file
├─ requirements.txt
└─ README.md
```

---

Quick start:

1. Create conda env: `conda env create -f environment.yml` or install requirements from `requirements.txt`.
2. Place dataset in `data/kvasir` with `images/` and `masks/`.
3. Convert masks to YOLO labels:
   `python scripts/convert_masks_to_yolo.py --images data/kvasir/images --masks data/kvasir/masks --labels_out data/kvasir/labels`
4. Split:
   `python scripts/split_train_val.py --images data/kvasir/images --labels data/kvasir/labels --out data/processed --val_ratio 0.2`
5. Train:
   `yolo detect train data=configs/kvasir.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=16 project=models name=yolov8_kvasir`
6. Visualize predictions:
   `python scripts/infer_and_viz.py --weights models/yolov8_kvasir/weights/best.pt --imgs data/processed/images/val --out results/predictions`
   
---

Additional tools

- Video inference (run a trained model on video files):

```bash
python scripts/video_infer_yolo.py \
   --weights runs/detect/train/weights/best.pt \
   --video data/videos/sample.mp4 \
   --out results/video_out.mp4 \
   --csv results/detections.csv \
   --conf 0.25 --imgsz 640 --skip 1
```

- Multiple bbox export from masks:

The original converter writes a single bbox per image. Use the `--multi` flag to export one bbox per connected mask component instead:

```bash
python scripts/convert_masks_to_yolo.py \
   --images data/archive/Kvasir-SEG/Kvasir-SEG/images \
   --masks  data/archive/Kvasir-SEG/Kvasir-SEG/masks \
   --labels_out data/processed/labels --multi
```

- Evaluation on validation split (Ultralytics val):

```bash
python scripts/eval_val.py --weights runs/detect/train/weights/best.pt --data yolo_data.yaml --imgsz 640
```

Notes
- If you don't have real videos, you can synthesize a demo video from images using `ffmpeg`:

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'data/archive/Kvasir-SEG/Kvasir-SEG/images/*.jpg' \
   -c:v libx264 -pix_fmt yuv420p results/sample_from_images.mp4
```

---
If you'd like, I can also add a small evaluation notebook and CI checks to verify scripts run in a minimal CPU environment.
