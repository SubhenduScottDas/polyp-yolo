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
