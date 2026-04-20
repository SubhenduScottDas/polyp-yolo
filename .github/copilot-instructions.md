# GitHub Copilot Instructions - Polyp YOLO Detection

*Updated: April 2026 | Originally generated: November 14, 2025*

> **Note:** This file lives at `.github/copilot-instructions.md` — the standard location auto-loaded by GitHub Copilot in VS Code and on GitHub.com. Personal session prompts are stored locally in `copilot/` (gitignored).

## Project Overview
This repository implements a **YOLO-based polyp detection system** for medical imaging. The project converts segmentation masks from the Kvasir-SEG dataset into YOLO bounding box format and provides a complete pipeline for training, inference, and evaluation. A Phase 2 temporal CADe pipeline (SORT tracking + confidence smoothing) was added in April 2026.

### Architecture
```
Data Flow: Segmentation Masks → YOLO Labels → Training → Inference (Images/Videos) → Evaluation
Phase 1:  convert_masks_to_yolo.py → split_train_val.py → [train] → video_infer_yolo.py → eval_val.py
Phase 2:  pipeline/detector.py → pipeline/tracker.py → pipeline/temporal_buffer.py → pipeline/main_pipeline.py
```

## Core Components

### 1. Data Conversion (`scripts/convert_masks_to_yolo.py`)
- **Purpose**: Convert binary segmentation masks to YOLO bounding box format
- **Key Feature**: Supports multi-component mask detection with `--multi` flag
- **Usage**:
  ```bash
  python scripts/convert_masks_to_yolo.py \
    --images     data/archive/Kvasir-SEG/Kvasir-SEG/images \
    --masks      data/archive/Kvasir-SEG/Kvasir-SEG/masks \
    --labels_out data/processed/labels
  # With multi-component support:
  python scripts/convert_masks_to_yolo.py \
    --images     data/archive/Kvasir-SEG/Kvasir-SEG/images \
    --masks      data/archive/Kvasir-SEG/Kvasir-SEG/masks \
    --labels_out data/processed/labels \
    --multi
  ```
- **Multi-component Logic**: Uses `cv2.findContours()` to detect separate connected components in masks and creates individual bounding boxes for each

### 2. Dataset Splitting (`scripts/split_train_val.py`)
- **Purpose**: Random train/validation split (80/20 default)
- **Output**: Creates `data/processed/images/{train,val}/` and `data/processed/labels/{train,val}/` structure
- **Usage**:
  ```bash
  python scripts/split_train_val.py \
    --images data/archive/Kvasir-SEG/Kvasir-SEG/images \
    --labels data/processed/labels \
    --out    data/processed
  ```

### 3. Training Configuration (`yolo_data.yaml`)
- **Dataset**: Single class (`nc: 1`, `names: ['polyp']`)
- **Paths**: Points to `data/processed/images/train` and `data/processed/images/val`
- **Training Command**:
  ```bash
  yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=50 imgsz=640 batch=16 name=polyp_yolov8n
  ```

### 4. Inference Scripts
#### Image Inference (`scripts/infer_and_viz.py`)
- **Purpose**: Single image detection with visualization
- **Output**: Annotated images with bounding boxes

#### Video Inference — Phase 1 (`scripts/video_infer_yolo.py`)
- **Purpose**: Frame-by-frame video processing (no temporal memory)
- **Features**: Annotated video output, optional CSV logging (`frame,class_id,class_name,conf,x1,y1,x2,y2`)
- **Usage**:
  ```bash
  python scripts/video_infer_yolo.py \
    --weights models/polyp_yolov8n_clean/weights/best.pt \
    --video   data/test-set/videos/PolipoMSDz2.mpg \
    --out     results/PolipoMSDz2_annotated.mp4 \
    --csv     results/PolipoMSDz2_detections.csv \
    --conf    0.5
  ```

#### Video Inference — Phase 2 (`pipeline/main_pipeline.py`)
- **Purpose**: SORT-tracked, confidence-smoothed, gap-recovering temporal pipeline
- **Usage**:
  ```bash
  python pipeline/main_pipeline.py \
    --weights models/polyp_yolov8n_clean/weights/best.pt \
    --video   data/test-set/videos/PolipoMSDz2.mpg \
    --conf    0.5
  ```
- **Output CSV columns**: `frame, track_id, class_id, class_name, conf_raw, conf_smooth, x1, y1, x2, y2, recovered`

### 5. Evaluation (`scripts/eval_val.py`)
- **Purpose**: Run YOLO validation metrics on test set
- **Metrics**: mAP@50, mAP@50-95, precision, recall
- **Usage**: `python scripts/eval_val.py --weights models/polyp_yolov8n_clean/weights/best.pt`

## Development Guidelines

### Code Style
- **Imports**: Standard library → Third party → Local imports
- **Functions**: Type hints preferred, docstrings for complex logic
- **Error Handling**: Graceful failure with informative messages
- **Paths**: Use `pathlib.Path` for cross-platform compatibility

### Naming Conventions
- **Scripts**: `verb_noun.py` (e.g., `convert_masks_to_yolo.py`)
- **Functions**: `snake_case` with descriptive names
- **Variables**: `lower_case` for locals, `UPPER_CASE` for constants
- **Model Names**: `polyp_yolov8{variant}_{training_suffix}`

### Performance Considerations
- **Training**: GPU strongly recommended (CPU training very slow)
- **Batch Size**: Start with 16, reduce if memory issues
- **Image Size**: 640px default, reduce to 512/416 for speed
- **Multi-component**: Use `--multi` flag only if masks contain multiple separate polyps

### Data Conventions
- **Class ID**: Always `0` for polyp (single-class detection)
- **Coordinates**: YOLO normalized format `[class_id, x_center, y_center, width, height]`
- **File Structure**: Maintain parallel `images/` and `labels/` directories
- **Splits**: Default 80/20 train/val, seed=42 for reproducibility

### Common Commands
```bash
# Full pipeline from scratch
python scripts/convert_masks_to_yolo.py \
  --images data/archive/Kvasir-SEG/Kvasir-SEG/images \
  --masks  data/archive/Kvasir-SEG/Kvasir-SEG/masks \
  --labels_out data/processed/labels --multi
python scripts/split_train_val.py \
  --images data/archive/Kvasir-SEG/Kvasir-SEG/images \
  --labels data/processed/labels --out data/processed
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=25 imgsz=640 batch=16 name=polyp_yolov8n

# Quick demo training (CPU-friendly)
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=3 imgsz=512 batch=4 name=polyp_demo

# Evaluation
python scripts/eval_val.py --weights models/polyp_yolov8n_clean/weights/best.pt

# Phase 2 — run all test videos through temporal pipeline
for f in data/test-set/videos/*.mpg; do
  python pipeline/main_pipeline.py --weights models/polyp_yolov8n_clean/weights/best.pt --video "$f" --conf 0.5
done

# Video synthesis for testing (requires ffmpeg)
ffmpeg -framerate 30 -pattern_type glob -i 'data/archive/Kvasir-SEG/Kvasir-SEG/images/*.jpg' -vf "scale=640:480" -c:v libx264 -pix_fmt yuv420p sample_video.mp4
```

### Troubleshooting
- **KeyboardInterrupt during training**: Partial weights may be saved in `runs/detect/train*/weights/`
- **Memory issues**: Reduce batch size and/or image size
- **No detections**: Check if model path exists, verify class confidence threshold
- **Multi-component issues**: Inspect mask connectivity with `cv2.findContours()`

### Expected Outcomes
- **Training**: Target mAP@50 > 0.7 for good polyp detection (achieved: 89.4%)
- **Inference Speed**: ~30-60 FPS on modern GPUs for 640px images
- **Dataset**: 1000 images from Kvasir-SEG, 800 train / 200 val after split

## File Organization
```
├── .github/
│   └── copilot-instructions.md  # This file — auto-loaded by GitHub Copilot
├── scripts/              # Core Phase 1 processing pipeline
├── pipeline/             # Phase 2 temporal CADe pipeline (SORT + smoothing)
├── data/
│   ├── processed/        # LOCAL ONLY — generated YOLO-format data (gitignored)
│   ├── archive/          # LOCAL ONLY — raw Kvasir-SEG dataset (gitignored)
│   └── test-set/         # TRACKED — GIANA challenge colonoscopy videos
├── models/
│   └── polyp_yolov8n_clean/  # TRACKED — production model weights (best.pt)
├── results/              # TRACKED — detection CSVs only (mp4s excluded from remote)
├── copilot/              # LOCAL ONLY — gitignored; personal Copilot session prompts
└── yolo_data.yaml        # YOLO training dataset configuration
```

### Dependencies
Primary: `ultralytics`, `opencv-python-headless`, `torch`, `filterpy`
Secondary: `pandas`, `tqdm`, `albumentations`, `pycocotools`, `scipy`
See `requirements.txt` for complete list.

---
*YOLO v8 | Single-class polyp detection | Phase 2: SORT temporal CADe*