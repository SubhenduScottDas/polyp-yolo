# GitHub Copilot Instructions - Polyp YOLO Detection

## Project Overview
This repository implements a **YOLO-based polyp detection system** for medical imaging. The project converts segmentation masks from the Kvasir-SEG dataset into YOLO bounding box format and provides a complete pipeline for training, inference, and evaluation.

### Architecture
```
Data Flow: Segmentation Masks → YOLO Labels → Training → Inference (Images/Videos) → Evaluation
Scripts: convert_masks_to_yolo.py → split_train_val.py → [train] → infer_and_viz.py/video_infer_yolo.py → eval_val.py
```

## Core Components

### 1. Data Conversion (`scripts/convert_masks_to_yolo.py`)
- **Purpose**: Convert binary segmentation masks to YOLO bounding box format
- **Key Feature**: Supports multi-component mask detection with `--multi` flag
- **Usage**: 
  ```bash
  python scripts/convert_masks_to_yolo.py --input_dir data/archive/Kvasir-SEG/Kvasir-SEG --output_dir data/processed
  python scripts/convert_masks_to_yolo.py --input_dir data/archive/Kvasir-SEG/Kvasir-SEG --output_dir data/processed --multi
  ```
- **Multi-component Logic**: Uses `cv2.findContours()` to detect separate connected components in masks and creates individual bounding boxes for each

### 2. Dataset Splitting (`scripts/split_train_val.py`)
- **Purpose**: Random train/validation split (80/20 default)
- **Output**: Creates `data/processed/images/{train,val}/` structure
- **Usage**: `python scripts/split_train_val.py`

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

#### Video Inference (`scripts/video_infer_yolo.py`)
- **Purpose**: Frame-by-frame video processing
- **Features**: 
  - Annotated video output
  - Optional CSV logging (`frame,class_id,class_name,conf,x1,y1,x2,y2`)
- **Usage**: 
  ```bash
  python scripts/video_infer_yolo.py --video input.mp4 --model models/polyp_yolov8n/weights/best.pt --output annotated_video.mp4 --csv detections.csv
  ```

### 5. Evaluation (`scripts/eval_val.py`)
- **Purpose**: Run YOLO validation metrics on test set
- **Metrics**: mAP@50, mAP@50-95, precision, recall
- **Usage**: `python scripts/eval_val.py --model models/polyp_yolov8n/weights/best.pt`

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
python scripts/convert_masks_to_yolo.py --input_dir data/archive/Kvasir-SEG/Kvasir-SEG --output_dir data/processed --multi
python scripts/split_train_val.py
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=25 imgsz=640 batch=16 name=polyp_yolov8n

# Quick demo training (CPU-friendly)
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=3 imgsz=512 batch=4 name=polyp_demo

# Evaluation
python scripts/eval_val.py --model models/polyp_yolov8n/weights/best.pt

# Video synthesis for testing (requires ffmpeg)
ffmpeg -framerate 30 -pattern_type glob -i 'data/archive/Kvasir-SEG/Kvasir-SEG/images/*.jpg' -vf "scale=640:480" -c:v libx264 -pix_fmt yuv420p sample_video.mp4
```

### Troubleshooting
- **KeyboardInterrupt during training**: Partial weights may be saved in `runs/detect/train*/weights/`
- **Memory issues**: Reduce batch size and/or image size
- **No detections**: Check if model path exists, verify class confidence threshold
- **Multi-component issues**: Inspect mask connectivity with `cv2.findContours()`

### Expected Outcomes
- **Training**: Target mAP@50 > 0.7 for good polyp detection
- **Inference Speed**: ~30-60 FPS on modern GPUs for 640px images
- **Dataset**: 1000 images from Kvasir-SEG, typically 800 train / 200 val after split

## File Organization
```
├── scripts/              # Core processing pipeline
├── data/
│   ├── processed/        # YOLO-format data (tracked)
│   └── archive/          # Raw Kvasir-SEG dataset (gitignored)
├── models/               # Training outputs (gitignored)
├── results/              # Inference outputs (gitignored)
├── yolo_data.yaml        # YOLO training configuration
└── .github/
    └── copilot-instructions.md  # This file
```

### Dependencies
Primary: `ultralytics`, `opencv-python-headless`, `torch`
Secondary: `pandas`, `tqdm`, `albumentations`, `pycocotools`
See `requirements.txt` for complete list.

---
*Generated on November 14, 2025 | YOLO v8 | Single-class polyp detection*