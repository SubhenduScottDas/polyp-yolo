# 🎯 Training Guide

**Last Updated**: April 2026

## Overview

This document provides comprehensive guidance for training YOLO models on the Kvasir-SEG polyp dataset, from data preparation to model optimization.

## 🚀 Quick Training

### Prerequisites
```bash
# Ensure conda environment is active
conda activate polypbench

# Verify GPU availability (recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### One-Command Training
```bash
# Complete pipeline from scratch
python scripts/convert_masks_to_yolo.py \
  --images     data/archive/Kvasir-SEG/Kvasir-SEG/images \
  --masks      data/archive/Kvasir-SEG/Kvasir-SEG/masks \
  --labels_out data/processed/labels \
  --multi
python scripts/split_train_val.py \
  --images data/archive/Kvasir-SEG/Kvasir-SEG/images \
  --labels data/processed/labels \
  --out    data/processed
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=50 imgsz=640 batch=16 name=polyp_yolov8n
```

## 📊 Training Configuration

### Model Variants
| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| YOLOv8n | 3.2M | Fastest | Good | Development/Testing |
| YOLOv8s | 11.2M | Fast | Better | Balanced Performance |
| YOLOv8m | 25.9M | Medium | High | Production |
| YOLOv8l | 43.7M | Slow | Higher | High Accuracy Needed |

### Recommended Settings
```yaml
# For Development (Fast Iteration)
model: yolov8n.pt
epochs: 25
imgsz: 512
batch: 8

# For Production (Best Results) 
model: yolov8m.pt
epochs: 100
imgsz: 640
batch: 16
```

## 🔧 Advanced Training Options

### Hyperparameter Tuning
```bash
# Automatic hyperparameter search
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=50 tune=True

# Custom hyperparameters
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml \
  lr0=0.01 lrf=0.1 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3 warmup_momentum=0.8 box=7.5 cls=0.5 dfl=1.5
```

### Data Augmentation
```bash
# Enhanced augmentation for better generalization
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=10.0 translate=0.1 scale=0.5 shear=2.0 \
  flipud=0.5 fliplr=0.5 mosaic=1.0 mixup=0.15
```

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# View training in browser (if using Weights & Biases)
wandb login
export WANDB_PROJECT="polyp-detection"

# TensorBoard monitoring
tensorboard --logdir runs/detect/
```

### Training Metrics to Watch
- **mAP@50**: Primary metric (target: >0.85)
- **mAP@50-95**: Comprehensive accuracy (target: >0.65)
- **Box Loss**: Bounding box regression (should decrease)
- **Class Loss**: Classification accuracy (should decrease)
- **DFL Loss**: Distribution focal loss (should decrease)

## 🎛️ Training Strategies

### Progressive Training
```bash
# Stage 1: Quick validation (3 epochs)
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=3 name=validation

# Stage 2: Medium training (25 epochs)
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=25 name=medium

# Stage 3: Full training (100 epochs)
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml epochs=100 name=production
```

### Multi-GPU Training
```bash
# Distribute training across multiple GPUs
yolo task=detect mode=train model=yolov8n.pt data=yolo_data.yaml \
  epochs=50 batch=32 device=0,1,2,3
```

## 🔍 Training Troubleshooting

### Common Issues & Solutions

#### Low mAP Performance
```bash
# Increase training epochs
epochs=100

# Larger image size
imgsz=832

# Better data augmentation
mosaic=1.0 mixup=0.2
```

#### Memory Issues
```bash
# Reduce batch size
batch=8

# Smaller image size  
imgsz=416

# Use gradient accumulation
batch=4 accumulate=4  # Effective batch size = 16
```

#### Training Instability
```bash
# Lower learning rate
lr0=0.001

# Increase warmup
warmup_epochs=5

# Use mixed precision
amp=True
```

## 📊 Expected Training Results

### Baseline Performance (50 epochs)
```
Model: YOLOv8n
Dataset: Kvasir-SEG (800 train, 200 val)
Results:
├── mAP@50: 89.4%
├── mAP@50-95: 70.7%
├── Precision: 82.8%
├── Recall: 86.4%
└── Training Time: ~7.15 hours (CPU)
```

### Training Curves
- **Epochs 1-10**: Rapid improvement, mAP@50 reaches ~60%
- **Epochs 10-30**: Steady improvement, mAP@50 reaches ~80%  
- **Epochs 30-50**: Fine-tuning, mAP@50 reaches ~89%
- **Epochs 50+**: Diminishing returns, risk of overfitting

## 🎯 Optimization Tips

### For Speed
- Use YOLOv8n model
- Set `imgsz=416` 
- Reduce `batch=4-8`
- Use `amp=True` (mixed precision)

### For Accuracy
- Use YOLOv8m or YOLOv8l model
- Set `imgsz=832`
- Increase `epochs=100+`
- Enable data augmentation

### For Memory Efficiency
- Use gradient accumulation: `batch=4 accumulate=4`
- Enable mixed precision: `amp=True`
- Reduce workers: `workers=2`

## 🔄 Resume Training

### From Checkpoint
```bash
# Resume interrupted training
yolo task=detect mode=train model=runs/detect/polyp_yolov8n/weights/last.pt resume=True

# Continue from specific epoch
yolo task=detect mode=train model=runs/detect/polyp_yolov8n/weights/epoch50.pt \
  data=yolo_data.yaml epochs=100
```

## 📝 Training Logs

### Important Files
```
runs/detect/polyp_yolov8n/
├── weights/
│   ├── best.pt          # Best performing model
│   ├── last.pt          # Latest checkpoint
│   └── epoch*.pt        # Epoch checkpoints
├── results.png          # Training curves
├── confusion_matrix.png # Classification performance
├── labels.jpg           # Label distribution
└── args.yaml           # Training arguments
```

### Post-Training Analysis
```bash
# Evaluate final model
python scripts/eval_val.py --weights runs/detect/polyp_yolov8n/weights/best.pt

# Test on sample videos
python scripts/video_infer_yolo.py \
  --video   data/test-set/videos/PolipoMSDz2.mpg \
  --weights runs/detect/polyp_yolov8n/weights/best.pt \
  --out     test_results.mp4
```

---

**💡 Pro Tip**: Start with a small number of epochs (3-5) to validate your data pipeline, then scale up to full training once you confirm everything works correctly.

## 🧠 Phase 2 — Temporal CADe Pipeline

After training, the Phase 2 temporal pipeline builds on top of the trained model without any retraining. It adds tracking, confidence smoothing, and gap recovery. Two tracker backends are available:

- **SORT** (`--tracker sort`): Kalman filter + Hungarian algorithm, IoU-only association → `results/phase_2_sort/`
- **DeepSORT** (`--tracker deepsort`): appearance embeddings (PyTorch MobileNet) + IoU association → `results/phase_2_deepsort/`

See `pipeline/main_pipeline.py` and the [Phase 2 section in README.md](README.md#what-was-built).

---

© 2025 Subhendu Das