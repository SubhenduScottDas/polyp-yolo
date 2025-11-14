# 📊 Repository Size Analysis

**Last Updated**: November 14, 2025

## Overview

This document provides a detailed breakdown of the polyp-yono repository size, explaining what's included in the GitHub repository versus what remains local-only during development.

## 📈 Size Comparison Summary

| Scenario | Size | Description |
|----------|------|-------------|
| **🏠 Local Development** | **666 MB** | Complete development environment with training data |
| **🌐 GitHub Clone** | **~97 MB** | Production-ready package for end users |
| **💾 Git Repository** | **241 MB** | Git history and version control data |

---

## 🏠 Local Repository (666 MB)

### What's Included Locally
Your complete development environment contains everything needed for research and development:

#### Training Data (Local Only - NOT in Git)
- **`data/archive/` (162 MB)**: Original Kvasir-SEG dataset
  - 1000 endoscopy images (.jpg)
  - 1000 segmentation masks (.jpg)
- **`data/processed/` (76 MB)**: Generated YOLO training data
  - Train/validation image symlinks
  - YOLO format label files (.txt)

#### Training Outputs (Local Only - NOT in Git)
- **`models/polyp_yolov8n_quick/` (36 MB)**: Quick training experiment
- **`runs/` (22 MB)**: Ultralytics training session logs
- **`test_output/` (14 MB)**: Temporary inference results
- **`models/polyp_yolov8n/` (1.2 MB)**: Additional model variants

#### Shared Components (✅ IN Git)
- **`data/test-set/` (53 MB)**: Test videos and validation data
- **`results/` (34 MB)**: Comprehensive test results
- **`models/polyp_yolov8n_clean/` (6 MB)**: Production model
- **Core files** (<10 MB): Scripts, configs, documentation

---

## 🌐 GitHub Clone (97 MB)

### What Users Download
A complete, production-ready research package containing:

#### 🎯 Core Components
- **Trained Model** (`models/polyp_yolov8n_clean/`): **6 MB**
  - `best.pt`: Production weights (89.4% mAP@50)
  - `args.yaml`: Training configuration
  - `results.csv`: Performance metrics

#### 🧪 Test Data & Validation
- **Medical Test Videos** (`data/test-set/videos/`): **35 MB**
  - 7 endoscopy videos (.mpv format)
  - Multiple polyp morphologies (MSD, Pedunculated, Ileocecal valve)
- **Test Image Frames** (`data/test-set/frames/`): **18 MB**
  - Sequential and non-sequential test cases
  - Positive and occluded polyp examples

#### 📊 Results & Documentation
- **Test Results** (`results/`): **34 MB**
  - Detection CSV files with frame-by-frame analysis
  - Annotated videos showing bounding boxes
  - Performance validation across polyp types
- **Pre-trained Weights** (`yolov8n.pt`): **6.2 MB**
- **Documentation & Scripts** (<1 MB): Complete pipeline code

---

## 🔧 Technical Details

### Git Repository Structure
```bash
# What's tracked in Git (97 MB)
├── models/polyp_yolov8n_clean/     # ✅ Production model (6 MB)
├── data/test-set/                  # ✅ Test videos & frames (53 MB)  
├── results/                        # ✅ Test results & annotations (34 MB)
├── scripts/                        # ✅ Core pipeline scripts
├── yolov8n.pt                     # ✅ Pre-trained weights (6.2 MB)
└── docs & configs                  # ✅ Documentation & configuration

# What's local-only (569 MB)
├── data/archive/                   # ❌ Kvasir-SEG dataset (162 MB)
├── data/processed/                 # ❌ Generated training data (76 MB)
├── models/polyp_yolov8n_quick/     # ❌ Experimental models (36 MB)
├── runs/                          # ❌ Training logs (22 MB)
└── test_output/                   # ❌ Temporary results (14 MB)
```

### GitIgnore Strategy
The repository uses a sophisticated `.gitignore` configuration:

```gitignore
# Training data - local only
data/archive/          # Original dataset
data/processed/        # Generated YOLO data

# Training outputs - local only  
runs/                  # Ultralytics training logs
models/                # All models EXCEPT production

# Test data - included for reproducibility
!data/test-set/        # Test videos and frames
!results/              # Test results and annotations
!models/polyp_yolov8n_clean/  # Production model
```

---

## 🎯 Benefits of This Approach

### ✅ For End Users
- **Quick Download**: 97 MB vs 666 MB (85% reduction)
- **Immediate Use**: Pre-trained model included
- **Complete Testing**: All test data and results provided
- **Full Reproducibility**: Can verify all documented results

### ✅ For Researchers  
- **Complete Package**: Everything needed for peer review
- **Validation Data**: Real medical test videos included
- **Performance Evidence**: Comprehensive test results
- **Easy Deployment**: Production-ready model weights

### ✅ For Developers
- **Clean Separation**: Training vs. production data
- **Flexible Development**: Full local environment
- **Version Control**: Only essential components tracked
- **Efficient Collaboration**: Reasonable repository size

---

## 📋 Quick Commands

### Check Local Size
```bash
du -sh .                    # Total local size
du -sh data/ models/ results/  # Component breakdown
```

### Check Git Size  
```bash
du -sh .git                 # Git repository size
git count-objects -vH       # Detailed git statistics
git ls-files | xargs du -ch | tail -1  # Tracked files size
```

### Repository Stats
```bash
# Files tracked in git
git ls-files | wc -l

# Total repository objects
git rev-list --all --count

# Repository compression ratio
git gc --aggressive
```

---

## 🔄 Size Evolution

| Date | Event | Clone Size | Notes |
|------|-------|------------|-------|
| Nov 14, 2025 | Added test data | 97 MB | Complete research package |
| Nov 13, 2025 | Production model | 42 MB | Model weights included |
| Nov 13, 2025 | Initial setup | 35 MB | Scripts and configs only |

---

**💡 Pro Tip**: This size analysis demonstrates best practices for machine learning repositories - providing complete functionality while maintaining reasonable download sizes through smart data management strategies.