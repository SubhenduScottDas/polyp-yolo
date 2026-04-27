# 📊 Repository Size Analysis

**Last Updated**: April 2026

## Overview

This document provides a detailed breakdown of the polyp-yono repository size, explaining what's included in the GitHub repository versus what remains local-only during development.

## 📈 Size Comparison Summary

| Scenario | Size | Description |
|----------|------|-------------|
| **🏠 Local Development** | **~670 MB** | Complete development environment with training data |
| **🌐 GitHub Clone** | **~45 MB** | Production-ready package for end users |
| **💾 Git Repository** | **~55 MB** | Git history and version control data |

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
- **`yolov8n.pt` (6.2 MB)**: Base YOLOv8n weights (AGPL-3.0; auto-downloaded by `ultralytics` CLI — not stored on remote)

#### Shared Components (✅ IN Git)
- **`data/test-set/` (53 MB)**: Test videos and validation data
- **`results/` (34 MB)**: Comprehensive test results
- **`models/polyp_yolov8n_clean/` (6 MB)**: Production model
- **Core files** (<10 MB): Scripts, configs, documentation

---

## 🌐 GitHub Clone (~45 MB)

### What Users Download
A complete, production-ready research package containing:

#### 🎯 Core Components
- **Trained Model** (`models/polyp_yolov8n_clean/`): **6 MB**
  - `best.pt`: Production weights (89.4% mAP@50)
  - `args.yaml`: Training configuration
  - `results.csv`: Performance metrics

#### 🧪 Test Data & Validation
- **Medical Test Videos** (`data/test-set/videos/`): **35 MB**
  - 7 endoscopy videos (.mpg format) — GIANA challenge dataset
  - Multiple polyp morphologies (MSD, Pedunculated, Ileocecal valve, Rectal carpet)
- **Test Image Frames** (`data/test-set/`): included
  - Sequential and non-sequential test cases

#### 📊 Results & Detection Data
- **Phase 1 Detection CSVs** (`results/phase_1/*_phase1.csv`): **~1 MB**
  - Frame-by-frame detection data with confidence scores
  - All 7 test videos logged
- **Phase 2 SORT CSVs** (`results/phase_2_sort/logs/*_tracked.csv` + `*.json`): **~1 MB**
  - SORT-tracked output with `conf_raw`, `conf_smooth`, `recovered` columns + tracking log JSON + metrics JSON
  - All 7 test videos logged
- **Phase 2 DeepSORT CSVs** (`results/phase_2_deepsort/logs/*_deepsort.csv` + `*.json`): **~1 MB**
  - DeepSORT-tracked output (appearance + IoU) with same columns + tracking log JSON + metrics JSON
  - All 7 test videos logged
- **Documentation & Scripts** (<1 MB): Complete pipeline code

> ℹ️ **Annotated `.mp4` videos are excluded from remote** (regeneratable; saves ~110 MB on clone size).
> Regenerate with `scripts/video_infer_yolo.py` (Phase 1) or `pipeline/main_pipeline.py` (Phase 2).

---

## 🔧 Technical Details

### Git Repository Structure
```bash
# What's tracked in Git (~45 MB)
├── models/polyp_yolov8n_clean/     # ✅ Production model (6 MB)
│   ├── weights/best.pt             #    Best weights only
│   ├── args.yaml                   #    Training config
│   └── results.csv                 #    Metrics log
├── data/test-set/                  # ✅ Test videos & frames (35 MB)
├── results/phase_1/*.csv           # ✅ Phase 1 detection CSVs (~1 MB)
├── results/phase_2_sort/logs/      # ✅ Phase 2 SORT tracking CSVs + JSON (~1 MB)
├── results/phase_2_sort/metrics/   # ✅ Phase 2 SORT metrics JSON
├── results/phase_2_deepsort/logs/  # ✅ Phase 2 DeepSORT tracking CSVs + JSON (~1 MB)
├── results/phase_2_deepsort/metrics/ # ✅ Phase 2 DeepSORT metrics JSON
├── results/sample_inference/       # ✅ Example outputs
├── scripts/                        # ✅ Core Phase 1 pipeline
├── pipeline/                       # ✅ Phase 2 temporal CADe pipeline (SORT + DeepSORT)
├── .github/copilot-instructions.md # ✅ Copilot dev context
└── docs & configs                  # ✅ Documentation & configuration

# What's local-only (~625 MB)
├── data/archive/                   # ❌ Kvasir-SEG dataset (162 MB)
├── data/processed/                 # ❌ Generated training data (76 MB)
├── models/ (except clean)          # ❌ Experimental model variants
├── runs/                           # ❌ Ultralytics training logs (22 MB)
├── results/phase_1/*.mp4           # ❌ Phase 1 annotated videos (~90 MB)
├── results/phase_2_sort/videos/    # ❌ Phase 2 SORT annotated videos (~20 MB)
├── results/phase_2_deepsort/videos/ # ❌ Phase 2 DeepSORT annotated videos (~20 MB)
├── yolov8n.pt                      # ❌ Base YOLOv8n weights (6.2 MB, AGPL-3.0)
├── test_output/                    # ❌ Temporary inference results (14 MB)
├── copilot/                        # ❌ Personal Copilot session prompts
└── thesis-prep-docs/               # ❌ Thesis LaTeX and drafts
```

### GitIgnore Strategy
The `.gitignore` uses selective inclusion (exclude-all, then explicitly allow):

```gitignore
# Base YOLO weights — auto-downloaded by ultralytics CLI, not stored on remote
yolov8n.pt

# Training data — local only
data/*
!data/test-set/        # Allow test videos and frames

# Models — exclude all EXCEPT production
models/
!models/polyp_yolov8n_clean/
!models/polyp_yolov8n_clean/weights/
!models/polyp_yolov8n_clean/weights/best.pt
!models/polyp_yolov8n_clean/args.yaml
!models/polyp_yolov8n_clean/results.csv

# Results — CSVs/JSON only; .mp4 files are regeneratable
results/*
!results/demo_detections.csv
!results/sample_inference/

!results/phase_1/
results/phase_1/*.mp4          # Exclude Phase 1 annotated videos
!results/phase_1/*.csv         # Allow Phase 1 detection CSVs

!results/phase_2_sort/
!results/phase_2_sort/logs/
results/phase_2_sort/videos/   # Exclude Phase 2 SORT annotated videos
!results/phase_2_sort/logs/*.csv
!results/phase_2_sort/logs/*.json
!results/phase_2_sort/metrics/

!results/phase_2_deepsort/
!results/phase_2_deepsort/logs/
results/phase_2_deepsort/videos/  # Exclude Phase 2 DeepSORT annotated videos
!results/phase_2_deepsort/logs/*.csv
!results/phase_2_deepsort/logs/*.json
!results/phase_2_deepsort/metrics/

# Always excluded
runs/          # Ultralytics training logs
copilot/       # Personal prompt files
thesis-prep-docs/
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
| Nov 13, 2025 | Initial setup | ~1 MB | Scripts and configs only |
| Nov 13, 2025 | Production model added | ~7 MB | `polyp_yolov8n_clean/weights/best.pt` included |
| Nov 14, 2025 | Test data & results added | ~97 MB | Test videos + annotated `.mp4` results included |
| Apr 2026 | Phase 2 pipeline added | ~97 MB | `pipeline/` package; Phase 2 SORT CSVs tracked |
| Apr 2026 | Phase 2 DeepSORT added | **~47 MB** | DeepSORT tracker + restructured results (`phase_1/`, `phase_2_sort/`, `phase_2_deepsort/`) |
| Apr 2026 | Remote cleanup | **~45 MB** | Removed `yolov8n.pt` (AGPL-3.0) + 12 annotated `.mp4` videos (~55 MB saved) |

---

**💡 Pro Tip**: This size analysis demonstrates best practices for machine learning repositories - providing complete functionality while maintaining reasonable download sizes through smart data management strategies.

---

© 2026 Subhendu Das