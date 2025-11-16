# Statistical Analysis and Visualization Scripts

This folder contains Python scripts for generating thesis figures, statistical analysis, and visualizations for the Polyp Detection project.

## Scripts

### 1. `generate_thesis_figures.py`

Generates professional diagrams and visualizations for the M.Tech thesis on Deep Learning-Based Polyp Detection Using YOLOv8.

**Features:**
- System overview diagram (Chapter 1)
- Complete system architecture diagram (Chapter 3)
- Mask-to-bounding box conversion visualization using actual Kvasir-SEG data
- Data augmentation examples with 8 different techniques

**Requirements:**
```bash
pip install opencv-python-headless matplotlib numpy
```

**Usage:**

Generate all figures (default):
```bash
python generate_thesis_figures.py
```

Generate specific figures:
```bash
# Only system overview
python generate_thesis_figures.py --figures overview

# Only architecture diagram
python generate_thesis_figures.py --figures architecture

# Multiple specific figures
python generate_thesis_figures.py --figures overview architecture mask2bbox
```

Custom paths:
```bash
python generate_thesis_figures.py \
    --output-dir /path/to/output \
    --kvasir-dir /path/to/kvasir-seg
```

**Arguments:**
- `--output-dir`: Output directory for generated figures (default: `../../thesis-prep-docs/draft-thesis/Figure`)
- `--kvasir-dir`: Path to Kvasir-SEG dataset directory (default: `../../data/archive/Kvasir-SEG/Kvasir-SEG`)
- `--figures`: Which figures to generate (choices: `overview`, `architecture`, `mask2bbox`, `augmentation`, `all`)

**Output:**
- `Figure/chp1/polyp_detection_overview.png` - 4-stage pipeline diagram
- `Figure/chp3/system_architecture.png` - 5-layer system architecture
- `Figure/chp3/mask_to_bbox.png` - Mask to bbox conversion visualization
- `Figure/chp3/augmentation_examples.png` - 8 augmentation techniques

All figures generated at **300 DPI** for publication quality.

---

## Adding New Analysis Scripts

When adding new statistical analysis or visualization scripts:

1. **Follow naming convention**: `<action>_<subject>.py` (e.g., `analyze_video_detections.py`)
2. **Include docstrings**: Module-level and function-level documentation
3. **Add CLI arguments**: Use `argparse` for flexibility
4. **Update this README**: Document usage and requirements
5. **Use relative paths**: Make scripts portable across different setups

### Example Template:

```python
#!/usr/bin/env python3
"""
Script description here.

Author: Subhendu Das
Date: November 2024
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.png')
    args = parser.parse_args()
    
    # Your code here

if __name__ == '__main__':
    main()
```

---

## Future Scripts (Planned)

- `analyze_detection_confidence.py` - Confidence score distribution analysis
- `compare_model_performance.py` - Compare different YOLO variants
- `visualize_training_metrics.py` - Enhanced training curve visualization
- `generate_confusion_matrix.py` - Custom confusion matrix with annotations
- `analyze_video_statistics.py` - Frame-by-frame detection statistics
- `create_performance_tables.py` - LaTeX table generation for results

---

## Project Structure

```
scripts/
├── statistical-analysis/
│   ├── README.md                      # This file
│   ├── generate_thesis_figures.py     # Main figure generation script
│   └── [future analysis scripts]
├── convert_masks_to_yolo.py
├── split_train_val.py
├── infer_and_viz.py
├── video_infer_yolo.py
└── eval_val.py
```

---

## Reproducibility

All scripts use:
- **Fixed random seeds** (where applicable) for reproducibility
- **Relative paths** from script location
- **Configurable parameters** via command-line arguments
- **Clear documentation** for all functions

To reproduce all thesis figures:
```bash
cd scripts/statistical-analysis
python generate_thesis_figures.py --figures all
```

---

## License

Part of the Polyp Detection YOLOv8 project.  
© 2024 Subhendu Das, IIIT Kalyani
