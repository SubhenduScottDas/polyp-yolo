# Image Assets - Draft Thesis

## ✅ All Images Self-Contained

All images referenced in the LaTeX files are now located within the `draft-thesis/` folder structure. You can upload this folder to Overleaf without any external dependencies.

---

## 📁 Image Inventory

### Header (Logo)
- `Figure/header/IIITK.png` ✓ (Copied from template)

### Chapter 1 - Introduction
- `Figure/chp1/polyp_detection_overview.png` ✓ (Placeholder - replace with custom diagram)

### Chapter 3 - Methodology
- `Figure/chp3/system_architecture.png` ✓ (Placeholder - replace with custom diagram)
- `Figure/chp3/mask_to_bbox.png` ✓ (Placeholder - replace with custom diagram)
- `Figure/chp3/augmentation_examples.png` ✓ (Placeholder - replace with custom diagram)

### Chapter 5 - Results and Analysis
**Real Training Results (from `models/polyp_yolov8n_clean/`):**
- `Figure/chp5/results.png` ✓ (Training metrics over 50 epochs)
- `Figure/chp5/BoxPR_curve.png` ✓ (Precision-Recall curve)
- `Figure/chp5/BoxP_curve.png` ✓ (Precision curve)
- `Figure/chp5/BoxR_curve.png` ✓ (Recall curve - extra, not referenced)
- `Figure/chp5/BoxF1_curve.png` ✓ (F1 score curve - extra, not referenced)
- `Figure/chp5/confusion_matrix.png` ✓ (Non-normalized confusion matrix - extra)
- `Figure/chp5/confusion_matrix_normalized.png` ✓ (Normalized confusion matrix)
- `Figure/chp5/val_batch0_pred.jpg` ✓ (Validation predictions batch 0)
- `Figure/chp5/val_batch1_pred.jpg` ✓ (Validation predictions batch 1)

---

## 🎨 Placeholder Images

The following are **placeholder images** that you should replace with actual diagrams:

1. **`Figure/chp1/polyp_detection_overview.png`**
   - Should show: High-level system overview (Input → Detection → Output)
   - Suggestion: Create using draw.io, PowerPoint, or Inkscape

2. **`Figure/chp3/system_architecture.png`**
   - Should show: Complete pipeline architecture
   - Include: Dataset → Preprocessing → YOLOv8 → Training → Inference → Evaluation
   
3. **`Figure/chp3/mask_to_bbox.png`**
   - Should show: Visual example of mask-to-bounding box conversion
   - Include: Original image + Mask + Resulting bbox overlaid

4. **`Figure/chp3/augmentation_examples.png`**
   - Should show: Grid of augmented images
   - Include: Original, Mosaic, Mixup, HSV adjusted, Flipped versions

---

## 🔧 How to Replace Placeholders

### Option 1: Create Diagrams Locally
1. Use tools like:
   - **draw.io** (https://draw.io) - Free diagram tool
   - **PowerPoint** - Export as PNG (300 DPI)
   - **Inkscape** - Vector graphics (export as PNG)
   - **Python matplotlib/seaborn** - Programmatic figures

2. Save with exact filenames as listed above
3. Replace in `draft-thesis/Figure/chp*/` folders
4. Re-upload to Overleaf (or just replace the specific files)

### Option 2: Create Diagrams in Overleaf
1. Use TikZ code directly in LaTeX:
```latex
\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}
    % Your diagram code here
  \end{tikzpicture}
  \caption{Your caption}
  \label{fig:label}
\end{figure}
```

2. Or use LaTeX drawing packages:
   - `tikz` for technical diagrams
   - `pgfplots` for charts
   - `forest` for tree diagrams

### Option 3: Use Repository Test Images
You can create these diagrams from existing repository data:

```python
# Example: Create mask_to_bbox visualization
import cv2
import matplotlib.pyplot as plt

# Load sample from Kvasir-SEG
img = cv2.imread('data/archive/Kvasir-SEG/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg')
mask = cv2.imread('data/archive/Kvasir-SEG/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg', 0)

# Create 3-panel figure showing: original → mask → bbox
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# ... add visualization code ...
plt.savefig('thesis-prep-docs/draft-thesis/Figure/chp3/mask_to_bbox.png', dpi=300)
```

---

## ✓ Compilation Status

**All images present** - LaTeX will compile without missing figure errors!

- Placeholders will show white background with text labels
- Replace them before final submission for professional appearance
- Actual training results (Chapter 5) are already high-quality and ready

---

## 📦 Overleaf Upload Checklist

When uploading to Overleaf:
- ✅ Upload entire `draft-thesis/` folder maintaining structure
- ✅ Set `Polyp_Detection_Thesis.tex` as main document
- ✅ Verify all `.tex`, `.bib`, `.sty`, `.cls` files are present
- ✅ Check that `Figure/` folder structure is preserved
- ✅ Compile with: pdflatex → bibtex → pdflatex → pdflatex

**No external file dependencies** - Everything is self-contained! 🎉

---

*Generated: November 16, 2025*
