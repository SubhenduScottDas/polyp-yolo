# M.Tech Thesis: Deep Learning Based Polyp Detection Using YOLOv8

**IIIT Kalyani Executive M.Tech Thesis (2023-2025)**

This folder contains the complete LaTeX source for the M.Tech thesis on deep learning-based polyp detection using YOLOv8.

---

## 📂 Folder Structure

```
draft-thesis/
├── Polyp_Detection_Thesis.tex    # Main thesis document (compile this)
├── header.tex                     # Title pages, certificates, TOC
├── Abbreviation.tex               # List of abbreviations
├── ch-abstract.tex                # Abstract (1 page)
├── ch-acknowledgement.tex         # Acknowledgements
├── ch-certificate_self.tex        # Self-declaration certificate
├── ch-certificate_sup.tex         # Supervisor certificate
├── ch-mypub.tex                   # Publications list
├── Reference/
│   └── Bibliography.bib           # All cited references (BibTeX)
├── chp1/
│   ├── chp1.tex                   # Chapter 1: Introduction
│   └── Figure/                    # Chapter 1 figures
├── chp2/
│   ├── chp2.tex                   # Chapter 2: Literature Review
│   └── Figure/                    # Chapter 2 figures
├── chp3/
│   ├── chp3.tex                   # Chapter 3: Methodology
│   └── Figure/                    # Chapter 3 figures
├── chp4/
│   ├── chp4.tex                   # Chapter 4: Implementation
│   └── Figure/                    # Chapter 4 figures
├── chp5/
│   ├── chp5.tex                   # Chapter 5: Results and Analysis
│   └── Figure/
│       ├── BoxF1_curve.png        # Training metrics
│       ├── BoxPR_curve.png
│       ├── confusion_matrix.png
│       ├── results.png
│       ├── train_batch*.jpg
│       └── val_batch*_pred.jpg
└── chp6/
    ├── chp6.tex                   # Chapter 6: Conclusion and Future Work
    └── Figure/                    # Chapter 6 figures
```

---

## 🚀 Quick Start

### Prerequisites
- **LaTeX Distribution**: TeXLive (2022+) or MikTeX
- **Required Packages**: 
  - graphicx, hyperref, amsmath, algorithm2e, listings, tikz
  - geometry, fancyhdr, setspace, caption, subcaption
  - multirow, booktabs, xcolor, url, acronym

### Compilation Steps

#### ⚠️ IMPORTANT: Full Compilation Sequence for Bibliography

**You MUST run this complete sequence to get bibliography and all cross-references working:**

```bash
pdflatex Polyp_Detection_Thesis.tex
bibtex Polyp_Detection_Thesis
pdflatex Polyp_Detection_Thesis.tex
pdflatex Polyp_Detection_Thesis.tex
```

**Why 4 steps?**
1. **1st pdflatex**: Generates .aux file with citation keys and reference markers
2. **bibtex**: Reads Bibliography.bib (60+ entries) and creates formatted bibliography
3. **2nd pdflatex**: Incorporates bibliography and updates references
4. **3rd pdflatex**: Resolves all cross-references, page numbers, and TOC entries

**⚠️ Skipping any step will result in:**
- Missing bibliography (blank References section)
- Undefined citation warnings ([?] in text)
- Incorrect page numbers in Table of Contents
- Missing entries in List of Figures/Tables

#### Option 1: Using TeXShop/TeXworks/TeXmaker (GUI)
1. Open `Polyp_Detection_Thesis.tex`
2. Set compiler to **pdflatex**
3. Compile sequence:
   ```
   pdflatex → bibtex → pdflatex → pdflatex
   ```

#### Option 2: Using Command Line (macOS/Linux)
```bash
cd /path/to/draft-thesis

# Full compilation
pdflatex Polyp_Detection_Thesis.tex
bibtex Polyp_Detection_Thesis
pdflatex Polyp_Detection_Thesis.tex
pdflatex Polyp_Detection_Thesis.tex

# Or use latexmk (automated)
latexmk -pdf Polyp_Detection_Thesis.tex
```

#### Option 3: Using Overleaf (Online)
1. Create new blank project in Overleaf
2. Upload all files maintaining folder structure
3. Set main document to `Polyp_Detection_Thesis.tex`
4. Click **Recompile**

---

## ✏️ Customization Guide

### Student Information
Edit `header.tex` lines 10-20:
```latex
% Replace these with your details
\newcommand{\thesisTitle}{Deep Learning Based Polyp Detection Using YOLOv8}
\newcommand{\studentName}{Your Full Name}
\newcommand{\regNumber}{Your Registration Number}
\newcommand{\submissionYear}{2025}
\newcommand{\supervisorName}{Dr. Supervisor Name}
\newcommand{\supervisorDesignation}{Professor, Dept. of CSE}
```

### Adding Your Publications
Edit `ch-mypub.tex` to list your conference/journal papers.

### Modifying Acknowledgements
Edit `ch-acknowledgement.tex` to personalize thank-you notes.

### Adding More Figures
Place images in appropriate `chp*/Figure/` folders and reference using:
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{chp5/Figure/your_image.png}
  \caption{Your caption here.}
  \label{fig:your_label}
\end{figure}
```

### Adding Citations
Add new entries to `Reference/Bibliography.bib` following BibTeX format:
```bibtex
@article{author2024title,
  title={Article Title},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2024}
}
```
Cite in text using `\cite{author2024title}`.

---

## 📊 Chapter Overview

### Chapter 1: Introduction (18 pages)
- Background on colorectal cancer and polyp detection
- Motivation for AI-based CAD systems
- Problem statement and research objectives
- Thesis organization

### Chapter 2: Literature Review (22 pages)
- Traditional computer vision methods
- Deep learning evolution (CNN, segmentation, object detection)
- YOLO architecture progression (v1 → v8)
- Critical analysis and research gaps

### Chapter 3: Methodology (24 pages)
- Kvasir-SEG dataset preparation
- Mask-to-bounding box conversion
- YOLOv8 architecture details
- Training configuration and optimization
- Inference and evaluation metrics

### Chapter 4: Implementation (28 pages)
- System architecture
- Python implementation with code listings:
  - `convert_masks_to_yolo.py`
  - `split_train_val.py`
  - `infer_and_viz.py`
  - `video_infer_yolo.py`
  - `eval_val.py`
- Deployment considerations

### Chapter 5: Results and Analysis (26 pages)
- Training results (89.4% mAP@50, 70.7% mAP@50-95)
- Validation performance metrics
- Visual detection examples
- Real-world video testing (7 colonoscopy videos)
- Error analysis and failure cases

### Chapter 6: Conclusion and Future Work (14 pages)
- Summary of contributions
- Key findings and clinical impact
- Limitations
- 17 categories of future research directions

**Total Estimated Pages: ~150-160 pages**

**Bibliography: 60+ References** - All properly cited academic papers from medical imaging, deep learning, and YOLO research

---

## 🛠️ Troubleshooting

### Common Errors

**1. Missing Package Errors**
```
! LaTeX Error: File `algorithm2e.sty' not found.
```
**Solution**: Install missing package via TeX package manager:
```bash
tlmgr install algorithm2e  # For TeXLive
```

**2. Bibliography Not Appearing or Showing [?]**
```
LaTeX Warning: Citation 'jha2020kvasir' undefined.
LaTeX Warning: There were undefined references.
```
**Solution**: This is NORMAL on first compilation. You MUST run the full sequence:
```bash
pdflatex Polyp_Detection_Thesis.tex    # 1st run - generates .aux
bibtex Polyp_Detection_Thesis          # Process bibliography
pdflatex Polyp_Detection_Thesis.tex    # 2nd run - incorporate bib
pdflatex Polyp_Detection_Thesis.tex    # 3rd run - resolve refs
```

After the 3rd pdflatex run, all citations should resolve and References section should contain 60+ entries.

**Verify bibliography is working:**
- Check if `Polyp_Detection_Thesis.bbl` file was created (after bibtex)
- References section should appear after Chapter 6
- Citations in text should show as [1], [2], etc. (not [?])

**3. Blank Pages Between Chapters**
This is **normal** and intentional for two-sided printing:
- LaTeX uses `\cleardoublepage` to start new chapters on odd (right-hand) pages
- Blank pages ensure chapters always begin on the right side
- This is standard for thesis/book formatting
- **Not an error** - leave as is for professional printing

**2. Bibliography Not Compiling**
```
LaTeX Warning: Citation 'jha2020kvasir' undefined.
```
**Solution**: Run bibtex after first pdflatex:
```bash
pdflatex Polyp_Detection_Thesis.tex
bibtex Polyp_Detection_Thesis    # Note: no .tex extension
pdflatex Polyp_Detection_Thesis.tex
pdflatex Polyp_Detection_Thesis.tex
```

**3. Missing Figure Errors**
```
! LaTeX Error: File `chp5/Figure/results.png' not found.
```
**Solution**: Ensure all referenced figures exist in correct folders. Check file extensions (`.png`, `.jpg`, `.pdf`).

**4. Overfull/Underfull hbox Warnings**
These are minor formatting warnings, usually safe to ignore. To fix:
- Adjust word breaks with `\hyphenation{}`
- Use `\sloppy` for paragraphs with long URLs

**5. Too Many Open Files (macOS)**
If compilation fails with file limit errors:
```bash
ulimit -n 2048
```

---

## 📝 Writing Tips

### LaTeX Best Practices
1. **Use labels for cross-references**:
   ```latex
   As shown in Figure~\ref{fig:architecture}...
   Equation~\eqref{eq:loss_function} represents...
   ```

2. **Use consistent units**:
   ```latex
   \SI{640}{\pixel} × \SI{640}{\pixel}  % Requires siunitx package
   ```

3. **Break long paragraphs**:
   - Keep paragraphs under 8-10 lines
   - Use `\par` or blank lines for separation

4. **Code listings**:
   ```latex
   \begin{lstlisting}[language=Python, caption={Script name}]
   # Your Python code here
   \end{lstlisting}
   ```

5. **Mathematical equations**:
   ```latex
   \begin{equation}
     \text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
     \label{eq:map}
   \end{equation}
   ```

### Academic Writing Guidelines
- **Tense**: Past tense for methodology, present for results discussion
- **Voice**: Primarily passive ("The model was trained...") but active is acceptable ("We trained...")
- **Formatting**: Use `\textit{}` for emphasis, `\textbf{}` for key terms (sparingly)
- **Abbreviations**: Define on first use, then add to `Abbreviation.tex`

---

## 🔧 Advanced Configuration

### Changing Bibliography Style
Edit line in `Polyp_Detection_Thesis.tex`:
```latex
\bibliographystyle{ieeetr}  % Change to: plain, alpha, unsrt, etc.
```

### Adjusting Margins
Edit `header.tex`:
```latex
\usepackage[left=1.5in, right=1in, top=1in, bottom=1in]{geometry}
```

### Changing Line Spacing
Edit `header.tex`:
```latex
\onehalfspacing  % Options: \singlespacing, \doublespacing
```

### Adding Appendices
Create `appendix/appendixA.tex` and add before `\bibliography`:
```latex
\appendix
\include{appendix/appendixA}
```

---

## 📚 Additional Resources

### LaTeX References
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [CTAN Package Archive](https://ctan.org/)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)

### YOLO and Deep Learning
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Academic Writing
- [IEEE Editorial Style Manual](https://journals.ieeeauthorcenter.ieee.org/)
- [APA Citation Guide](https://apastyle.apa.org/)

---

## 🎓 Submission Checklist

Before final submission:
- [ ] **Run COMPLETE compilation sequence** (pdflatex → bibtex → pdflatex → pdflatex)
- [ ] **Verify bibliography appears** - Should have 60+ references after Chapter 6
- [ ] **Check all citations resolve** - No [?] in text, all show as [1], [2], etc.
- [ ] **Verify Table of Contents** - All chapters, sections with correct page numbers
- [ ] **Check List of Figures/Tables** - All entries present with page numbers
- [ ] **Check List of Abbreviations** - Should show full list (30+ entries)
- [ ] **Verify all figures display** - No missing image errors
- [ ] **Proofread content** - Check for typos and grammatical errors
- [ ] **Confirm student info** - Name: Subhendu Das, Reg No: 1041
- [ ] **Get supervisor's signature** - Dr. Oishila Bandyopadhyay on certificate pages
- [ ] **Print on institutional letterhead** - If required by department
- [ ] **Create soft copies** - Spiral-bound (2-3 copies)
- [ ] **Prepare hard-bound copy** - After defense/revisions

### ✅ Final PDF Verification

**Expected structure in compiled PDF:**
1. **Front Matter** (Roman numerals i, ii, iii, ...)
   - Title pages (2 pages)
   - Dedication
   - Supervisor Certificate
   - Self Declaration
   - Acknowledgements
   - Abstract
   - Table of Contents
   - List of Figures
   - List of Tables
   - List of Algorithms
   - List of Abbreviations (~30 entries)

2. **Main Content** (Arabic numerals 1, 2, 3, ...)
   - Chapter 1: Introduction
   - Chapter 2: Literature Review
   - Chapter 3: Methodology
   - Chapter 4: Implementation
   - Chapter 5: Results and Analysis
   - Chapter 6: Conclusion and Future Work

3. **Back Matter**
   - Publications
   - References (60+ bibliography entries - IEEE format)

**Total Pages**: 150-160 pages

---

## 📧 Contact & Support

**Student**: Subhendu Das  
**Registration Number**: 1041  
**Email**: subhendu@iiitkalyani.ac.in  
**Supervisor**: Dr. Oishila Bandyopadhyay  
**Designation**: Assistant Professor  
**Department**: Computer Science and Engineering  
**Institution**: IIIT Kalyani, West Bengal, India  

**GitHub Repository**: [https://github.com/SubhenduScottDas/polyp-yolo](https://github.com/SubhenduScottDas/polyp-yolo)

---

## 🏆 Credits

- **Dataset**: Kvasir-SEG (Simula Research Laboratory, Norway)
- **Framework**: Ultralytics YOLOv8
- **Template**: IIIT Kalyani Executive M.Tech Thesis Template
- **Development Environment**: Python 3.8+, PyTorch 2.9.1, OpenCV 4.12

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | November 2024 | Initial draft with all 6 chapters |
| 1.1 | TBD | Post-review revisions |
| 2.0 | TBD | Final submission version |

---

**Generated on**: November 14, 2024  
**Last Modified**: November 14, 2024  
**Word Count**: ~35,000 words (estimated)  
**Page Count**: ~150 pages (estimated)

---

*For issues or questions about this thesis template, contact the Department of Computer Science and Engineering, IIIT Kalyani.*
