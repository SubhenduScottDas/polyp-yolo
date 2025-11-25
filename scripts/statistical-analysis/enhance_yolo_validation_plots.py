#!/usr/bin/env python3
"""
Enhance YOLO Validation Plots with Larger Fonts
================================================
Uses actual YOLO validation data to regenerate plots with thesis-ready font sizes.
Matches the exact format of YOLO's plots but with significantly larger labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import seaborn as sns

def plot_curves_enhanced(results, save_dir, dpi=150):
    """
    Recreate YOLO P/R/F1/PR curves with larger fonts using actual validation data
    """
    # Get the actual curve data from YOLO validation
    px = results.box.px  # Confidence thresholds (X-axis for P/R/F1)
    p_curve = results.box.p_curve[0]  # Precision values
    r_curve = results.box.r_curve[0]  # Recall values  
    f1_curve = results.box.f1_curve[0]  # F1 values
    
    # === Precision vs Confidence ===
    _, ax = plt.subplots(figsize=(15, 10), dpi=dpi)
    ax.plot(px, p_curve, linewidth=4, color='#0047AB', alpha=1.0)
    ax.set_xlabel('Confidence', fontsize=32, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=32, fontweight='bold')
    ax.set_title('Precision-Confidence Curve', fontsize=34, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=28, width=3, length=10)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(save_dir / 'BoxP_curve.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'✓ Created {save_dir / "BoxP_curve.png"}')
    
    # === Recall vs Confidence ===
    _, ax = plt.subplots(figsize=(15, 10), dpi=dpi)
    ax.plot(px, r_curve, linewidth=4, color='#0047AB', alpha=1.0)
    ax.set_xlabel('Confidence', fontsize=32, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=32, fontweight='bold')
    ax.set_title('Recall-Confidence Curve', fontsize=34, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=28, width=3, length=10)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(save_dir / 'BoxR_curve.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'✓ Created {save_dir / "BoxR_curve.png"}')
    
    # === F1 vs Confidence ===
    _, ax = plt.subplots(figsize=(15, 10), dpi=dpi)
    ax.plot(px, f1_curve, linewidth=4, color='#0047AB', alpha=1.0)
    ax.set_xlabel('Confidence', fontsize=32, fontweight='bold')
    ax.set_ylabel('F1', fontsize=32, fontweight='bold')
    ax.set_title('F1-Confidence Curve', fontsize=34, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=28, width=3, length=10)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(save_dir / 'BoxF1_curve.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'✓ Created {save_dir / "BoxF1_curve.png"}')
    
    # === Precision-Recall Curve ===
    _, ax = plt.subplots(figsize=(15, 10), dpi=dpi)
    ax.plot(r_curve, p_curve, linewidth=4, color='#0047AB', alpha=1.0, label=f'all classes {results.box.map50:.3f} mAP@0.5')
    ax.set_xlabel('Recall', fontsize=32, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=32, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=34, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=28, width=3, length=10)
    ax.legend(fontsize=24, loc='lower left')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(save_dir / 'BoxPR_curve.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'✓ Created {save_dir / "BoxPR_curve.png"}')

def plot_results_enhanced(csv_path, save_dir, dpi=150):
    """
    Recreate results.png with larger fonts using actual training data
    """
    df = pd.read_csv(csv_path)
    
    _, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=dpi)
    axes = axes.flatten()
    
    # Define metrics (matching YOLO's layout)
    metrics = [
        ('train/box_loss', 'Box Loss (Train)', '#0047AB'),
        ('train/cls_loss', 'Class Loss (Train)', '#0047AB'),
        ('train/dfl_loss', 'DFL Loss (Train)', '#0047AB'),
        ('metrics/precision(B)', 'Precision', '#0047AB'),
        ('metrics/recall(B)', 'Recall', '#0047AB'),
        ('metrics/mAP50(B)', 'mAP@50', '#0047AB'),
        ('metrics/mAP50-95(B)', 'mAP@50-95', '#0047AB'),
        ('val/box_loss', 'Box Loss (Val)', '#0047AB')
    ]
    
    epochs = df['epoch'].values
    
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx]
        if col in df.columns:
            values = df[col].values
            ax.plot(epochs, values, linewidth=3, color=color, alpha=1.0)
            ax.set_xlabel('Epoch', fontsize=22, fontweight='bold')
            ax.set_ylabel(title, fontsize=22, fontweight='bold')
            ax.tick_params(labelsize=20, width=2.5, length=8)
            ax.grid(True, alpha=0.3, linewidth=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
    
    plt.suptitle('Training Results', fontsize=26, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'✓ Created {save_dir / "results.png"}')

def plot_confusion_matrix_enhanced(results, save_dir, dpi=150):
    """
    Plot confusion matrix with larger fonts using actual validation data
    """
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        
        # Normalize
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        # Plot
        _, ax = plt.subplots(figsize=(14, 12), dpi=dpi)
        
        labels = ['background', 'polyp']
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': ''},
                   ax=ax, linewidths=3, linecolor='white',
                   annot_kws={'size': 32, 'weight': 'bold'},
                   square=True)
        
        ax.set_xlabel('Predicted', fontsize=32, fontweight='bold', labelpad=15)
        ax.set_ylabel('True', fontsize=32, fontweight='bold', labelpad=15)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=34, fontweight='bold', pad=20)
        ax.tick_params(labelsize=28, width=3, length=10)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix_normalized.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f'✓ Created {save_dir / "confusion_matrix_normalized.png"}')

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / 'models/polyp_yolov8n_clean/weights/best.pt'
    csv_path = base_dir / 'models/polyp_yolov8n_clean/results.csv'
    data_yaml = base_dir / 'yolo_data.yaml'
    output_dir = base_dir / 'scripts/statistical-analysis/results_original'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Enhancing YOLO Validation Plots with Larger Fonts")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Output: {output_dir}\n")
    
    # Run validation to get actual data
    print("[1/3] Running YOLO validation to get actual curve data...")
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), plots=False, verbose=False)
    print(f"  Validation metrics: P={results.box.mp:.3f}, R={results.box.mr:.3f}, mAP@50={results.box.map50:.3f}")
    
    # Generate plots
    print("\n[2/3] Generating P/R/F1/PR curves with larger fonts...")
    plot_curves_enhanced(results, output_dir)
    
    print("\n[3/3] Generating training results and confusion matrix...")
    plot_results_enhanced(csv_path, output_dir)
    plot_confusion_matrix_enhanced(results, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ All enhanced plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
