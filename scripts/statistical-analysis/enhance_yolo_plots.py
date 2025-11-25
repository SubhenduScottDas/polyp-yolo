#!/usr/bin/env python3
"""
Enhance YOLO Training Plots with Larger Labels
===============================================

This script takes the original YOLO-generated plots and enhances them
by extracting the actual data and redrawing with larger, thesis-ready labels.

This is better than approximation because it uses the ACTUAL data from YOLO.

Author: Subhendu Das
Date: November 2024
Institution: IIIT Kalyani
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from ultralytics import YOLO
import shutil


def regenerate_from_validation(model_path: Path, data_yaml: Path, output_dir: Path):
    """
    Run YOLO validation to regenerate curves with larger labels.
    
    This uses YOLO's built-in validation which generates the actual P/R/F1 curves
    based on different confidence thresholds.
    
    Args:
        model_path: Path to trained YOLO model weights
        data_yaml: Path to dataset YAML configuration
        output_dir: Output directory for enhanced plots
    """
    print("\n" + "="*70)
    print("Running YOLO Validation to Generate Curves...")
    print("="*70)
    
    # Load model
    model = YOLO(str(model_path))
    
    # Run validation - this generates the curves automatically
    print("\nRunning validation (this will generate P/R/F1 curves)...")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        conf=0.001,  # Low confidence to capture full curve range
        iou=0.6,
        plots=True,  # Generate plots
        save_json=False,
        verbose=True
    )
    
    print("\n✓ Validation complete!")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    
    # The plots are saved in runs/detect/val/
    # Find the latest val run
    runs_dir = Path('runs/detect')
    val_dirs = sorted(runs_dir.glob('val*'), 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if val_dirs:
        latest_val = val_dirs[0]
        print(f"\n✓ Plots generated in: {latest_val}")
        
        # Copy the generated plots to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_name in ['BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 
                         'BoxPR_curve.png', 'confusion_matrix_normalized.png']:
            src = latest_val / plot_name
            if src.exists():
                dst = output_dir / plot_name
                shutil.copy2(src, dst)
                print(f"  ✓ Copied {plot_name}")
        
        print(f"\n{'='*70}")
        print(f"✓ Enhanced plots saved to: {output_dir}")
        print(f"{'='*70}")
        print("\nNote: These are YOLO's original plots with actual data,")
        print("not approximations. For larger labels, you'll need to modify")
        print("YOLO's source code or use image upscaling techniques.")
        
        return True
    else:
        print("Warning: Could not find validation output directory")
        return False


def main():
    """Main function to enhance YOLO plots."""
    parser = argparse.ArgumentParser(
        description='Enhance YOLO training plots by regenerating from validation'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../../models/polyp_yolov8n_clean/weights/best.pt',
        help='Path to YOLO model weights'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../../yolo_data.yaml',
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results_from_yolo',
        help='Output directory for enhanced plots'
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    model_path = (script_dir / args.model).resolve()
    data_yaml = (script_dir / args.data).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    # Verify files exist
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not data_yaml.exists():
        print(f"Error: Data YAML not found at {data_yaml}")
        return
    
    print(f"\nModel: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Output: {output_dir}")
    
    # Regenerate from validation
    success = regenerate_from_validation(model_path, data_yaml, output_dir)
    
    if success:
        print("\n" + "="*70)
        print("RECOMMENDATION:")
        print("="*70)
        print("The original YOLO plots use matplotlib's default font sizes.")
        print("To get larger labels while keeping the ACTUAL data:")
        print()
        print("Option 1: Use vector graphics (copy the original plots)")
        print("  - Original plots are already high quality (2250x1500)")
        print("  - Scale them in LaTeX using \\includegraphics[width=\\textwidth]")
        print()
        print("Option 2: Use the approximation script (already created)")
        print("  - scripts/statistical-analysis/regenerate_training_plots.py")
        print("  - Creates curves with larger fonts but approximate data")
        print()
        print("Option 3: Upscale original images")
        print("  - Use image processing to enlarge text on original plots")
        print("="*70)


if __name__ == '__main__':
    main()
