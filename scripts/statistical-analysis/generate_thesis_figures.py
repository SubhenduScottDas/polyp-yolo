#!/usr/bin/env python3
"""
Generate Thesis Figures for Polyp Detection Project
====================================================

This script generates professional diagrams and visualizations for the thesis
on Deep Learning-Based Polyp Detection Using YOLOv8.

Author: Subhendu Das
Date: November 2024
Institution: IIIT Kalyani
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import argparse


def create_system_overview(output_path: str):
    """
    Create high-level system overview diagram for Chapter 1.
    
    Shows: Input → Processing → Detection → Output pipeline
    
    Args:
        output_path: Path to save the generated figure
    """
    _, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define boxes: (x, y, width, height, label, color)
    boxes = [
        (1, 2.5, 2, 1, 'Colonoscopy\nVideo/Image', '#E8F4F8'),
        (4, 2.5, 2, 1, 'YOLOv8\nDetection', '#B8E6F0'),
        (7, 2.5, 2, 1, 'Polyp\nLocalization', '#88D8E8'),
        (10, 2.5, 2, 1, 'Visual\nAnnotation', '#58CAE0')
    ]

    for x, y, w, h, label, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                              edgecolor='#2C5F7F', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='#1A3A4F')

    # Add arrows between boxes
    arrows = [(3, 3, 4, 3), (6, 3, 7, 3), (9, 3, 10, 3)]
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                               mutation_scale=30, linewidth=2.5, color='#2C5F7F')
        ax.add_patch(arrow)

    # Add title
    ax.text(7, 5.3, 'Automated Polyp Detection System Pipeline', 
            ha='center', fontsize=14, fontweight='bold', color='#1A3A4F')

    # Add stage labels
    stage_labels = [
        (2, 1.2, 'Input'),
        (5, 1.2, 'Processing'),
        (8, 1.2, 'Detection'),
        (11, 1.2, 'Output')
    ]
    for x, y, label in stage_labels:
        ax.text(x, y, label, ha='center', fontsize=9, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created {output_path}')
    plt.close()


def create_system_architecture(output_path: str):
    """
    Create complete system architecture diagram for Chapter 3.
    
    Shows 5-layer architecture:
    1. Data Layer (Kvasir-SEG)
    2. Preprocessing Layer (Mask to bbox, Split, Augmentation)
    3. Model Layer (YOLOv8 components)
    4. Training & Validation Layer
    5. Inference Layer
    
    Args:
        output_path: Path to save the generated figure
    """
    _, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7.5, 7.5, 'Complete YOLOv8 Polyp Detection System Architecture', 
            ha='center', fontsize=15, fontweight='bold', color='#1A3A4F')

    # Layer 1: Data
    y1 = 6
    boxes_l1 = [
        (1, y1, 2.5, 0.8, 'Kvasir-SEG\nDataset\n(1000 images)', '#FFE6E6'),
    ]
    for x, y, w, h, label, color in boxes_l1:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              edgecolor='#8B0000', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='#8B0000')

    # Layer 2: Preprocessing
    y2 = 4.5
    boxes_l2 = [
        (0.5, y2, 1.8, 0.8, 'Mask to\nBbox', '#FFF4E6'),
        (2.5, y2, 1.8, 0.8, 'Train/Val\nSplit (80/20)', '#FFF4E6'),
        (4.5, y2, 1.8, 0.8, 'Data\nAugmentation', '#FFF4E6'),
    ]
    for x, y, w, h, label, color in boxes_l2:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              edgecolor='#CC6600', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='#CC6600')

    # Arrow from data to preprocessing
    arrow = FancyArrowPatch((2.25, y1), (3.5, y2+0.8), arrowstyle='->', 
                           mutation_scale=25, linewidth=2, color='#555')
    ax.add_patch(arrow)

    # Layer 3: Model Architecture
    y3 = 3
    boxes_l3 = [
        (7, y3, 2.2, 1.2, 'YOLOv8-nano\nBackbone\n(CSPDarknet)', '#E6F3FF'),
        (9.5, y3, 2.2, 1.2, 'Neck\n(PANet/FPN)', '#E6F3FF'),
        (12, y3, 2.2, 1.2, 'Detection\nHead', '#E6F3FF'),
    ]
    for x, y, w, h, label, color in boxes_l3:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              edgecolor='#0066CC', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='#0066CC')

    # Arrows between model components
    arrow1 = FancyArrowPatch((9.2, y3+0.6), (9.5, y3+0.6), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#0066CC')
    arrow2 = FancyArrowPatch((11.7, y3+0.6), (12, y3+0.6), arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='#0066CC')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    # Arrow from preprocessing to model
    arrow = FancyArrowPatch((5.5, y2+0.4), (7, y3+0.6), arrowstyle='->', 
                           mutation_scale=25, linewidth=2, color='#555')
    ax.add_patch(arrow)

    # Layer 4: Training & Validation
    y4 = 1.5
    boxes_l4 = [
        (7, y4, 2.8, 0.8, 'Training\n(50 epochs, BS=16)', '#E6FFE6'),
        (10.2, y4, 2.8, 0.8, 'Validation\n(mAP@50: 89.4%)', '#E6FFE6'),
    ]
    for x, y, w, h, label, color in boxes_l4:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              edgecolor='#006600', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='#006600')

    # Arrow from model to training
    arrow = FancyArrowPatch((10.5, y3), (8.5, y4+0.8), arrowstyle='->', 
                           mutation_scale=25, linewidth=2, color='#555')
    ax.add_patch(arrow)

    # Layer 5: Inference Pipeline
    y5 = 0.2
    boxes_l5 = [
        (1, y5, 2.5, 0.8, 'Image/Video\nInput', '#F0E6FF'),
        (4, y5, 2.5, 0.8, 'Model\nInference', '#F0E6FF'),
        (7, y5, 2.5, 0.8, 'NMS\nPost-processing', '#F0E6FF'),
        (10, y5, 2.5, 0.8, 'Annotated\nOutput', '#F0E6FF'),
    ]
    for x, y, w, h, label, color in boxes_l5:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              edgecolor='#6600CC', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='#6600CC')

    # Arrows in inference pipeline
    for i in range(3):
        x1 = 3.5 + i*3
        x2 = 4 + i*3
        arrow = FancyArrowPatch((x1, y5+0.4), (x2, y5+0.4), arrowstyle='->', 
                               mutation_scale=25, linewidth=2, color='#6600CC')
        ax.add_patch(arrow)

    # Add layer labels on left side
    labels = [
        (0.2, y1+0.4, 'Data', '#8B0000'),
        (0.2, y2+0.4, 'Preprocessing', '#CC6600'),
        (6.5, y3+0.6, 'Model\nArchitecture', '#0066CC'),
        (6.5, y4+0.4, 'Training &\nValidation', '#006600'),
        (0.2, y5+0.4, 'Inference', '#6600CC')
    ]
    for x, y, text, color in labels[:2] + labels[4:]:
        ax.text(x, y, text, fontsize=10, fontweight='bold', color=color)
    for x, y, text, color in labels[2:4]:
        ax.text(x, y, text, fontsize=10, fontweight='bold', color=color, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created {output_path}')
    plt.close()


def create_system_architecture_clearer(output_path: str):
    """
    Create complete system architecture diagram with LARGER, MORE READABLE labels.
    Optimized for thesis with improved legibility and layout.
    
    Shows 5-layer architecture with enhanced visibility:
    1. Data Layer (Kvasir-SEG)
    2. Preprocessing Layer (Mask to bbox, Split, Augmentation)
    3. Model Layer (YOLOv8 components)
    4. Training & Validation Layer
    5. Inference Layer
    
    Args:
        output_path: Path to save the generated figure
    """
    _, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title - MUCH LARGER
    ax.text(10, 11.2, 'Complete YOLOv8 Polyp Detection System Architecture', 
            ha='center', fontsize=22, fontweight='bold', color='#1A3A4F')

    # Layer 1: Data - LARGER boxes and text
    y1 = 9
    boxes_l1 = [
        (2, y1, 4, 1.5, 'Kvasir-SEG\nDataset\n(1000 images)', '#FFE6E6'),
    ]
    for x, y, w, h, label, color in boxes_l1:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                              edgecolor='#8B0000', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#8B0000')

    # Layer 2: Preprocessing - LARGER boxes
    y2 = 6.5
    boxes_l2 = [
        (1.5, y2, 3, 1.3, 'Mask to\nBbox', '#FFF4E6'),
        (5, y2, 3, 1.3, 'Train/Val\nSplit (80/20)', '#FFF4E6'),
        (8.5, y2, 3, 1.3, 'Data\nAugmentation', '#FFF4E6'),
    ]
    for x, y, w, h, label, color in boxes_l2:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                              edgecolor='#CC6600', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#CC6600')

    # Arrow from data to preprocessing
    arrow = FancyArrowPatch((4, y1), (6.5, y2+1.3), arrowstyle='->', 
                           mutation_scale=35, linewidth=3, color='#555')
    ax.add_patch(arrow)

    # Layer 3: Model Architecture - LARGER boxes
    y3 = 4.3
    boxes_l3 = [
        (11.5, y3, 3.5, 1.8, 'YOLOv8-nano\nBackbone\n(CSPDarknet)', '#E6F3FF'),
        (15.5, y3, 3.5, 1.8, 'Neck\n(PANet/FPN)', '#E6F3FF'),
    ]
    # Detection head box
    boxes_l3.append((11.5, y3-2.2, 7.5, 1.5, 'Detection Head\n(Anchor-Free, Decoupled)', '#E6F3FF'))
    
    for x, y, w, h, label, color in boxes_l3:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                              edgecolor='#0066CC', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#0066CC')

    # Arrows between model components (top row)
    arrow1 = FancyArrowPatch((15, y3+0.9), (15.5, y3+0.9), arrowstyle='->', 
                            mutation_scale=35, linewidth=3.5, color='#0066CC')
    ax.add_patch(arrow1)
    
    # Arrow from neck to detection head
    arrow2 = FancyArrowPatch((15, y3), (15, y3-2.2+1.5), arrowstyle='->', 
                            mutation_scale=35, linewidth=3.5, color='#0066CC')
    ax.add_patch(arrow2)

    # Arrow from preprocessing to model
    arrow = FancyArrowPatch((10, y2+0.6), (11.5, y3+0.9), arrowstyle='->', 
                           mutation_scale=35, linewidth=3, color='#555')
    ax.add_patch(arrow)

    # Layer 4: Training & Validation - LARGER boxes
    y4 = 8.5
    boxes_l4 = [
        (12, y4, 4, 1.3, 'Training\n(50 epochs, BS=16)', '#E6FFE6'),
        (16.5, y4, 3, 1.3, 'Validation\n(mAP@50: 89.4%)', '#E6FFE6'),
    ]
    for x, y, w, h, label, color in boxes_l4:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                              edgecolor='#006600', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#006600')

    # Arrow from model to training
    arrow = FancyArrowPatch((13.5, y3+1.8), (14, y4), arrowstyle='->', 
                           mutation_scale=35, linewidth=3, color='#555')
    ax.add_patch(arrow)

    # Layer 5: Inference Pipeline - LARGER boxes
    y5 = 0.1
    boxes_l5 = [
        (2, y5, 3.5, 1.2, 'Image/Video\nInput', '#F0E6FF'),
        (6.5, y5, 3.5, 1.2, 'Model\nInference', '#F0E6FF'),
        (11, y5, 3.5, 1.2, 'NMS\nPost-processing', '#F0E6FF'),
        (15.5, y5, 3.5, 1.2, 'Annotated\nOutput', '#F0E6FF'),
    ]
    for x, y, w, h, label, color in boxes_l5:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                              edgecolor='#6600CC', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#6600CC')

    # Arrows in inference pipeline
    for i in range(3):
        x1 = 5.5 + i*4.5
        x2 = 6.5 + i*4.5
        arrow = FancyArrowPatch((x1, y5+0.6), (x2, y5+0.6), arrowstyle='->', 
                               mutation_scale=35, linewidth=3, color='#6600CC')
        ax.add_patch(arrow)

    # Add step numbers in circles OUTSIDE boxes - positioned further left/right
    steps = [
        (0.5, y1+0.75, '1', '#8B0000'),
        (0.5, y2+1.95, '2', '#CC6600'),
        (10.8, y3+0.9, '3', '#0066CC'),
        (11.2, y4+0.65, '4', '#006600'),
        (0.5, y5+1.5, '5', '#6600CC')
    ]
    
    for x, y, num, color in steps:
        circle = plt.Circle((x, y), 0.35, color=color, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, num, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='white', zorder=11)
    
    # Add layer labels OUTSIDE boxes next to numbers - LARGER text
    labels = [
        (0.90, y1+0.75, 'Data', '#8B0000', 18),
        (0.90, y2+1.95, 'Pre-\nprocessing', '#CC6600', 18),
        (10.2, y3+0.9, 'Model\nArchitecture', '#0066CC', 18),
        (10.6, y4+0.65, 'Training &\nValidation', '#006600', 18),
        (0.90, y5+1.5, 'Inference', '#6600CC', 18)
    ]
    
    for x, y, text, color, fontsize in labels:
        ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=color,
                ha='left' if x < 5 else 'right', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created {output_path}')
    plt.close()


def create_mask_to_bbox_visualization(kvasir_dir: str, output_path: str, num_examples: int = 3):
    """
    Create mask to bounding box conversion visualization using actual Kvasir-SEG data.
    
    Shows 3-column layout: Original Image | Segmentation Mask | YOLO Bounding Box
    
    Args:
        kvasir_dir: Path to Kvasir-SEG dataset directory
        output_path: Path to save the generated figure
        num_examples: Number of example images to show (default: 3)
    """
    data_dir = Path(kvasir_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    if not image_dir.exists():
        print(f"Warning: {image_dir} not found. Skipping mask_to_bbox visualization.")
        return
    
    image_files = list(image_dir.glob('*.jpg'))[:num_examples]
    
    if not image_files:
        print(f"Warning: No images found in {image_dir}")
        return

    fig, axes = plt.subplots(num_examples, 3, figsize=(14, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(image_files):
        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}")
            continue
            
        mask = cv2.imread(str(mask_path), 0)
        
        # Resize for consistency
        img = cv2.resize(img, (400, 400))
        mask = cv2.resize(mask, (400, 400))
        
        # Find bounding box from mask
        coords = np.where(mask > 127)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Create bbox image
            img_bbox = img.copy()
            cv2.rectangle(img_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Create mask overlay
            mask_colored = np.zeros_like(img)
            mask_colored[mask > 127] = [255, 255, 0]  # Yellow
            img_overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            
            # Plot
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f'Original Image {idx+1}', fontweight='bold', fontsize=12)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(img_overlay)
            axes[idx, 1].set_title('Segmentation Mask', fontweight='bold', fontsize=12)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(img_bbox)
            axes[idx, 2].set_title('YOLO Bounding Box', fontweight='bold', fontsize=12)
            axes[idx, 2].axis('off')

    fig.suptitle('Mask to Bounding Box Conversion Process', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created {output_path}')
    plt.close()


def create_augmentation_examples(kvasir_dir: str, output_path: str, sample_idx: int = 5):
    """
    Create data augmentation examples visualization.
    
    Shows 8 different augmentation techniques applied to a sample polyp image.
    
    Args:
        kvasir_dir: Path to Kvasir-SEG dataset directory
        output_path: Path to save the generated figure
        sample_idx: Index of sample image to use (default: 5)
    """
    data_dir = Path(kvasir_dir)
    image_dir = data_dir / 'images'
    
    if not image_dir.exists():
        print(f"Warning: {image_dir} not found. Skipping augmentation examples.")
        return
    
    image_files = list(image_dir.glob('*.jpg'))
    if sample_idx >= len(image_files):
        sample_idx = 0
    
    img_path = image_files[sample_idx]

    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))

    # Create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # 1. Original
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Horizontal flip
    img_hflip = cv2.flip(img, 1)
    axes[1].imshow(img_hflip)
    axes[1].set_title('Horizontal Flip', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. Vertical flip
    img_vflip = cv2.flip(img, 0)
    axes[2].imshow(img_vflip)
    axes[2].set_title('Vertical Flip', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # 4. Rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 30, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h))
    axes[3].imshow(img_rot)
    axes[3].set_title('Rotation (30°)', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    # 5. Brightness increase
    img_bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
    axes[4].imshow(img_bright)
    axes[4].set_title('Brightness Increase', fontsize=11, fontweight='bold')
    axes[4].axis('off')

    # 6. HSV adjustment
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:,:,0] = (img_hsv[:,:,0] + 15) % 180  # Hue shift
    img_hsv[:,:,1] = np.clip(img_hsv[:,:,1] * 1.2, 0, 255)  # Saturation
    img_hsv = img_hsv.astype(np.uint8)
    img_hsv_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    axes[5].imshow(img_hsv_rgb)
    axes[5].set_title('HSV Adjustment', fontsize=11, fontweight='bold')
    axes[5].axis('off')

    # 7. Gaussian blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    axes[6].imshow(img_blur)
    axes[6].set_title('Gaussian Blur', fontsize=11, fontweight='bold')
    axes[6].axis('off')

    # 8. Center crop + resize
    crop_size = 300
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    img_crop = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    img_crop = cv2.resize(img_crop, (400, 400))
    axes[7].imshow(img_crop)
    axes[7].set_title('Center Crop + Resize', fontsize=11, fontweight='bold')
    axes[7].axis('off')

    fig.suptitle('Data Augmentation Techniques Applied During Training', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created {output_path}')
    plt.close()


def main():
    """Main function to generate all thesis figures."""
    parser = argparse.ArgumentParser(
        description='Generate thesis figures for Polyp Detection project'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../thesis-prep-docs/draft-thesis/Figure',
        help='Output directory for generated figures'
    )
    parser.add_argument(
        '--kvasir-dir',
        type=str,
        default='../../data/archive/Kvasir-SEG/Kvasir-SEG',
        help='Path to Kvasir-SEG dataset directory'
    )
    parser.add_argument(
        '--figures',
        nargs='+',
        choices=['overview', 'architecture', 'mask2bbox', 'augmentation', 'all'],
        default=['all'],
        help='Which figures to generate'
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    kvasir_dir = (script_dir / args.kvasir_dir).resolve()
    
    # Create output directories
    (output_dir / 'chp1').mkdir(parents=True, exist_ok=True)
    (output_dir / 'chp3').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Thesis Figures")
    print("=" * 60)
    
    figures_to_generate = args.figures
    if 'all' in figures_to_generate:
        figures_to_generate = ['overview', 'architecture', 'mask2bbox', 'augmentation']
    
    # Generate requested figures
    if 'overview' in figures_to_generate:
        print("\n[1/4] Generating system overview diagram...")
        create_system_overview(str(output_dir / 'chp1' / 'polyp_detection_overview.png'))
    
    if 'architecture' in figures_to_generate:
        print("\n[2/4] Generating system architecture diagram...")
        create_system_architecture(str(output_dir / 'chp3' / 'system_architecture.png'))
        print("\n[2b/4] Generating clearer system architecture diagram...")
        create_system_architecture_clearer(str(script_dir / 'system_architecture_clearer.png'))
    
    if 'mask2bbox' in figures_to_generate:
        print("\n[3/4] Generating mask to bbox visualization...")
        create_mask_to_bbox_visualization(
            str(kvasir_dir),
            str(output_dir / 'chp3' / 'mask_to_bbox.png')
        )
    
    if 'augmentation' in figures_to_generate:
        print("\n[4/4] Generating augmentation examples...")
        create_augmentation_examples(
            str(kvasir_dir),
            str(output_dir / 'chp3' / 'augmentation_examples.png')
        )
    
    print("\n" + "=" * 60)
    print("✓ All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
