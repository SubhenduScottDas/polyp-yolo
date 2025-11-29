#!/usr/bin/env python3
"""
Create system overview diagram with larger labels for Chapter 1.
Recreates polyp_detection_overview.png with enhanced readability.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def create_system_overview_large_labels(output_path: str):
    """
    Create high-level system overview diagram with larger, more readable labels.
    
    Shows: Input → Processing → Detection → Output pipeline
    
    Args:
        output_path: Path to save the generated figure
    """
    _, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Define boxes: (x, y, width, height, label, color)
    boxes = [
        (1.5, 2.8, 2.5, 1.4, 'Colonoscopy\nVideo/Image', '#E8F4F8'),
        (5, 2.8, 2.5, 1.4, 'YOLOv8\nDetection', '#B8E6F0'),
        (8.5, 2.8, 2.5, 1.4, 'Polyp\nLocalization', '#88D8E8'),
        (12, 2.8, 2.5, 1.4, 'Visual\nAnnotation', '#58CAE0')
    ]

    for x, y, w, h, label, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                              edgecolor='#2C5F7F', facecolor=color, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#1A3A4F')

    # Add arrows between boxes
    arrows = [(4, 3.5, 5, 3.5), (7.5, 3.5, 8.5, 3.5), (11, 3.5, 12, 3.5)]
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                               mutation_scale=35, linewidth=3, color='#2C5F7F')
        ax.add_patch(arrow)

    # Add title
    ax.text(8, 5.8, 'Automated Polyp Detection System Pipeline', 
            ha='center', fontsize=16, fontweight='bold', color='#1A3A4F')

    # Add stage labels with LARGER font size
    stage_labels = [
        (2.75, 1.5, 'Input'),
        (6.25, 1.5, 'Processing'),
        (9.75, 1.5, 'Detection'),
        (13.25, 1.5, 'Output')
    ]
    for x, y, label in stage_labels:
        ax.text(x, y, label, ha='center', fontsize=16, 
                style='italic', fontweight='600', color='#2C5F7F')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Created system overview diagram: {output_path}')
    plt.close()


def main():
    """Generate the system overview diagram."""
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Output to piracy-free-image-enlarged-thesis folder
    output_path = script_dir.parent.parent / 'thesis-prep-docs' / 'piracy-free-image-enlarged-thesis' / 'polyp_detection_overview.png'
    
    print("=" * 70)
    print("Creating System Overview Diagram with Large Labels")
    print("=" * 70)
    print(f"Output path: {output_path}")
    print()
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the figure
    create_system_overview_large_labels(str(output_path))
    
    print()
    print("=" * 70)
    print("✓ Diagram created successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
