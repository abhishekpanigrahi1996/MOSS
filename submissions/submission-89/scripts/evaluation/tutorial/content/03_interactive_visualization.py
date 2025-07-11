"""
GMM Model Evaluation Tutorial - Interactive Visualization
========================================================

This module demonstrates how to create an interactive visualization of GMM data
using matplotlib with sliders to adjust the SNR in real-time.

This version uses the new visualization API for dramatically simplified code
while maintaining the exact same functionality as the original.
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new visualization API
from scripts.evaluation.tutorial.src.visualization import VisualizationPipeline

# Set device
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive GMM Visualization')
    parser.add_argument('--kmeans', action='store_true', help='Run KMeans and display centers')
    parser.add_argument('--clusters', type=int, default=None, help='Number of clusters for KMeans')
    args = parser.parse_args()
    
    # Define paths
    experiment_base_dir = Path('/mount/Storage/gmm-v4/output/final_experiments')
    tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check display
    has_display = os.environ.get('DISPLAY', '') != ''
    
    # Initialize pipeline
    print("Initializing visualization pipeline...")
    pipeline = VisualizationPipeline(
        experiment_dir=experiment_base_dir,
        output_dir=tutorial_output_dir,
        device=device
    )
    
    # Determine what to show
    show = ['points', 'true_centers', 'predictions']
    if args.kmeans:
        show.append('kmeans')
    
    # Create interactive visualization using the API
    print("Creating interactive visualization...")
    widget = pipeline.create_interactive(
        parameters={'snr_db': (3, 15, 0.15)},  # SNR range with slider
        models='baseline_16_layers',           # Model to use
        datasets='standard',                   # Dataset type
        show=show,                            # What to display
        figsize=(10, 8)
    )
    
    # Save snapshot
    save_path = tutorial_output_dir / "interactive_visualization.png"
    print(f"Saving visualization snapshot to {save_path}")
    widget.save_snapshot(save_path)
    print(f"Snapshot saved to {save_path}")
    
    # Show if display available
    if has_display:
        print("Showing interactive visualization...")
        widget.show()
    else:
        print("No display detected. Skipping interactive visualization.")
        print("The visualization snapshot has been saved and can be viewed as an image.")