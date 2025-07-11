"""
GMM Model Evaluation Tutorial - Part 2: Flow Speed Comparison
============================================================

This tutorial explores how flow speed varies with SNR (Signal-to-Noise Ratio)
and how models with different layer counts adapt their flow speeds.

Key concepts:
- Flow speed: How fast the diffusion process transforms data at each layer
- Total flow: Flow speed × number of layers (total transformation capacity)
- SNR dependency: Why flow speed changes with noise levels
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 2: Flow Speed Comparison")
print("=" * 70)
print()

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import utilities
from scripts.evaluation.tutorial.src.io import (
    load_model_from_experiment,
    create_data_samples
)
from scripts.evaluation.tutorial.src.visualization import (
    set_plotting_style,
    format_axis_with_grid,
    format_legend,
    create_comparison_figure,
    save_figure
)
from scripts.evaluation.tutorial.src.eval_utils import get_flow_prediction

# Set global plotting style
set_plotting_style()

print("Understanding Flow Speed in Diffusion Models")
print("-" * 44)
print("\nFlow speed determines how much transformation happens at each layer:")
print("- High SNR (low noise): Less transformation needed → Lower flow speed")
print("- Low SNR (high noise): More transformation needed → Higher flow speed")
print()
print("Models adapt their flow speed based on the number of layers:")
print("- 16-layer model: Needs higher flow speed per layer")
print("- 64-layer model: Can use lower flow speed per layer")
print("- Total flow (speed × layers) should be similar for same performance")
print()

def plot_flow_speed_comparison(model_configs, snr_range=(3, 15), num_points=100, save_path=None):
    """
    Plot flow speed and total flow comparisons for different models.
    
    This function demonstrates:
    1. How to extract flow predictions from models
    2. How flow speed varies with SNR
    3. The relationship between layer count and flow speed
    """
    print("\nComparing Flow Speeds Across Models")
    print("-" * 35)
    
    # Define paths
    output_dir = Path('/mount/Storage/gmm-v4/output')
    experiment_base_dir = output_dir / 'final_experiments'
    
    # Create SNR values for evaluation
    print(f"\nCreating SNR range from {snr_range[0]} to {snr_range[1]} dB with {num_points} points")
    snr_db = torch.linspace(snr_range[0], snr_range[1], num_points)
    snr_db_np = snr_db.cpu().numpy()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = create_comparison_figure(n_plots=2, figsize=(12, 4), dpi=300)
    
    # Dictionary to store flow speeds for analysis
    flow_speeds = {}
    
    # Get baseline models only for cleaner comparison
    baseline_models = {k: v for k, v in model_configs.items() if k.startswith("baseline_")}
    print(f"\nComparing {len(baseline_models)} baseline models:")
    
    # Load each model and get predictions
    for model_key, model_config in baseline_models.items():
        try:
            # Load model
            model_path = experiment_base_dir / model_config["path"]
            print(f"\n  Loading {model_config['name']} ({model_config['layers']} layers)...")
            
            model, config = load_model_from_experiment(model_path, load_best=False, device=device)
            
            # Get flow prediction
            predictor = get_flow_prediction(model)
            flow_speed = predictor(snr_db).detach().cpu().numpy()
            
            # Store flow speed for analysis
            flow_speeds[model_config["name"]] = flow_speed
            
            # Analyze flow speed characteristics
            print(f"    Flow speed range: {flow_speed.min():.3f} to {flow_speed.max():.3f}")
            print(f"    Average flow speed: {flow_speed.mean():.3f}")
            
            # Plot flow speed on the first subplot
            ax1.plot(snr_db_np, flow_speed, 
                    label=model_config["name"], 
                    linestyle=model_config["linestyle"],
                    color=model_config["color"],
                    linewidth=1.5)
            
            # Calculate total flow (flow_speed * num_layers)
            total_flow = flow_speed * model_config["layers"]
            print(f"    Total flow range: {total_flow.min():.1f} to {total_flow.max():.1f}")
            
            # Plot total flow on the second subplot
            ax2.plot(snr_db_np, total_flow, 
                    label=model_config["name"], 
                    linestyle=model_config["linestyle"],
                    color=model_config["color"],
                    linewidth=1.5)
            
        except Exception as e:
            print(f"    Error loading model: {e}")
    
    # Configure the first subplot (Flow Speed)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Flow Speed')
    ax1.set_title('Flow Speed vs SNR')
    ax1.set_xlim(snr_range)
    format_axis_with_grid(ax1, x_minor_step=1, y_minor_step=0.1)
    format_legend(ax1)
    
    # Configure the second subplot (Total Flow)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Total Flow (Speed × Layers)')
    ax2.set_title('Total Flow vs SNR')
    ax2.set_xlim(snr_range)
    format_axis_with_grid(ax2, x_minor_step=1, y_minor_step=1)
    format_legend(ax2)
    
    print("\nKey Observations:")
    print("- Models with fewer layers use higher flow speeds")
    print("- Flow speed decreases as SNR increases (less noise = less transformation needed)")
    print("- Total flow is relatively consistent across models (similar total capacity)")
    
    # Save the figure if path is provided
    if save_path:
        save_figure(fig, save_path, dpi=300)
        print(f"\nFigure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("\nDefining Model Configurations")
    print("-" * 29)
    
    # Define models to compare
    model_configs = {
        "baseline_16_layers": {
            "name": "16 Layers", 
            "path": "baseline_16_layers", 
            "linestyle": "-", 
            "color": "blue", 
            "layers": 16
        },
        "baseline_32_layers": {
            "name": "32 Layers", 
            "path": "baseline_32_layers", 
            "linestyle": "--", 
            "color": "red", 
            "layers": 32
        },
        "baseline_64_layers": {
            "name": "64 Layers", 
            "path": "baseline_64_layers", 
            "linestyle": "-.", 
            "color": "green", 
            "layers": 64
        },
        "simple_16_layers": {
            "name": "Simple 16L", 
            "path": "simple_16_layers", 
            "linestyle": ":", 
            "color": "purple", 
            "layers": 16
        },
        "hard_16_layers": {
            "name": "Hard 16L", 
            "path": "hard_16_layers", 
            "linestyle": "--", 
            "color": "orange", 
            "layers": 16
        }
    }
    
    print("\nWe'll compare baseline models with different layer counts:")
    print("- 16 layers: Fewer layers, expects higher flow speed")
    print("- 32 layers: Medium depth, balanced flow speed")
    print("- 64 layers: Many layers, can use lower flow speed")
    
    # Create output directory if it doesn't exist
    tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot flow speed comparison
    print("\nGenerating Flow Speed Comparison")
    print("=" * 32)
    
    fig = plot_flow_speed_comparison(
        model_configs=model_configs,
        snr_range=(3, 15),
        num_points=100,
        save_path=tutorial_output_dir / "flow_speed_comparison.png"
    )
    
    print("\nFlow speed comparison complete.")
    plt.show() 