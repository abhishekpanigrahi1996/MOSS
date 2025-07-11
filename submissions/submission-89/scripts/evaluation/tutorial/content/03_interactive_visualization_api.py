"""
GMM Model Evaluation Tutorial - Part 3: Interactive Visualization API
====================================================================

This tutorial demonstrates creating interactive visualizations using
the visualization pipeline API with real-time parameter sliders.

What you'll see:
1. Creating interactive widgets with parameter sliders
2. Real-time model predictions as parameters change
3. Multi-parameter control widgets
4. Converting animations to interactive sliders
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 3: Interactive Visualization API")
print("=" * 70)
print()

import sys
import os
from pathlib import Path

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new Phase 3 API
from scripts.evaluation.tutorial.src.visualization_pipeline import VisualizationPipeline

# Define paths
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

print("Interactive Visualization with Real-time Parameter Control")
print("-" * 57)
print("\nInteractive widgets allow you to:")
print("- Adjust parameters with sliders and see instant updates")
print("- Observe how model predictions change with different SNR values")
print("- Compare different visualization components (points, centers, predictions)")
print()

def create_snr_interactive_widget():
    """Create interactive widget with SNR slider."""
    print("\nCreating Interactive Widget with SNR Slider")
    print("-" * 43)
    
    # Initialize pipeline
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(experiment_dir=experiment_dir)
    
    print("\nThe widget will show:")
    print("- Data points: Noisy observations")
    print("- True centers: Actual GMM component centers")
    print("- Predictions: Model's predicted centers")
    print("\nAdjust the SNR slider to see how predictions change with noise level.")
    
    widget = pipeline.create_interactive(
        parameters={'snr_db': (3, 15)},  # SNR range with automatic slider
        models='baseline_16_layers',      # Model to use
        datasets='standard',              # Dataset type
        show=['points', 'true_centers', 'predictions']
    )
    
    # Save a snapshot
    snapshot_path = tutorial_output_dir / "interactive_snr_snapshot.png"
    widget.save_snapshot(snapshot_path)
    print(f"\nSnapshot saved to: {snapshot_path}")
    
    return widget

def create_multi_parameter_widget():
    """Create interactive widget with multiple parameter sliders."""
    print("\nCreating Multi-Parameter Interactive Widget")
    print("-" * 43)
    
    pipeline = VisualizationPipeline()
    
    print("\nThis widget includes multiple controls:")
    print("- SNR slider: Adjust noise level (3-15 dB)")
    print("- Flow speed slider: Control diffusion speed (0.1-2.0)")
    print("\nAdditional visualization: K-means clustering results")
    
    widget = pipeline.create_interactive(
        parameters={
            'snr_db': (3, 15),           # SNR slider
            'flow_speed': (0.1, 2.0),    # Flow speed slider  
        },
        models='baseline_16_layers',
        show=['points', 'true_centers', 'predictions', 'kmeans']  # Include KMeans
    )
    
    # Save snapshot
    snapshot_path = tutorial_output_dir / "multi_parameter_snapshot.png"
    widget.save_snapshot(snapshot_path)
    print(f"\nSnapshot saved to: {snapshot_path}")
    
    return widget

def demonstrate_animation_to_widget_conversion():
    """Show how animations can be converted to interactive widgets."""
    print("\nConverting Animations to Interactive Widgets")
    print("-" * 44)
    
    pipeline = VisualizationPipeline()
    
    print("\nSometimes you want to convert an animation into an interactive slider.")
    print("This is useful when you want to manually control the animation frame.")
    
    # 1. Create animation first
    print("\nStep 1: Creating animation with 30 frames...")
    animation = pipeline.create_animation(
        parameter='snr_db',
        parameter_range=(5, 12),
        frames=30,
        models='baseline_16_layers'
    )
    
    # 2. Convert to interactive widget
    print("Step 2: Converting animation to interactive slider...")
    widget = animation.as_slider()
    
    print("\nThe slider now controls which frame of the animation is displayed.")
    print("This gives you precise control over the visualization state.")
    
    # Save snapshot
    snapshot_path = tutorial_output_dir / "converted_widget_snapshot.png"
    widget.save_snapshot(snapshot_path)
    print(f"\nSnapshot saved to: {snapshot_path}")
    
    return animation, widget

if __name__ == "__main__":
    # Basic SNR interactive widget
    snr_widget = create_snr_interactive_widget()
    
    # Multi-parameter widget
    multi_widget = create_multi_parameter_widget()
    
    # Animation-to-widget conversion
    animation, converted_widget = demonstrate_animation_to_widget_conversion()
    
    print(f"\n\nAll visualization snapshots have been saved to:")
    print(f"  {tutorial_output_dir}")
    
    # Show interactive widgets if display available
    if os.environ.get('DISPLAY', '') != '':
        print("\nDisplaying interactive widgets...")
        print("Use the sliders to adjust parameters in real-time!")
        snr_widget.show()
        multi_widget.show()
        converted_widget.show()
    else:
        print("\nNo display detected. Widget snapshots saved as images.") 