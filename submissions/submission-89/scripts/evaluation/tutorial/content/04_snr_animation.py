"""
GMM Model Evaluation Tutorial - SNR Animation
=============================================

This tutorial demonstrates how to create animations showing the effect of 
Signal-to-Noise Ratio (SNR) on GMM model predictions.

Topics covered:
1. Creating animations of model behavior across SNR levels
2. Understanding how noise affects model predictions
3. Visualizing the transition from high-noise to low-noise regimes
4. Different output formats for animations
"""

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

def create_snr_animation():
    """Create SNR animation showing model behavior across noise levels."""
    print("\n==== Creating SNR Animation ====")
    
    # Initialize pipeline
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(experiment_dir=experiment_dir)
    
    print("Creating animation from SNR 3-15 dB...")
    print("This will show how the model's predictions change as noise decreases")
    
    animation = pipeline.create_animation(
        parameter='snr_db',
        parameter_range=(3, 15),
        frames=100,
        fps=20,
        models='baseline_16_layers',
        datasets='standard',
        show=['points', 'true_centers', 'predictions'],
        save_path=tutorial_output_dir / "snr_animation.mp4"
    )
    
    print("✓ Animation created!")
    print("  - At low SNR (3 dB): High noise, model predictions are inaccurate")
    print("  - At high SNR (15 dB): Low noise, model predictions align with true centers")
    return animation

def create_focused_animation():
    """Create animation focused on critical SNR transition range."""
    print("\n==== Creating Focused SNR Animation ====")
    
    pipeline = VisualizationPipeline()
    
    print("Focusing on SNR range 5-12 dB where the transition is most visible...")
    
    animation = pipeline.create_animation(
        parameter='snr_db',
        parameter_range=(5, 12),      # Critical transition range
        frames=50,
        fps=15,
        models='baseline_16_layers',
        show=['points', 'true_centers', 'predictions', 'kmeans'],  # Include KMeans comparison
        save_path=tutorial_output_dir / "focused_snr_animation.mp4"
    )
    
    print("✓ Focused animation created!")
    print("  - This range captures the transition from poor to good predictions")
    print("  - KMeans baseline included for comparison")
    
    # Convert to interactive slider
    print("\nConverting to interactive slider for frame-by-frame analysis...")
    slider_widget = animation.as_slider()
    
    return animation, slider_widget

def create_different_formats():
    """Create animations in different formats for various use cases."""
    print("\n==== Creating Animations in Different Formats ====")
    
    pipeline = VisualizationPipeline()
    
    # Define animation parameters
    base_params = {
        'parameter': 'snr_db',
        'parameter_range': (6, 12),
        'frames': 30,
        'models': 'baseline_16_layers'
    }
    
    print("Creating animation in multiple formats...")
    
    # 1. Create as MP4 animation
    print("\n1. MP4 format - best for presentations and videos")
    anim_mp4 = pipeline.create_animation(
        **base_params,
        save_path=tutorial_output_dir / "snr_transition.mp4"
    )
    
    # 2. Save as GIF
    print("\n2. GIF format - good for embedding in documents")
    anim_mp4.save_gif(tutorial_output_dir / "snr_transition.gif")
    
    # 3. Save as frame sequence
    print("\n3. Frame sequence - useful for detailed analysis")
    frames_dir = tutorial_output_dir / "animation_frames"
    frames_dir.mkdir(exist_ok=True)
    anim_mp4.save_frames(frames_dir)
    print(f"   Saved {base_params['frames']} individual frames")
    
    # 4. Interactive widget
    print("\n4. Interactive widget - best for exploration")
    widget = anim_mp4.as_slider()
    
    print("\n✓ All formats created!")
    return anim_mp4, widget

if __name__ == "__main__":
    print("=" * 60)
    print("GMM SNR ANIMATION TUTORIAL")
    print("=" * 60)
    
    # Basic animation showing full SNR range
    print("\n1. Full SNR range animation")
    animation = create_snr_animation()
    
    # Focused animation on critical transition
    print("\n2. Focused transition animation")
    focused_anim, slider = create_focused_animation()
    
    # Different output formats
    print("\n3. Multiple output formats")
    demo_anim, demo_widget = create_different_formats()
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("• SNR controls the noise level in the data")
    print("• Low SNR (3-5 dB): Model struggles to identify cluster centers")
    print("• Medium SNR (6-10 dB): Transition region where model improves rapidly")
    print("• High SNR (11-15 dB): Model accurately predicts cluster centers")
    print("• The animation clearly shows this progression")
    
    print(f"\nAll outputs saved to: {tutorial_output_dir}")
    
    # Show interactive widgets if display available
    if os.environ.get('DISPLAY', '') != '':
        print("\nDisplaying interactive widgets for exploration...")
        slider.show()
        demo_widget.show()
    else:
        print("\nNo display detected. Animations saved as files for later viewing.") 