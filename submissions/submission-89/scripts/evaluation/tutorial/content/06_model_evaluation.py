"""
GMM Model Evaluation Tutorial - Model Comparison
==============================================

This tutorial demonstrates how to evaluate and compare different GMM models
on various datasets, showing how model performance varies with data conditions.

Topics covered:
1. Comparing model predictions on high vs low SNR data
2. Understanding how models handle different noise levels
3. Visualizing model accuracy across conditions
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.evaluation.tutorial.src.visualization import VisualizationPipeline
from scripts.evaluation.tutorial.src.io import create_data_samples

# Define paths
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

# Define fixed random seeds for reproducibility (same as backup)
RANDOM_SEEDS = {
    'high_snr_fixed': 90,  # Seed for high SNR sample → 15.0 dB
    'low_snr_fixed': 123   # Seed for low SNR sample → 5.0 dB
}

def create_comparison_with_fixed_seeds():
    """Create model evaluation using fixed seeds for reproducible SNR values."""
    
    print("=== FIXED SEEDS VERSION (Like Backup) ===")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pipeline
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(
        experiment_dir=experiment_dir,
        output_dir=tutorial_output_dir,
        device=device
    )
    
    # Generate samples with fixed seeds
    print("\n1. Generating fixed seed samples...")
    samples = {}
    
    for dataset_name, seed in RANDOM_SEEDS.items():
        print(f"   Generating {dataset_name} (seed={seed})...")
        
        # Generate with fixed seed (like backup)
        inputs, targets = create_data_samples(
            dataset_name=dataset_name,
            num_samples=1,
            points_per_gmm=1000,
            device=device,
            base_seed=seed,  # ← Fixed seed for reproducibility
            loader_id=f"{dataset_name}_{seed}"
        )
        
        snr_value = targets['snr_db'][0].item()
        print(f"   → SNR: {snr_value:.1f} dB")
        
        samples[dataset_name] = {
            'points': inputs[0],
            'centers': targets['centers'][0],
            'labels': targets['labels'][0],
            'snr_db': snr_value
        }
    
    # Create visualization using original model names if available
    print("\n2. Creating model evaluation with fixed samples...")
    
    # Try original models first (simple_16_layers, hard_16_layers)
    try:
        models_to_try = ['simple_16_layers', 'hard_16_layers']
        results = []
        
        for model_name in models_to_try:
            for dataset_name, label in zip(['high_snr_fixed', 'low_snr_fixed'], ['High SNR', 'Low SNR']):
                sample = samples[dataset_name]
                
                # Use the sample with the specific model
                result = pipeline._process_single_data_input(
                    data=sample,
                    models=model_name,
                    parameter_values=None,
                    show=['points', 'true_centers', 'predictions']
                )
                
                result['metadata']['title'] = f"{model_name}\n{label} ({sample['snr_db']:.1f} dB)"
                results.append(result)
        
        # Create the grid
        from scripts.evaluation.tutorial.src.visualization import create_comparison_grid, save_figure
        
        titles = [result['metadata']['title'] for result in results]
        fig, axes = create_comparison_grid(
            results=results,
            layout='2x2',
            show_predictions=True,
            show_kmeans=False,
            titles=titles,
            figsize=(12, 12)
        )
        
        save_path = tutorial_output_dir / "model_evaluation_fixed_seeds.png"
        save_figure(fig, save_path)
        print(f"   ✓ Saved fixed seeds comparison: {save_path}")
        
    except Exception as e:
        print(f"   ✗ Could not use original models: {e}")
        print("   Using baseline_16_layers instead...")
        
        # Fallback to baseline model
        results = []
        for dataset_name, label in zip(['high_snr_fixed', 'low_snr_fixed'], ['High SNR', 'Low SNR']):
            sample = samples[dataset_name]
            
            result = pipeline._process_single_data_input(
                data=sample,
                models='baseline_16_layers',
                parameter_values=None,
                show=['points', 'true_centers', 'predictions']
            )
            
            result['metadata']['title'] = f"baseline_16_layers\n{label} ({sample['snr_db']:.1f} dB)"
            results.append(result)
        
        from scripts.evaluation.tutorial.src.visualization import create_comparison_grid, save_figure
        
        titles = [result['metadata']['title'] for result in results]
        fig, axes = create_comparison_grid(
            results=results,
            layout='1x2',
            show_predictions=True,
            show_kmeans=False,
            titles=titles,
            figsize=(16, 8)
        )
        
        save_path = tutorial_output_dir / "model_evaluation_fixed_seeds_baseline.png"
        save_figure(fig, save_path)
        print(f"   ✓ Saved fixed seeds comparison: {save_path}")

def create_comparison_with_random_seeds():
    """Create model evaluation using random seeds (current implementation)."""
    
    print("\n=== RANDOM SEEDS VERSION (Current Implementation) ===")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pipeline
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(
        experiment_dir=experiment_dir,
        output_dir=tutorial_output_dir,
        device=device
    )
    
    print("\n1. Generating random samples (no seeds)...")
    
    # Use current implementation (no fixed seeds)
    fig = pipeline.scatter_plot(
        datasets=['high_snr_fixed', 'low_snr_fixed'],
        models='baseline_16_layers',
        show=['points', 'true_centers', 'predictions'],
        layout='1x2',
        titles=['High SNR (Random)', 'Low SNR (Random)'],
        figsize=(16, 8),
        save_path=tutorial_output_dir / "model_evaluation_random_seeds.png"
    )
    
    print("   ✓ Saved random seeds comparison")

def main():
    """Run the comparison between fixed and random seeds."""
    print("\n" + "="*70)
    print("FIXED SEEDS vs RANDOM SEEDS COMPARISON")
    print("="*70)
    print("\nThis script demonstrates the difference between:")
    print("1. FIXED SEEDS (backup version): Reproducible SNR values")
    print("   • high_snr_fixed + seed 90 → 15.0 dB")
    print("   • low_snr_fixed + seed 123 → 5.0 dB")
    print()
    print("2. RANDOM SEEDS (current): Variable SNR values each run")
    print("   • high_snr_fixed → random SNR around 15 dB")
    print("   • low_snr_fixed → random SNR around 5 dB")
    
    # Run both comparisons
    create_comparison_with_fixed_seeds()
    create_comparison_with_random_seeds()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {tutorial_output_dir}")
    print("\nGenerated Files:")
    print("  • model_evaluation_fixed_seeds.png - Using fixed seeds (reproducible)")
    print("  • model_evaluation_random_seeds.png - Using random seeds (variable)")
    print("\nKey Difference:")
    print("  Fixed seeds → Same SNR values every run (15.0 dB, 5.0 dB)")
    print("  Random seeds → Different SNR values each run")

if __name__ == "__main__":
    main() 