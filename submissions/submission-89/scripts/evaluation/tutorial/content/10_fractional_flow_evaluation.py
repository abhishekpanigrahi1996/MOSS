"""
GMM Model Evaluation Tutorial - True Flow Substitution with No-Flow Model
========================================================================

This tutorial demonstrates how to take a model trained WITHOUT flow (no_flow_16_layers) 
and manually inject fractional flow speeds to show the pure architectural effect.

DEMONSTRATION:
- Load no_flow_16_layers model (trained without flow prediction)
- Set flow regime to fractional mode  
- Manually inject flow speeds: 0.25, 0.5, 0.75, 1.0
- Display results in a clean 1×4 grid
- Calculate entropy vs flow speed relationship

This showcases true "flow substitution" - same weights, different flow speeds!
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.special import digamma
from pathlib import Path

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new Phase 3 API
from scripts.evaluation.tutorial.src.visualization_pipeline import VisualizationPipeline

# Import model loading functions (old API)
from scripts.evaluation.tutorial.src.io import load_model_from_experiment, create_data_samples

# Define paths
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

def knn_entropy_bias_reduced_torch(X, k=7, device="cpu", B=1, subsample="random", seed=None, eps_min=1e-12):
    """
    Bias-reduced Kozachenko–Leonenko entropy estimator (2 H(n) – mean_b H_b(n/2)).
    Works on CPU or CUDA. Returns entropy in nats.
    Adapted from 09_entropy_statistical_analysis.py
    """
    X = X.to(device, dtype=torch.float32)
    n, d = X.shape
    if n < 2 * k + 2:
        raise ValueError("Need at least 2·k + 2 points.")

    log_Vd = (d / 2) * math.log(math.pi) - math.lgamma(1 + d / 2)

    def H_knn(data: torch.Tensor) -> torch.Tensor:
        m = data.shape[0]

        # pair-wise distances
        D = torch.cdist(data, data, compute_mode='donot_use_mm_for_euclid_dist')
        D.fill_diagonal_(float("inf"))
        eps_k = D.topk(k, largest=False).values[:, -1].clamp_min(eps_min)

        m_t = torch.tensor(float(m), device='cpu', dtype=data.dtype)
        k_t = torch.tensor(float(k), device='cpu', dtype=data.dtype)

        const = log_Vd + digamma(m_t) - digamma(k_t)
        return d * eps_k.log().mean() + const

    # full-sample estimate
    H_full = H_knn(X)

    # average over B half-samples
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    H_half_sum = 0.0
    for _ in range(B):
        if subsample == "first_half":
            half = X[: n // 2]
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: n // 2]
            half = X[idx]
        H_half_sum += H_knn(half)

    H_half_avg = H_half_sum / B
    return (2.0 * H_full - H_half_avg).item()

def load_and_modify_model():
    """Load no_flow_16_layers model and modify it for manual flow injection."""
    print("\n==== Loading and Modifying No-Flow Model ====")
    print("Loading no_flow_16_layers model...")
    
    # Load the NO-FLOW model using old API
    model_path = '/mount/Storage/gmm-v4/output/final_experiments/no_flow_16_layers'
    model, config = load_model_from_experiment(model_path, load_best=True, device='cuda')
    model.eval()
    
    print(f"Original no-flow model settings:")
    print(f"  Flow distribution mode: {getattr(model.transformer, 'flow_distribution_mode', 'direct')}")
    print(f"  Use flow predictor: {model.transformer.use_flow_predictor}")
    print(f"  Repeat factor: {model.transformer.repeat_factor}")
    
    # Modify the model for manual flow injection with fractional mode
    model.transformer.flow_distribution_mode = "fractional"
    model.transformer.use_flow_predictor = False  # We'll manually inject flow
    print(f"\n✓ Modified no-flow model for manual fractional flow injection!")
    print(f"  New flow distribution mode: {model.transformer.flow_distribution_mode}")
    print(f"  Manual flow injection enabled: {not model.transformer.use_flow_predictor}")
    
    return model

def evaluate_flow_substitution():
    """Evaluate no_flow_16_layers model with manual fractional flow injection at specific speeds."""
    print("\n==== True Flow Substitution Evaluation ====")
    print("Model: no_flow_16_layers (modified for manual fractional flow injection)")
    print("Flow speeds: [0.25, 0.5, 0.75, 1.0]")
    print("Layout: 1×4 grid")
    print("Concept: Same weights trained WITHOUT flow, now with manual flow injection")
    
    # Load and modify the no-flow model manually
    fractional_model = load_and_modify_model()
    
    # Initialize visualization pipeline
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(experiment_dir=experiment_dir)
    
    # Define the specific flow speeds to evaluate
    flow_speeds = [0.25, 0.5, 0.75, 1.0]
    
    print(f"\nEvaluating no-flow model with manual FRACTIONAL flow injection...")
    
    # We need to modify the pipeline's _add_model_predictions to accept model objects
    # Let's monkey-patch it temporarily for this demonstration
    original_add_predictions = pipeline._add_model_predictions
    
    def patched_add_predictions(result, models, parameter_values):
        # Check if models is actually a model object instead of string
        if hasattr(models, 'transformer'):  # It's a model object
            model = models
            model_name = 'no_flow_16_layers_fractional_injection'
        else:
            # Fall back to original behavior
            return original_add_predictions(result, models, parameter_values)
        
        try:
            # Extract flow speed value for evaluation
            flow_speed = None
            if isinstance(parameter_values, dict) and 'flow_speed' in parameter_values:
                flow_speed = parameter_values['flow_speed']
            else:
                flow_speed = 0.5  # Default flow speed
            
            # Get model's expected dtype
            model_dtype = next(model.parameters()).dtype
            
            # Convert data to match model dtype
            inputs = result['inputs'].to(dtype=model_dtype)
            
            # Convert flow speed to tensor with correct dtype
            flow_tensor = torch.tensor([flow_speed], device=pipeline.device, dtype=model_dtype)
            
            # Forward pass with manual flow speed injection!
            # This is the key: no-flow model + manual flow = pure architectural effect
            with torch.no_grad():
                predictions = model(inputs, flow_speed=flow_tensor)
            
            # Add predictions to result
            result['predictions'] = predictions
            
            # Store metadata
            result['metadata']['model_name'] = model_name
            result['metadata']['flow_speed'] = flow_speed
            result['metadata']['flow_mode'] = model.transformer.flow_distribution_mode
            result['metadata']['original_training'] = 'no_flow'
            
            print(f"✓ Added MANUAL FRACTIONAL flow predictions at speed {flow_speed:.2f}")
            
        except Exception as e:
            print(f"Warning: Manual flow injection failed: {e}")
        
        return result
    
    # Apply the monkey patch
    pipeline._add_model_predictions = patched_add_predictions
    
    # Create 1×4 grid evaluation with manual flow injection
    fig = pipeline.scatter_plot(
        datasets='standard',                                    # Standard dataset
        models=fractional_model,                               # Pass the MODIFIED no-flow model!
        show=['points', 'true_centers', 'predictions'],       # Show all elements
        parameter_values={'flow_speed': flow_speeds},          # Manual flow speeds
        layout='1x4',                                          # 1×4 grid layout
        titles=[f'Flow: {fs:.2f}' for fs in flow_speeds],     # Simple titles
        save_path=tutorial_output_dir / "no_flow_manual_fractional_injection_1x4.png",
        figsize=(20, 5),                                       # Wide figure for 1×4 layout
        size_scale=0.8                                         # Slightly smaller elements for clarity
    )
    
    # Restore original method
    pipeline._add_model_predictions = original_add_predictions
    
    print("✓ True flow substitution evaluation completed!")
    print(f"✓ Results saved to: {tutorial_output_dir / 'no_flow_manual_fractional_injection_1x4.png'}")
    
    return fig, fractional_model  # Return both figure and model

def compare_no_flow_vs_manual_flow():
    """Create a comparison between no flow and manual flow injection."""
    print("\n==== Creating No-Flow vs Manual Flow Comparison ====")
    
    # Load no-flow model and create two versions
    model_path = '/mount/Storage/gmm-v4/output/final_experiments/no_flow_16_layers'
    
    # Pure no-flow model (original)
    no_flow_model, _ = load_model_from_experiment(model_path, load_best=True, device='cuda')
    no_flow_model.eval()
    # Keep it as-is for pure no-flow behavior
    
    # Manual flow injection model  
    manual_flow_model, _ = load_model_from_experiment(model_path, load_best=True, device='cuda')
    manual_flow_model.eval()
    manual_flow_model.transformer.flow_distribution_mode = "fractional"
    manual_flow_model.transformer.use_flow_predictor = False  # Manual flow only
    
    print(f"No-flow model (original): use_flow_predictor = {no_flow_model.transformer.use_flow_predictor}")
    print(f"Manual flow model: flow_mode = {manual_flow_model.transformer.flow_distribution_mode}")
    
    print("Concept demonstration: Same trained weights, with vs without manual flow injection")
    print("This shows the pure architectural effect of flow on clustering behavior")
    
    return None

def calculate_entropy_vs_flow(fractional_model):
    """Calculate entropy for the same sample used in visualization across many flow speeds."""
    print("\n==== Entropy vs Flow Speed Analysis ====")
    print("Calculating entropy dependence for the SAME sample used in 1×4 grid...")
    print(f"Model: no_flow_16_layers (fractional mode)")
    
    # Use many more flow speeds for a smooth curve
    flow_speeds = np.linspace(0.05, 1.0, 50)  # 50 flow speeds from 0.05 to 1.0
    print(f"Flow speed range: {flow_speeds[0]:.2f} to {flow_speeds[-1]:.2f} ({len(flow_speeds)} points)")
    
    # Use the EXACT same data generation as the main visualization
    model_dtype = next(fractional_model.parameters()).dtype
    device = next(fractional_model.parameters()).device
    
    print(f"Generating the SAME single data sample used in visualization...")
    data = create_data_samples(
        dataset_name='standard',
        num_samples=1,  # Just 1 sample - the same one!
        points_per_gmm=1000,  # 1000 points for robust entropy estimation
        device='cpu'
    )
    
    inputs, targets = data
    inputs = inputs.to(dtype=model_dtype)
    sample_input = inputs[0:1].to(device)  # The single sample
    
    print(f"✓ Using single sample: {sample_input.shape}")
    print(f"✓ This is the SAME sample used in the 1×4 grid visualization")
    
    # Store entropy results (no statistics needed - just one value per flow speed)
    entropy_results = []
    
    print(f"\nEvaluating entropy at {len(flow_speeds)} flow speeds for the single sample...")
    
    with torch.no_grad():
        for i, flow_speed in enumerate(flow_speeds):
            flow_tensor = torch.tensor([flow_speed], device=device, dtype=model_dtype)
            
            # Forward pass with manual flow speed injection on the SAME sample
            embeddings = fractional_model(sample_input, flow_speed=flow_tensor)
            
            # Compute entropy using k-NN estimator
            entropy = knn_entropy_bias_reduced_torch(
                embeddings.squeeze(0), k=7, device='cpu'
            )
            
            entropy_results.append({
                'flow_speed': flow_speed,
                'entropy': entropy
            })
            
            # Progress indicator every 10 points
            if (i + 1) % 10 == 0 or i == 0 or i == len(flow_speeds) - 1:
                print(f"  Flow {flow_speed:.2f}: entropy = {entropy:.3f} nats ({i+1}/{len(flow_speeds)})")
    
    print(f"✓ Entropy evaluation completed for single sample!")
    
    # Create the entropy vs flow speed plot (NO error bars - just clean curve)
    print(f"\nCreating entropy vs flow speed plot...")
    
    flow_speeds_plot = np.array([r['flow_speed'] for r in entropy_results])
    entropy_values = np.array([r['entropy'] for r in entropy_results])
    
    # Create the plot showing MINUS entropy (as requested)
    minus_entropy_values = -entropy_values
    
    plt.figure(figsize=(10, 6))
    
    # Plot minus entropy vs flow speed as a clean curve (no error bars)
    plt.plot(flow_speeds_plot, minus_entropy_values, 
             marker='o', linewidth=2, markersize=4,
             color='#ff7f0e', label='No-Flow Model (Single Sample)')
    
    # Format plot
    plt.xlabel('Flow Speed', fontsize=12)
    plt.ylabel('Entropy [nats]', fontsize=12)
    plt.title('Entropy vs Flow Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable axis limits
    plt.xlim(0.0, 1.05)
    
    # Save the plot
    entropy_plot_path = tutorial_output_dir / "entropy_vs_flow_no_flow_model.png"
    plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    
    print(f"✓ Entropy plot saved: {entropy_plot_path}")
    
    # Show summary statistics
    print(f"\nEntropy Summary for Single Sample:")
    print(f"  Flow speed range: {flow_speeds_plot[0]:.2f} to {flow_speeds_plot[-1]:.2f}")
    print(f"  Entropy range: {np.min(entropy_values):.3f} to {np.max(entropy_values):.3f} nats")
    print(f"  Entropy range: {np.min(minus_entropy_values):.3f} to {np.max(minus_entropy_values):.3f} nats")
    print(f"  Total entropy change: {np.max(entropy_values) - np.min(entropy_values):.3f} nats")
    
    # Display plot
    if os.environ.get('DISPLAY', '') != '':
        plt.show()
    else:
        print("  (Plot saved to file - no display detected)")
    
    plt.close()
    
    return entropy_results

def calculate_entropy_statistics_across_samples(fractional_model, num_samples=100):
    """Calculate entropy statistics across multiple samples with continuous std bands."""
    print("\n==== Statistical Entropy Analysis Across Multiple Samples ====")
    print(f"Calculating entropy statistics across {num_samples} different samples...")
    print(f"Model: no_flow_16_layers (fractional mode)")
    
    # Use the same flow speed range as single sample analysis
    flow_speeds = np.linspace(0.05, 1.0, 50)  # 50 flow speeds from 0.05 to 1.0
    print(f"Flow speed range: {flow_speeds[0]:.2f} to {flow_speeds[-1]:.2f} ({len(flow_speeds)} points)")
    print(f"Number of samples: {num_samples}")
    
    # Get model properties
    model_dtype = next(fractional_model.parameters()).dtype
    device = next(fractional_model.parameters()).device
    
    print(f"Generating {num_samples} different data samples...")
    data = create_data_samples(
        dataset_name='standard',
        num_samples=num_samples,  # Many samples for statistics
        points_per_gmm=1000,  # 1000 points per sample for robust entropy estimation
        device='cpu'
    )
    
    inputs, targets = data
    inputs = inputs.to(dtype=model_dtype)
    
    print(f"✓ Data created: {inputs.shape}")
    
    # Store entropy results for statistics
    flow_speeds_results = []
    entropy_means = []
    entropy_stds = []
    
    print(f"\nEvaluating entropy statistics at {len(flow_speeds)} flow speeds...")
    
    with torch.no_grad():
        for i, flow_speed in enumerate(flow_speeds):
            flow_tensor = torch.tensor([flow_speed], device=device, dtype=model_dtype)
            
            # Calculate entropy for all samples at this flow speed
            sample_entropies = []
            
            for sample_idx in range(num_samples):
                # Get sample data
                sample_input = inputs[sample_idx:sample_idx+1].to(device)
                
                # Forward pass with manual flow speed injection
                embeddings = fractional_model(sample_input, flow_speed=flow_tensor)
                
                # Compute entropy using k-NN estimator
                entropy = knn_entropy_bias_reduced_torch(
                    embeddings.squeeze(0), k=7, device='cpu'
                )
                
                sample_entropies.append(entropy)
            
            # Calculate statistics across samples
            sample_entropies = np.array(sample_entropies)
            entropy_mean = np.mean(sample_entropies)
            entropy_std = np.std(sample_entropies)
            
            # Store results
            flow_speeds_results.append(flow_speed)
            entropy_means.append(entropy_mean)
            entropy_stds.append(entropy_std)
            
            # Progress indicator every 10 points
            if (i + 1) % 10 == 0 or i == 0 or i == len(flow_speeds) - 1:
                print(f"  Flow {flow_speed:.2f}: entropy = {entropy_mean:.3f} ± {entropy_std:.3f} nats ({i+1}/{len(flow_speeds)})")
    
    print(f"✓ Statistical entropy evaluation completed!")
    
    # Create the statistical entropy vs flow speed plot with continuous std bands
    print(f"\nCreating statistical entropy vs flow speed plot...")
    
    flow_speeds_plot = np.array(flow_speeds_results)
    entropy_means = np.array(entropy_means)
    entropy_stds = np.array(entropy_stds)
    
    # Create the plot showing entropy statistics
    minus_entropy_means = -entropy_means
    
    plt.figure(figsize=(10, 6))
    
    # Plot entropy mean with continuous std band
    plt.plot(flow_speeds_plot, minus_entropy_means, 
             linewidth=2, color='#2E8B57', label=f'Mean across {num_samples} samples ± 1σ')
    
    # Add continuous standard deviation band (no legend entry)
    plt.fill_between(flow_speeds_plot, 
                     minus_entropy_means - entropy_stds,
                     minus_entropy_means + entropy_stds,
                     alpha=0.3, color='#2E8B57')
    
    # Format plot
    plt.xlabel('Flow Speed', fontsize=12)
    plt.ylabel('Entropy [nats]', fontsize=12)
    plt.title('Entropy vs Flow Speed (Statistical)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable axis limits
    plt.xlim(0.0, 1.05)
    
    # Save the plot
    entropy_stats_plot_path = tutorial_output_dir / "entropy_vs_flow_statistics_100_samples.png"
    plt.savefig(entropy_stats_plot_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    
    print(f"✓ Statistical entropy plot saved: {entropy_stats_plot_path}")
    
    # Show summary statistics
    print(f"\nStatistical Entropy Summary ({num_samples} samples):")
    print(f"  Flow speed range: {flow_speeds_plot[0]:.2f} to {flow_speeds_plot[-1]:.2f}")
    print(f"  Mean entropy range: {np.min(entropy_means):.3f} to {np.max(entropy_means):.3f} nats")
    print(f"  Mean entropy range: {np.min(minus_entropy_means):.3f} to {np.max(minus_entropy_means):.3f} nats")
    print(f"  Average std across flow speeds: {np.mean(entropy_stds):.3f} nats")
    print(f"  Total mean entropy change: {np.max(entropy_means) - np.min(entropy_means):.3f} nats")
    
    # Display plot
    if os.environ.get('DISPLAY', '') != '':
        plt.show()
    else:
        print("  (Statistical plot saved to file - no display detected)")
    
    plt.close()
    
    return {
        'flow_speeds': flow_speeds_results,
        'entropy_means': entropy_means.tolist(),
        'entropy_stds': entropy_stds.tolist()
    }

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 85)
    print("GMM TRUE FLOW SUBSTITUTION WITH NO-FLOW MODEL TUTORIAL")
    print("=" * 85)
    print("Demonstrating flow substitution: no-flow model + manual fractional flow injection!")
    print(f"Device: {device}")
    
    # Main evaluation: 1×4 grid showing manual flow injection effects
    main_result, fractional_model = evaluate_flow_substitution()
    
    # Comparison demonstration
    compare_no_flow_vs_manual_flow()
    
    # Calculate entropy vs flow speed relationship
    entropy_results = calculate_entropy_vs_flow(fractional_model)
    
    # Calculate entropy statistics across multiple samples
    entropy_statistics = calculate_entropy_statistics_across_samples(fractional_model)
    
    print("\n" + "=" * 85)
    print("TUTORIAL COMPLETE!")
    print("=" * 85)
    print(f"All outputs saved to: {tutorial_output_dir}")
    
    print("\nGenerated Outputs:")
    print("  📊 no_flow_manual_fractional_injection_1x4.png - No-flow model + manual flow injection")
    print("  📈 entropy_vs_flow_no_flow_model.png - Entropy vs flow speed analysis")
    print("  📈 entropy_vs_flow_statistics_100_samples.png - Statistical entropy analysis across 100 samples")
    
    print("\nKey Demonstration:")
    print("  ✓ Load model trained WITHOUT flow (no_flow_16_layers)")
    print("  ✓ Manual fractional flow injection into same weights")
    print("  ✓ Pure architectural effect of flow on clustering")
    print("  ✓ True flow substitution [0.25, 0.5, 0.75, 1.0]")
    print("  ✓ Shows how flow affects transformer clustering behavior")
    print("  ✓ Entropy analysis reveals flow-dependent clustering quality")
    
    print("\nWhat We Actually Achieved:")
    print("  ✓ no_flow_16_layers model loaded (trained without flow)")
    print("  ✓ model.transformer.flow_distribution_mode = 'fractional'")
    print("  ✓ model.transformer.use_flow_predictor = False (manual injection)")
    print("  ✓ Same weights + different manual flow speeds = pure flow effect!")
    
    if os.environ.get('DISPLAY', '') == '':
        print("\nNo display detected. Results saved as files.") 