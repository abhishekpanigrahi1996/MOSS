#!/usr/bin/env python3
"""
Tutorial 09: Statistical Entropy Analysis Across Flow Regimes

This tutorial performs robust statistical analysis of entropy behavior across
different flow regimes using multiple samples from the low_snr_fixed dataset.

Features:
- 100 samples for statistical robustness
- Low SNR fixed dataset for realistic conditions
- Direct vs fractional flow distribution comparison
- No-flow vs baseline model comparison
- Standard deviation analysis (not standard error)
- Confidence interval visualization

Scientific Focus:
- Statistical significance of flow regime differences
- Entropy variability across samples
- Model architecture impact on entropy stability
- Flow distribution mode reliability
"""

import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy import stats
import math
from scipy.special import digamma

# Add project root to path for imports
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities
from scripts.evaluation.tutorial.src.io import (
    load_model_from_experiment,
    create_data_samples
)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def knn_entropy_bias_reduced_torch(X, k=3, device="cpu", B=1, subsample="random", seed=None, eps_min=1e-12):
    """
    Bias-reduced Kozachenko–Leonenko entropy estimator (2 H(n) – mean_b H_b(n/2)).
    Works on CPU or CUDA. Returns entropy in nats.
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

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = Path("scripts/evaluation/tutorial/output/entropy_statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_model_from_checkpoint(checkpoint_path):
    """Load model from experiment path."""
    print(f"Loading model from experiment: {checkpoint_path}")
    
    model, config = load_model_from_experiment(checkpoint_path, load_best=False, device='cpu')
    model.eval()
    
    print(f"Model loaded:")
    print(f"  Flow enabled: {model.transformer.use_flow_predictor}")
    print(f"  Current repeat factor: {model.transformer.repeat_factor}")
    print(f"  Current flow distribution mode: {getattr(model.transformer, 'flow_distribution_mode', 'direct')}")
    
    return model

def create_flow_regime_models(base_model_path):
    """Create models for different flow distribution modes."""
    print("\n=== Creating Flow Regime Models ===")
    print(f"Base model: {base_model_path}")
    
    # Load base model
    base_model = load_model_from_checkpoint(base_model_path)
    
    # Create direct model
    direct_model = load_model_from_checkpoint(base_model_path)
    direct_model.transformer.use_flow_predictor = False  # Manual flow control
    direct_model.transformer.flow_distribution_mode = "direct"
    
    # Create fractional model  
    fractional_model = load_model_from_checkpoint(base_model_path)
    fractional_model.transformer.use_flow_predictor = False  # Manual flow control
    fractional_model.transformer.flow_distribution_mode = "fractional"
    
    print(f"✓ Flow regime models created:")
    print(f"  Direct model: flow_distribution_mode = {direct_model.transformer.flow_distribution_mode}")
    print(f"  Fractional model: flow_distribution_mode = {fractional_model.transformer.flow_distribution_mode}")
    print(f"  Both models have repeat_factor = {direct_model.transformer.repeat_factor}")
    
    return direct_model, fractional_model

def create_evaluation_data(model_dtype, num_samples=100, points_per_sample=1000):
    """Create evaluation data using low_snr_fixed dataset."""
    print(f"\n=== Creating Evaluation Data ===")
    print(f"Generating {num_samples} samples with {points_per_sample} points each")
    print(f"Using low_snr_fixed dataset for realistic conditions")
    
    # Create data samples using the tutorial's data creation function
    data = create_data_samples(
        dataset_name='low_snr_fixed',
        num_samples=num_samples,
        points_per_gmm=points_per_sample,
        device='cpu'
    )
    
    # Unpack data
    inputs, targets = data
    
    # Convert to match model dtype
    inputs = inputs.to(model_dtype)
    targets = {k: v.to(model_dtype) if isinstance(v, torch.Tensor) else v 
               for k, v in targets.items()}
    
    # Extract key information
    centers = targets['centers']
    n_clusters = centers.shape[1]
    
    print(f"✓ Data created:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Centers shape: {centers.shape}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Dataset: low_snr_fixed (challenging conditions)")
    
    return inputs, targets

def evaluate_entropy_statistics(model, model_name, inputs, targets, flow_speeds, num_samples=100):
    """Evaluate entropy statistics across multiple samples."""
    print(f"\n=== Evaluating {model_name} Model ===")
    print(f"Flow distribution mode: {model.transformer.flow_distribution_mode}")
    print(f"Repeat factor: {model.transformer.repeat_factor}")
    print(f"Flow speed range: {flow_speeds[0]:.3f} to {flow_speeds[-1]:.3f}")
    print(f"Number of flow speeds: {len(flow_speeds)}")
    print(f"Number of samples: {num_samples}")
    
    device = next(model.parameters()).device
    
    results = {
        'flow_speeds': [],
        'entropy_means': [],
        'entropy_stds': [],
        'entropy_mins': [],
        'entropy_maxs': [],
        'entropy_all_samples': []  # Store all individual entropies for further analysis
    }
    
    with torch.no_grad():
        progress_bar = tqdm(flow_speeds, desc=f"Evaluating {model_name}")
        
        for flow_speed in progress_bar:
            # Set flow speed as tensor
            flow_tensor = torch.tensor([flow_speed], device=device, dtype=inputs.dtype)
            
            # Evaluate on all samples
            sample_entropies = []
            
            for sample_idx in range(num_samples):
                # Get sample data
                sample_input = inputs[sample_idx:sample_idx+1]  # Keep batch dimension
                sample_input = sample_input.to(device)
                
                # Forward pass with flow speed
                embeddings = model(sample_input, flow_speed=flow_tensor)
                
                # Compute entropy
                entropy = knn_entropy_bias_reduced_torch(
                    embeddings.squeeze(0), k=7, device='cpu'
                )
                
                sample_entropies.append(entropy)
            
            # Compute statistics
            sample_entropies = np.array(sample_entropies)
            entropy_mean = np.mean(sample_entropies)
            entropy_std = np.std(sample_entropies)  # Not divided by sqrt(n)
            entropy_min = np.min(sample_entropies)
            entropy_max = np.max(sample_entropies)
            
            # Store results
            results['flow_speeds'].append(flow_speed)
            results['entropy_means'].append(entropy_mean)
            results['entropy_stds'].append(entropy_std)
            results['entropy_mins'].append(entropy_min)
            results['entropy_maxs'].append(entropy_max)
            results['entropy_all_samples'].append(sample_entropies.tolist())
            
            # Progress update
            if len(results['flow_speeds']) % 10 == 0 or flow_speed == flow_speeds[0]:
                progress_bar.set_postfix({
                    'entropy': f"{entropy_mean:.3f}±{entropy_std:.3f}",
                    'range': f"[{entropy_min:.3f}, {entropy_max:.3f}]"
                })
    
    print(f"✓ {model_name} evaluation complete")
    print(f"  Entropy mean range: {np.min(results['entropy_means']):.4f} to {np.max(results['entropy_means']):.4f}")
    print(f"  Entropy std range: {np.min(results['entropy_stds']):.4f} to {np.max(results['entropy_stds']):.4f}")
    print(f"  Overall entropy range: {np.min(results['entropy_mins']):.4f} to {np.max(results['entropy_maxs']):.4f}")
    
    return results

def evaluate_entropy_statistics_snr(model, model_name, inputs, targets, snr_values, num_samples=100):
    """Evaluate entropy statistics across multiple samples as a function of SNR."""
    print(f"\n=== Evaluating {model_name} Model vs SNR ===")
    print(f"Flow distribution mode: {model.transformer.flow_distribution_mode}")
    print(f"Repeat factor: {model.transformer.repeat_factor}")
    print(f"SNR range: {snr_values[0]:.1f} to {snr_values[-1]:.1f} dB")
    print(f"Number of SNR values: {len(snr_values)}")
    print(f"Number of samples: {num_samples}")
    
    device = next(model.parameters()).device
    
    results = {
        'snr_values': [],
        'entropy_means': [],
        'entropy_stds': [],
        'entropy_mins': [],
        'entropy_maxs': [],
        'entropy_all_samples': []  # Store all individual entropies for further analysis
    }
    
    with torch.no_grad():
        progress_bar = tqdm(snr_values, desc=f"Evaluating {model_name} vs SNR")
        
        for snr_value in progress_bar:
            # Set SNR value as tensor
            snr_tensor = torch.tensor([snr_value], device=device, dtype=inputs.dtype)
            
            # Convert SNR to flow speed using the flow predictor
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'flow_predictor'):
                flow_speed = model.transformer.flow_predictor(targets={'snr_db': snr_tensor})
            else:
                # If no flow predictor, use a default flow speed
                flow_speed = torch.tensor([0.5], device=device, dtype=inputs.dtype)
            
            # Evaluate on all samples
            sample_entropies = []
            
            for sample_idx in range(num_samples):
                # Get sample data
                sample_input = inputs[sample_idx:sample_idx+1]  # Keep batch dimension
                sample_input = sample_input.to(device)
                
                # Forward pass with flow speed (converted from SNR)
                embeddings = model(sample_input, flow_speed=flow_speed)
                
                # Compute entropy
                entropy = knn_entropy_bias_reduced_torch(
                    embeddings.squeeze(0), k=7, device='cpu'
                )
                
                sample_entropies.append(entropy)
            
            # Compute statistics
            sample_entropies = np.array(sample_entropies)
            entropy_mean = np.mean(sample_entropies)
            entropy_std = np.std(sample_entropies)  # Not divided by sqrt(n)
            entropy_min = np.min(sample_entropies)
            entropy_max = np.max(sample_entropies)
            
            # Store results
            results['snr_values'].append(snr_value)
            results['entropy_means'].append(entropy_mean)
            results['entropy_stds'].append(entropy_std)
            results['entropy_mins'].append(entropy_min)
            results['entropy_maxs'].append(entropy_max)
            results['entropy_all_samples'].append(sample_entropies.tolist())
            
            # Progress update
            if len(results['snr_values']) % 5 == 0 or snr_value == snr_values[0]:
                progress_bar.set_postfix({
                    'snr': f"{snr_value:.1f}dB",
                    'flow': f"{flow_speed.item():.3f}",
                    'entropy': f"{entropy_mean:.3f}±{entropy_std:.3f}",
                    'range': f"[{entropy_min:.3f}, {entropy_max:.3f}]"
                })
    
    print(f"✓ {model_name} SNR evaluation complete")
    print(f"  Entropy mean range: {np.min(results['entropy_means']):.4f} to {np.max(results['entropy_means']):.4f}")
    print(f"  Entropy std range: {np.min(results['entropy_stds']):.4f} to {np.max(results['entropy_stds']):.4f}")
    print(f"  Overall entropy range: {np.min(results['entropy_mins']):.4f} to {np.max(results['entropy_maxs']):.4f}")
    
    return results

def create_statistical_plots(results_dict, snr_results_dict=None, save_plots=True):
    """Create simplified statistical plots with continuous std intervals."""
    print(f"\n=== Creating Statistical Plots ===")
    
    output_dir = ensure_output_directory()
    
    # Set up plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Colors for different conditions
    colors = {
        'direct': '#1f77b4',
        'fractional': '#ff7f0e'
    }
    
    # Separate plots for each model
    models = {'no_flow_16_layers': 'No-Flow Model (Manual Flow Injection)',
              'baseline_16_layers': 'Baseline Model (Flow Substitution)'}
    
    for model_key, model_title in models.items():
        # Filter results for this model
        model_results = {k: v for k, v in results_dict.items() if model_key in k}
        
        if not model_results:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for condition, results in model_results.items():
            flow_speeds = np.array(results['flow_speeds'])
            means = np.array(results['entropy_means'])
            stds = np.array(results['entropy_stds'])
            
            # Calculate upper and lower bounds
            ent_low = means - stds
            ent_high = means + stds
            
            # Determine mode
            mode = 'direct' if 'direct' in condition else 'fractional'
            color = colors[mode]
            label = f"{mode.title()} Mode"
            
            # Plot mean line
            ax.plot(flow_speeds, means, color=color, linewidth=2, label=label)
            
            # Fill between for std interval
            ax.fill_between(flow_speeds, ent_low, ent_high, 
                           color=color, alpha=0.3, label=f"{mode.title()} ±1σ")
        
        # Format plot
        ax.set_xlabel('Flow Speed', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title(f'{model_title}\n(100 Samples, Low SNR Fixed Dataset)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = output_dir / f"entropy_vs_flow_{model_key}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ {model_title} plot saved: {plot_path}")
        
        plt.show()
    
    # SNR plots for baseline model only
    if snr_results_dict:
        print(f"\n=== Creating SNR Dependence Plots ===")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for condition, results in snr_results_dict.items():
            snr_values = np.array(results['snr_values'])
            means = np.array(results['entropy_means'])
            stds = np.array(results['entropy_stds'])
            
            # Calculate upper and lower bounds
            ent_low = means - stds
            ent_high = means + stds
            
            # Determine mode
            mode = 'direct' if 'direct' in condition else 'fractional'
            color = colors[mode]
            label = f"{mode.title()} Mode"
            
            # Plot mean line
            ax.plot(snr_values, means, color=color, linewidth=2, label=label)
            
            # Fill between for std interval
            ax.fill_between(snr_values, ent_low, ent_high, 
                           color=color, alpha=0.3, label=f"{mode.title()} ±1σ")
        
        # Format plot
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('Baseline Model: Entropy vs SNR\n(100 Samples, Low SNR Fixed Dataset)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = output_dir / "entropy_vs_snr_baseline_16_layers.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Baseline SNR dependence plot saved: {plot_path}")
        
        plt.show()
    
    # Combined plot showing all models and modes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Combined colors for all conditions
    combined_colors = {
        'no_flow_16_layers_direct': '#1f77b4',
        'no_flow_16_layers_fractional': '#4d94ff',
        'baseline_16_layers_direct': '#ff7f0e', 
        'baseline_16_layers_fractional': '#ffb366'
    }
    
    for condition, results in results_dict.items():
        flow_speeds = np.array(results['flow_speeds'])
        means = np.array(results['entropy_means'])
        stds = np.array(results['entropy_stds'])
        
        # Calculate upper and lower bounds
        ent_low = means - stds
        ent_high = means + stds
        
        # Parse condition for label
        if 'no_flow' in condition:
            model_name = 'No-Flow'
        else:
            model_name = 'Baseline'
        
        mode = 'Direct' if 'direct' in condition else 'Fractional'
        label = f"{model_name} ({mode})"
        
        color = combined_colors.get(condition, '#000000')
        
        # Plot mean line
        ax.plot(flow_speeds, means, color=color, linewidth=2, label=label)
        
        # Fill between for std interval
        ax.fill_between(flow_speeds, ent_low, ent_high, 
                       color=color, alpha=0.2)
    
    # Format combined plot
    ax.set_xlabel('Flow Speed', fontsize=12)
    ax.set_ylabel('Entropy (nats)', fontsize=12)
    ax.set_title('Entropy vs Flow Speed - All Models and Modes\n(100 Samples, Low SNR Fixed Dataset)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = output_dir / "entropy_vs_flow_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined plot saved: {plot_path}")
    
    plt.show()

def analyze_model_comparison(base_experiment_names, num_samples=100):
    """Analyze multiple models with statistical robustness."""
    
    results_dict = {}
    snr_results_dict = {}
    
    for base_experiment_name, description in base_experiment_names.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING BASE MODEL: {base_experiment_name}")
        print(f"Description: {description}")
        print(f"{'='*80}")
        
        base_model_path = f"/mount/Storage/gmm-v4/output/final_experiments/{base_experiment_name}"
        
        if not Path(base_model_path).exists():
            print(f"❌ Model path not found: {base_model_path}")
            continue
        
        print(f"Using base model: {base_model_path}")
        
        # Create flow regime models
        direct_model, fractional_model = create_flow_regime_models(base_model_path)
        
        # Create evaluation data
        inputs, targets = create_evaluation_data(
            model_dtype=next(direct_model.parameters()).dtype,
            num_samples=num_samples,
            points_per_sample=1000
        )
        
        # Define flow speed range (0 to 1)
        flow_speeds = np.linspace(0.0, 1.0, 50)
        print(f"Flow speeds: {flow_speeds[0]:.3f} to {flow_speeds[-1]:.3f} ({len(flow_speeds)} points)")
        
        # Evaluate both models for flow dependence
        direct_results = evaluate_entropy_statistics(
            direct_model, f"{base_experiment_name}_direct", 
            inputs, targets, flow_speeds, num_samples
        )
        
        fractional_results = evaluate_entropy_statistics(
            fractional_model, f"{base_experiment_name}_fractional", 
            inputs, targets, flow_speeds, num_samples
        )
        
        # Store results
        results_dict[f"{base_experiment_name}_direct"] = direct_results
        results_dict[f"{base_experiment_name}_fractional"] = fractional_results
        
        # For baseline model, also analyze SNR dependence
        if base_experiment_name == 'baseline_16_layers':
            print(f"\n{'='*60}")
            print(f"SNR DEPENDENCE ANALYSIS FOR {base_experiment_name}")
            print(f"{'='*60}")
            
            # Define SNR range (3 to 15 dB)
            snr_values = np.linspace(3.0, 15.0, 25)
            print(f"SNR values: {snr_values[0]:.1f} to {snr_values[-1]:.1f} dB ({len(snr_values)} points)")
            
            # Evaluate both models for SNR dependence
            direct_snr_results = evaluate_entropy_statistics_snr(
                direct_model, f"{base_experiment_name}_direct", 
                inputs, targets, snr_values, num_samples
            )
            
            fractional_snr_results = evaluate_entropy_statistics_snr(
                fractional_model, f"{base_experiment_name}_fractional", 
                inputs, targets, snr_values, num_samples
            )
            
            # Store SNR results
            snr_results_dict[f"{base_experiment_name}_direct"] = direct_snr_results
            snr_results_dict[f"{base_experiment_name}_fractional"] = fractional_snr_results
        
        print(f"✓ Analysis complete for {base_experiment_name}")
    
    return results_dict, snr_results_dict

def main():
    """Main tutorial execution."""
    print("="*80)
    print("TUTORIAL 09: STATISTICAL ENTROPY ANALYSIS ACROSS FLOW REGIMES")
    print("="*80)
    print()
    print("This tutorial demonstrates:")
    print("1. Statistical analysis with continuous std intervals")
    print("2. Low SNR fixed dataset for realistic conditions")
    print("3. Separate plots for each model")
    print("4. Combined comparison plot")
    print("5. Robust analysis with 100 samples")
    print()
    
    # Create output directory
    output_dir = ensure_output_directory()
    
    # Define models to analyze
    base_experiments = {
        'no_flow_16_layers': 'No-Flow Model (Manual Flow Injection)',
        'baseline_16_layers': 'Baseline Model (Flow Substitution)'
    }
    
    # Perform statistical analysis with 100 samples for robustness
    results_dict, snr_results_dict = analyze_model_comparison(base_experiments, num_samples=100)
    
    if not results_dict:
        print("❌ No results generated. Check model paths.")
        return
    
    # Create simplified statistical plots
    create_statistical_plots(results_dict, snr_results_dict, save_plots=True)
    
    print(f"\n{'='*80}")
    print("TUTORIAL 09 COMPLETE!")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print()
    
    # List generated files
    files = list(output_dir.glob("*.png"))
    files.sort()
    
    print("Generated files:")
    for file in files:
        if file.is_file():
            print(f"  📄 {file.name}")
    
    print()
    print(f"📊 Files summary:")
    png_files = len(files)
    print(f"   📈 Plots: {png_files} PNG files")
    
    print()
    print("💡 Key Insights:")
    print("   • Continuous std intervals show entropy variability patterns")
    print("   • Separate plots clearly distinguish model behaviors")
    print("   • Combined plot enables direct comparison")
    print("   • 100 samples provide robust statistical foundation")

if __name__ == "__main__":
    main() 