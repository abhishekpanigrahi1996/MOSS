"""
GMM Model Evaluation Tutorial - Part 11: Model Performance Comparison
====================================================================

This tutorial compares the performance of different models:
- Baseline models with different layer counts (16, 32, 64)
- K-means clustering baseline
- No-flow model

We'll evaluate each model on the same dataset and visualize:
1. Log Wasserstein distance distributions
2. Average loss as a function of SNR
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 11: Model Performance Comparison")
print("=" * 70)
print()

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities
from scripts.evaluation.tutorial.src.io import (
    load_model_from_experiment,
    create_data_loader
)
from scripts.evaluation.tutorial.src.eval_utils import (
    evaluate_dataset
)
from scripts.evaluation.tutorial.src.visualization import (
    set_plotting_style,
    format_axis_with_grid,
    format_legend,
    save_figure
)

# Set plotting style
set_plotting_style()

# ================================================
# LOESS Functions for Conditional Expectation Estimation
# ================================================
def estimate_conditional_expectation(snr_values, log_wasserstein_values, model_name):
    """Estimate E[log_wasserstein | SNR] using LOESS with fixed frac=0.1."""
    print(f"\nEstimating conditional expectation for {model_name}")
    print("-" * (35 + len(model_name)))
    
    # Use fixed frac = 0.1
    frac = 0.1
    
    # Grid for smooth curves
    snr_grid = np.linspace(snr_values.min(), snr_values.max(), 50)
    
    # Compute LOESS fit
    loess_fit = lowess(log_wasserstein_values, snr_values, frac=frac, return_sorted=True)
    conditional_expectation = np.interp(snr_grid, loess_fit[:, 0], loess_fit[:, 1])
    
    print(f"  LOESS fit computed with frac={frac}")
    
    return snr_grid, conditional_expectation, frac

print("LOESS Conditional Expectation Estimation")
print("-" * 40)
print("This adds smooth conditional expectation curves E[log_wasserstein | SNR]")
print("using fixed smoothing parameter frac=0.1 for each model.")
print()

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('/mount/Storage/gmm-v4/output')
experiment_base_dir = output_dir / 'final_experiments'
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output/performance_comparison')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configuration
dataset_name = "diverse_snr_moderate"  # Uses moderate clusters with diverse SNR
batch_size = 1
total_samples = 2**9  # 512 samples (reduced for faster execution)

print("Evaluation Configuration")
print("-" * 24)
print(f"Dataset: {dataset_name}")
print(f"Total samples: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Device: {device}")
print()

# Define models to evaluate
model_configs = {
    "baseline_16_layers": {"name": "Baseline 16L", "path": "baseline_16_layers", "layers": 16},
    "baseline_32_layers": {"name": "Baseline 32L", "path": "baseline_32_layers", "layers": 32},
    "baseline_64_layers": {"name": "Baseline 64L", "path": "baseline_64_layers", "layers": 64},
    "no_flow_16_layers": {"name": "No Flow 16L", "path": "no_flow_16_layers", "layers": 16},
}

def evaluate_models(model_configs, dataset_name, total_samples, batch_size, device):
    """Evaluate all models and collect results."""
    results = {}
    
    print("\nEvaluating Models")
    print("-" * 17)
    
    # Create data loader
    data_loader = create_data_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        total_samples=total_samples,
        device=device
    )
    
    # Evaluate each model
    for model_key, config in tqdm(model_configs.items(), desc="Evaluating models"):
        print(f"\nEvaluating {config['name']}...")
        print(f"  Model path: {experiment_base_dir / config['path']}")
        
        try:
            # Load model
            model_path = experiment_base_dir / config["path"]
            model, _ = load_model_from_experiment(model_path, load_best=False, device=device)
            print(f"  ✓ Model loaded successfully")
            
            # Evaluate dataset with metrics
            print(f"  Evaluating on {len(data_loader)} batches...")
            eval_results_list = evaluate_dataset(
                model, 
                data_loader, 
                kmeans_on_inputs=True,  # Also compute K-means on inputs
                kmeans_on_predictions=True,  # Also compute K-means on predictions
                metrics=['log_wasserstein', 'log_kmeans_wasserstein', 'log_pred_kmeans_wasserstein'],
                device=device
            )
            
            # Aggregate results from all batches
            print(f"  Aggregating results from {len(eval_results_list)} batches...")
            all_log_wass = []
            all_snr_db = []
            all_log_kmeans_wass = []
            all_log_pred_kmeans_wass = []
            
            for i, batch_results in enumerate(eval_results_list):
                if 'metrics' in batch_results and 'log_wasserstein' in batch_results['metrics']:
                    batch_log_wass = batch_results['metrics']['log_wasserstein'].cpu().numpy()
                    all_log_wass.extend(batch_log_wass)
                if 'snr_values' in batch_results and batch_results['snr_values'] is not None:
                    batch_snr = batch_results['snr_values'].cpu().numpy()
                    all_snr_db.extend(batch_snr)
                if 'metrics' in batch_results and 'log_kmeans_wasserstein' in batch_results['metrics']:
                    batch_kmeans_wass = batch_results['metrics']['log_kmeans_wasserstein'].cpu().numpy()
                    all_log_kmeans_wass.extend(batch_kmeans_wass)
                if 'metrics' in batch_results and 'log_pred_kmeans_wasserstein' in batch_results['metrics']:
                    batch_pred_kmeans_wass = batch_results['metrics']['log_pred_kmeans_wasserstein'].cpu().numpy()
                    all_log_pred_kmeans_wass.extend(batch_pred_kmeans_wass)
            
            results[model_key] = {
                'log_wasserstein': np.array(all_log_wass),
                'snr_db': np.array(all_snr_db),
                'log_kmeans_wasserstein': np.array(all_log_kmeans_wass) if all_log_kmeans_wass else None,
                'log_pred_kmeans_wasserstein': np.array(all_log_pred_kmeans_wass) if all_log_pred_kmeans_wass else None
            }
            
            # Print detailed summary statistics
            log_wass_array = results[model_key]['log_wasserstein']
            snr_array = results[model_key]['snr_db']
            print(f"  ✓ Results aggregated:")
            print(f"    - Log Wasserstein: Mean={log_wass_array.mean():.3f}, Std={log_wass_array.std():.3f}")
            print(f"    - Log Wasserstein range: [{log_wass_array.min():.3f}, {log_wass_array.max():.3f}]")
            print(f"    - SNR range: [{snr_array.min():.1f}, {snr_array.max():.1f}] dB")
            print(f"    - Total evaluations: {len(log_wass_array)}")
            
        except Exception as e:
            print(f"  Error evaluating {config['name']}: {e}")
            results[model_key] = None
    
    # Extract K-means results from one of the models (they're all the same)
    print("\nExtracting K-means baseline results...")
    kmeans_results = None
    pred_kmeans_results = None
    
    for model_key in results:
        if results[model_key] is not None:
            # Extract K-means on inputs
            if results[model_key]['log_kmeans_wasserstein'] is not None and kmeans_results is None:
                kmeans_results = {
                    'log_wasserstein': results[model_key]['log_kmeans_wasserstein'],
                    'snr_db': results[model_key]['snr_db']
                }
            
            # Extract K-means on predictions from baseline_16_layers specifically
            if model_key == 'baseline_16_layers' and results[model_key]['log_pred_kmeans_wasserstein'] is not None:
                pred_kmeans_results = {
                    'log_wasserstein': results[model_key]['log_pred_kmeans_wasserstein'],
                    'snr_db': results[model_key]['snr_db']
                }
    
    if kmeans_results is not None:
        results['kmeans'] = kmeans_results
        print(f"  K-means (inputs) Log Wasserstein - Mean: {kmeans_results['log_wasserstein'].mean():.3f}, "
              f"Std: {kmeans_results['log_wasserstein'].std():.3f}")
    else:
        print("  Warning: Could not extract K-means (inputs) results")
    
    if pred_kmeans_results is not None:
        results['pred_kmeans'] = pred_kmeans_results
        print(f"  K-means (predictions) Log Wasserstein - Mean: {pred_kmeans_results['log_wasserstein'].mean():.3f}, "
              f"Std: {pred_kmeans_results['log_wasserstein'].std():.3f}")
    else:
        print("  Warning: Could not extract K-means (predictions) results")
    
    return results

def plot_wasserstein_distributions(results, save_path=None):
    """Create scatter plot of log Wasserstein distributions."""
    print("\nCreating Wasserstein Distribution Plot")
    print("-" * 39)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colors and markers for different models
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'x']
    
    # Plot each model's results
    for i, (model_key, color, marker) in enumerate(zip(results.keys(), colors, markers)):
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'pred_kmeans':
            label = 'K-means (predictions)'
        elif model_key in model_configs:
            label = model_configs[model_key]['name']
        else:
            label = model_key
        
        # Convert log wasserstein back to wasserstein for plotting
        wass = np.exp(log_wass)
        ax.scatter(snr_db, wass, c=color, marker=marker, alpha=0.6, s=20, label=label)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Model Performance: Wasserstein Distance vs SNR')
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"Saved distribution plot to: {save_path}")
    
    return fig

def compute_average_loss_vs_snr(results, snr_bins=10, save_path=None):
    """Compute and plot average log loss as a function of SNR."""
    print("\nComputing Average Loss vs SNR")
    print("-" * 30)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define SNR bins
    snr_min, snr_max = 3.0, 15.0
    bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Colors for different models
    colors = {'baseline_16_layers': 'blue', 'baseline_32_layers': 'red', 
              'baseline_64_layers': 'green', 'no_flow_16_layers': 'purple', 
              'kmeans': 'orange', 'pred_kmeans': 'brown'}
    
    # Compute average loss for each model
    for model_key in results:
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Compute average in each bin
        avg_losses = []
        std_losses = []
        
        for i in range(len(bin_edges) - 1):
            mask = (snr_db >= bin_edges[i]) & (snr_db < bin_edges[i + 1])
            if np.any(mask):
                avg_losses.append(log_wass[mask].mean())
                std_losses.append(log_wass[mask].std())
            else:
                avg_losses.append(np.nan)
                std_losses.append(np.nan)
        
        avg_losses = np.array(avg_losses)
        std_losses = np.array(std_losses)
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'pred_kmeans':
            label = 'K-means (predictions)'
        elif model_key in model_configs:
            label = model_configs[model_key]['name']
        else:
            label = model_key
        
        # Plot average values (convert from log to actual Wasserstein distance)
        valid_mask = ~np.isnan(avg_losses)
        avg_wass = np.exp(avg_losses[valid_mask])
        
        ax.plot(bin_centers[valid_mask], avg_wass, 
                color=colors.get(model_key, 'black'),
                marker='o', markersize=6, linewidth=2,
                label=label)
        
        print(f"\n{label}:")
        print(f"  Average log loss range: [{np.nanmin(avg_losses):.3f}, {np.nanmax(avg_losses):.3f}]")
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Average Wasserstein Distance')
    ax.set_title('Average Model Performance vs SNR')
    ax.set_xlim(snr_min - 0.5, snr_max + 0.5)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nSaved average loss plot to: {save_path}")
    
    return fig

def plot_loess_conditional_expectations(results, save_path=None):
    """Create plot showing LOESS conditional expectations."""
    print("\nCreating LOESS Conditional Expectation Plot")
    print("-" * 44)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Colors for different models
    colors = {'baseline_16_layers': 'blue', 'baseline_32_layers': 'red', 
              'baseline_64_layers': 'green', 'no_flow_16_layers': 'purple', 
              'kmeans': 'orange', 'pred_kmeans': 'brown'}
    
    loess_results = {}
    
    # Compute LOESS for each model
    for model_key in results:
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Skip if no SNR data
        if len(snr_db) == 0:
            continue
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'pred_kmeans':
            label = 'K-means (predictions)'
        elif model_key in model_configs:
            label = model_configs[model_key]['name']
        else:
            label = model_key
            
        # Compute LOESS conditional expectation
        snr_grid, conditional_exp, best_frac = estimate_conditional_expectation(
            snr_db, log_wass, label
        )
        
        loess_results[model_key] = {
            'snr_grid': snr_grid,
            'conditional_exp': conditional_exp,
            'best_frac': best_frac
        }
        
        color = colors.get(model_key, 'black')
        
        # Plot conditional expectation curve (convert from log to actual Wasserstein)
        ax.plot(snr_grid, np.exp(conditional_exp), color=color, linewidth=2.5, 
                label=f'{label} (frac={best_frac:.3f})')
        
        # Plot original data points (subset for clarity, convert to actual Wasserstein)
        n_points = min(200, len(snr_db))
        indices = np.random.choice(len(snr_db), n_points, replace=False)
        ax.scatter(snr_db[indices], np.exp(log_wass[indices]), color=color, alpha=0.3, s=10)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('E[Wasserstein Distance | SNR]', fontsize=12)
    ax.set_title('LOESS Conditional Expectation: E[Wasserstein Distance | SNR]', fontsize=14)
    ax.set_xlim(3, 15)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax, loc='upper right')
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nLOESS plot saved to: {save_path}")
    
    return fig, loess_results

def plot_wasserstein_distributions_v2(results, save_path=None):
    """Create scatter plot of log Wasserstein distributions (Version B)."""
    print("\nCreating Wasserstein Distribution Plot (Version B)")
    print("-" * 47)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colors and markers for different models
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', 'x', '+']
    
    # Plot each model's results
    for i, (model_key, color, marker) in enumerate(zip(results.keys(), colors, markers)):
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'baseline_16_layers':
            label = 'Baseline 16L'
        elif model_key.startswith('pred_kmeans_'):
            base_model = model_key.replace('pred_kmeans_', '')
            if base_model == 'baseline_16_layers':
                label = 'K-means on Baseline 16L predictions'
            elif base_model == 'baseline_32_layers':
                label = 'K-means on Baseline 32L predictions'
            elif base_model == 'baseline_64_layers':
                label = 'K-means on Baseline 64L predictions'
            elif base_model == 'no_flow_16_layers':
                label = 'K-means on No Flow 16L predictions'
            else:
                label = f'K-means on {base_model} predictions'
        else:
            label = model_key
        
        # Convert log wasserstein back to wasserstein for plotting
        wass = np.exp(log_wass)
        ax.scatter(snr_db, wass, c=color, marker=marker, alpha=0.6, s=20, label=label)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('K-means Predictions Performance vs Neural Model')
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"Saved distribution plot (v2) to: {save_path}")
    
    return fig

def compute_average_loss_vs_snr_v2(results, snr_bins=10, save_path=None):
    """Compute and plot average log loss as a function of SNR (Version B)."""
    print("\nComputing Average Loss vs SNR (Version B)")
    print("-" * 38)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define SNR bins
    snr_min, snr_max = 3.0, 15.0
    bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Colors for different models
    colors = {'kmeans': 'orange', 'baseline_16_layers': 'blue',
              'pred_kmeans_baseline_16_layers': 'red',
              'pred_kmeans_baseline_32_layers': 'green', 
              'pred_kmeans_baseline_64_layers': 'purple',
              'pred_kmeans_no_flow_16_layers': 'brown'}
    
    # Compute average loss for each model
    for model_key in results:
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Compute average in each bin
        avg_losses = []
        std_losses = []
        
        for i in range(len(bin_edges) - 1):
            mask = (snr_db >= bin_edges[i]) & (snr_db < bin_edges[i + 1])
            if np.any(mask):
                avg_losses.append(log_wass[mask].mean())
                std_losses.append(log_wass[mask].std())
            else:
                avg_losses.append(np.nan)
                std_losses.append(np.nan)
        
        avg_losses = np.array(avg_losses)
        std_losses = np.array(std_losses)
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'baseline_16_layers':
            label = 'Baseline 16L'
        elif model_key.startswith('pred_kmeans_'):
            base_model = model_key.replace('pred_kmeans_', '')
            if base_model == 'baseline_16_layers':
                label = 'K-means on Baseline 16L predictions'
            elif base_model == 'baseline_32_layers':
                label = 'K-means on Baseline 32L predictions'
            elif base_model == 'baseline_64_layers':
                label = 'K-means on Baseline 64L predictions'
            elif base_model == 'no_flow_16_layers':
                label = 'K-means on No Flow 16L predictions'
            else:
                label = f'K-means on {base_model} predictions'
        else:
            label = model_key
        
        # Plot average values (convert from log to actual Wasserstein distance)
        valid_mask = ~np.isnan(avg_losses)
        avg_wass = np.exp(avg_losses[valid_mask])
        
        ax.plot(bin_centers[valid_mask], avg_wass, 
                color=colors.get(model_key, 'black'),
                marker='o', markersize=6, linewidth=2,
                label=label)
        
        print(f"\n{label}:")
        print(f"  Average log loss range: [{np.nanmin(avg_losses):.3f}, {np.nanmax(avg_losses):.3f}]")
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Average Wasserstein Distance')
    ax.set_title('K-means Predictions Performance vs Neural Model (Averaged)')
    ax.set_xlim(snr_min - 0.5, snr_max + 0.5)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nSaved average loss plot (v2) to: {save_path}")
    
    return fig

def plot_loess_conditional_expectations_v2(results, save_path=None):
    """Create plot showing LOESS conditional expectations (Version B)."""
    print("\nCreating LOESS Conditional Expectation Plot (Version B)")
    print("-" * 52)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Colors for different models
    colors = {'kmeans': 'orange', 'baseline_16_layers': 'blue',
              'pred_kmeans_baseline_16_layers': 'red',
              'pred_kmeans_baseline_32_layers': 'green', 
              'pred_kmeans_baseline_64_layers': 'purple',
              'pred_kmeans_no_flow_16_layers': 'brown'}
    
    loess_results = {}
    
    # Compute LOESS for each model
    for model_key in results:
        if results[model_key] is None:
            continue
            
        log_wass = results[model_key]['log_wasserstein']
        snr_db = results[model_key]['snr_db']
        
        # Skip if no SNR data
        if len(snr_db) == 0:
            continue
        
        # Determine label
        if model_key == 'kmeans':
            label = 'K-means (inputs)'
        elif model_key == 'baseline_16_layers':
            label = 'Baseline 16L'
        elif model_key.startswith('pred_kmeans_'):
            base_model = model_key.replace('pred_kmeans_', '')
            if base_model == 'baseline_16_layers':
                label = 'K-means on Baseline 16L predictions'
            elif base_model == 'baseline_32_layers':
                label = 'K-means on Baseline 32L predictions'
            elif base_model == 'baseline_64_layers':
                label = 'K-means on Baseline 64L predictions'
            elif base_model == 'no_flow_16_layers':
                label = 'K-means on No Flow 16L predictions'
            else:
                label = f'K-means on {base_model} predictions'
        else:
            label = model_key
            
        # Compute LOESS conditional expectation
        snr_grid, conditional_exp, best_frac = estimate_conditional_expectation(
            snr_db, log_wass, label
        )
        
        loess_results[model_key] = {
            'snr_grid': snr_grid,
            'conditional_exp': conditional_exp,
            'best_frac': best_frac
        }
        
        color = colors.get(model_key, 'black')
        
        # Plot conditional expectation curve (convert from log to actual Wasserstein)
        ax.plot(snr_grid, np.exp(conditional_exp), color=color, linewidth=2.5, 
                label=f'{label} (frac={best_frac:.3f})')
        
        # Plot original data points (subset for clarity, convert to actual Wasserstein)
        n_points = min(200, len(snr_db))
        indices = np.random.choice(len(snr_db), n_points, replace=False)
        ax.scatter(snr_db[indices], np.exp(log_wass[indices]), color=color, alpha=0.3, s=10)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('E[Wasserstein Distance | SNR]', fontsize=12)
    ax.set_title('K-means Predictions Performance vs Neural Model (LOESS)', fontsize=14)
    ax.set_xlim(3, 15)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax, loc='upper right')
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nLOESS plot (v2) saved to: {save_path}")
    
    return fig, loess_results

# Main execution
if __name__ == "__main__":
    # Evaluate all models
    results = evaluate_models(model_configs, dataset_name, total_samples, batch_size, device)
    
    # Save results
    results_path = tutorial_output_dir / "model_evaluation_results.pt"
    torch.save(results, results_path)
    print(f"\nSaved evaluation results to: {results_path}")
    
    # Create visualizations
    print("\nGenerating Visualizations")
    print("-" * 25)
    
    # ========================================
    # Version A: Current plots without K-means on predictions
    # ========================================
    print("\nVersion A: Neural models + K-means baseline (without K-means on predictions)")
    results_a = {k: v for k, v in results.items() if k != 'pred_kmeans'}
    
    # 1A. Wasserstein distribution scatter plot
    dist_plot_a = plot_wasserstein_distributions(
        results_a, 
        save_path=tutorial_output_dir / "a_wasserstein_distributions.png"
    )
    
    # 2A. Average loss vs SNR plot
    avg_plot_a = compute_average_loss_vs_snr(
        results_a,
        snr_bins=10,
        save_path=tutorial_output_dir / "a_average_loss_vs_snr.png"
    )
    
    # 3A. LOESS conditional expectation plot
    loess_plot_a, loess_results_a = plot_loess_conditional_expectations(
        results_a,
        save_path=tutorial_output_dir / "a_loess_conditional_expectations.png"
    )
    
    # ========================================
    # Version B: K-means predictions on all models + baseline_16_model
    # ========================================
    print("\nVersion B: K-means predictions comparison + baseline_16_model")
    
    # Extract K-means predictions for all models that have them
    results_b = {}
    
    # Add K-means baseline
    if 'kmeans' in results and results['kmeans'] is not None:
        results_b['kmeans'] = results['kmeans']
    
    # Add baseline_16_layers model
    if 'baseline_16_layers' in results and results['baseline_16_layers'] is not None:
        results_b['baseline_16_layers'] = results['baseline_16_layers']
    
    # Add K-means predictions for all models that have them
    for model_key in ['baseline_16_layers', 'baseline_32_layers', 'baseline_64_layers', 'no_flow_16_layers']:
        if (model_key in results and results[model_key] is not None and 
            results[model_key]['log_pred_kmeans_wasserstein'] is not None):
            results_b[f'pred_kmeans_{model_key}'] = {
                'log_wasserstein': results[model_key]['log_pred_kmeans_wasserstein'],
                'snr_db': results[model_key]['snr_db']
            }
    
    # 1B. Wasserstein distribution scatter plot
    dist_plot_b = plot_wasserstein_distributions_v2(
        results_b, 
        save_path=tutorial_output_dir / "b_wasserstein_distributions.png"
    )
    
    # 2B. Average loss vs SNR plot
    avg_plot_b = compute_average_loss_vs_snr_v2(
        results_b,
        snr_bins=10,
        save_path=tutorial_output_dir / "b_average_loss_vs_snr.png"
    )
    
    # 3B. LOESS conditional expectation plot
    loess_plot_b, loess_results_b = plot_loess_conditional_expectations_v2(
        results_b,
        save_path=tutorial_output_dir / "b_loess_conditional_expectations.png"
    )
    
    # Save LOESS results for both versions
    loess_results_a_path = tutorial_output_dir / "a_loess_results.pt"
    loess_results_b_path = tutorial_output_dir / "b_loess_results.pt"
    torch.save(loess_results_a, loess_results_a_path)
    torch.save(loess_results_b, loess_results_b_path)
    print(f"\nSaved LOESS results A to: {loess_results_a_path}")
    print(f"Saved LOESS results B to: {loess_results_b_path}")
    
    print("\n" + "=" * 70)
    print("Performance Comparison Complete")
    print("=" * 70)
    print("\nGenerated two sets of plots:")
    print("Version A: Neural models comparison (without K-means on predictions)")
    print("  - Files: a_wasserstein_distributions.png, a_average_loss_vs_snr.png, a_loess_conditional_expectations.png")
    print("Version B: K-means predictions comparison + baseline_16_model")  
    print("  - Files: b_wasserstein_distributions.png, b_average_loss_vs_snr.png, b_loess_conditional_expectations.png")
    print("\nKey insights from the evaluation:")
    print("- Results show absolute log Wasserstein distances (lower is better)")
    print("- Y-axis shows log10 scale of Wasserstein distances")
    print("- Performance generally improves with higher SNR (less noise)")
    print("- Version A compares neural approaches vs traditional clustering")
    print("- Version B shows how well model outputs can be clustered vs the neural model itself")
    
    plt.show()