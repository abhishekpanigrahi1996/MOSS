"""
GMM Model Evaluation Tutorial - Part 12: SNR Performance Table
=============================================================

This script evaluates model performance across different fixed SNR datasets:
- high_snr_fixed (high SNR)
- average_snr_fixed (medium SNR) 
- low_snr_fixed (low SNR)

Creates a performance table showing average Wasserstein distance for each
model-dataset combination, with caching of evaluation results.
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 12: SNR Performance Table")
print("=" * 70)
print()

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

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

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('/mount/Storage/gmm-v4/output')
experiment_base_dir = output_dir / 'final_experiments'
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output/snr_performance')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configuration
datasets = ["high_snr_fixed", "average_snr_fixed", "low_snr_fixed"]
batch_size = 32  # Increased for faster GPU evaluation
total_samples = 4096

print("Evaluation Configuration")
print("-" * 24)
print(f"Datasets: {', '.join(datasets)}")
print(f"Total samples per dataset: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Device: {device}")
print()

# Define models to evaluate
model_configs = {
    "baseline_16_layers": {"name": "16 layers", "path": "baseline_16_layers"},
    "baseline_32_layers": {"name": "32 layers", "path": "baseline_32_layers"},
    "baseline_64_layers": {"name": "64 layers", "path": "baseline_64_layers"},
    "no_flow_16_layers": {"name": "No flow (16 layers)", "path": "no_flow_16_layers"},
}

def get_cache_path(dataset_name, model_key=None):
    """Get cache file path for evaluation results."""
    if model_key:
        return tutorial_output_dir / f"{dataset_name}_{model_key}_results.pt"
    else:
        return tutorial_output_dir / f"{dataset_name}_kmeans_results.pt"

def evaluate_model_on_dataset(model_key, model_config, dataset_name, data_loader, device):
    """Evaluate a single model on a dataset, with caching."""
    cache_path = get_cache_path(dataset_name, model_key)
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Evaluating {model_config['name']} on {dataset_name}...")
    
    try:
        # Load model
        model_path = experiment_base_dir / model_config["path"]
        model, _ = load_model_from_experiment(model_path, load_best=False, device=device)
        
        # Evaluate dataset
        eval_results_list = evaluate_dataset(
            model, 
            data_loader,
            kmeans_on_inputs=False,  # We'll compute K-means separately
            kmeans_on_predictions=False,
            metrics=['log_wasserstein'],
            device=device
        )
        
        # Aggregate results
        all_wass = []
        all_log_wass = []
        all_snr_db = []
        
        for batch_results in eval_results_list:
            if 'metrics' in batch_results and 'log_wasserstein' in batch_results['metrics']:
                batch_log_wass = batch_results['metrics']['log_wasserstein'].cpu().numpy()
                all_log_wass.extend(batch_log_wass)
                # Compute regular wasserstein from log wasserstein: exp(log_w) = w
                batch_wass = np.exp(batch_log_wass)
                all_wass.extend(batch_wass)
            if 'snr_values' in batch_results and batch_results['snr_values'] is not None:
                batch_snr = batch_results['snr_values'].cpu().numpy()
                all_snr_db.extend(batch_snr)
        
        results = {
            'wasserstein': np.array(all_wass),
            'log_wasserstein': np.array(all_log_wass),
            'snr_db': np.array(all_snr_db) if all_snr_db else None,
            'avg_wasserstein': np.mean(all_wass),
            'std_wasserstein': np.std(all_wass),
            'avg_log_wasserstein': np.mean(all_log_wass),
            'std_log_wasserstein': np.std(all_log_wass)
        }
        
        # Cache results
        torch.save(results, cache_path)
        print(f"  Cached results to: {cache_path}")
        
        return results
        
    except Exception as e:
        print(f"  Error evaluating {model_config['name']}: {e}")
        return None

def evaluate_kmeans_baseline(dataset_name, data_loader, device):
    """Evaluate K-means baseline on a dataset, with caching."""
    cache_path = get_cache_path(dataset_name)
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached K-means results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Computing K-means baseline for {dataset_name}...")
    
    try:
        # Use any model just to get K-means results (we'll use baseline_16_layers)
        model_path = experiment_base_dir / "baseline_16_layers"
        model, _ = load_model_from_experiment(model_path, load_best=False, device=device)
        
        # Evaluate with K-means only
        eval_results_list = evaluate_dataset(
            model, 
            data_loader,
            kmeans_on_inputs=True,
            kmeans_on_predictions=False,
            metrics=['log_kmeans_wasserstein'],
            device=device
        )
        
        # Aggregate K-means results
        all_kmeans_wass = []
        all_log_kmeans_wass = []
        all_snr_db = []
        
        for batch_results in eval_results_list:
            if 'metrics' in batch_results and 'log_kmeans_wasserstein' in batch_results['metrics']:
                batch_log_kmeans_wass = batch_results['metrics']['log_kmeans_wasserstein'].cpu().numpy()
                all_log_kmeans_wass.extend(batch_log_kmeans_wass)
                # Compute regular wasserstein from log wasserstein: exp(log_w) = w
                batch_kmeans_wass = np.exp(batch_log_kmeans_wass)
                all_kmeans_wass.extend(batch_kmeans_wass)
            if 'snr_values' in batch_results and batch_results['snr_values'] is not None:
                batch_snr = batch_results['snr_values'].cpu().numpy()
                all_snr_db.extend(batch_snr)
        
        results = {
            'wasserstein': np.array(all_kmeans_wass),
            'log_wasserstein': np.array(all_log_kmeans_wass),
            'snr_db': np.array(all_snr_db) if all_snr_db else None,
            'avg_wasserstein': np.mean(all_kmeans_wass),
            'std_wasserstein': np.std(all_kmeans_wass),
            'avg_log_wasserstein': np.mean(all_log_kmeans_wass),
            'std_log_wasserstein': np.std(all_log_kmeans_wass)
        }
        
        # Cache results
        torch.save(results, cache_path)
        print(f"  Cached K-means results to: {cache_path}")
        
        return results
        
    except Exception as e:
        print(f"  Error computing K-means baseline: {e}")
        return None

# Main evaluation
print("\nEvaluating Models on Fixed SNR Datasets")
print("=" * 40)

# Initialize results table
results_table = pd.DataFrame(
    index=datasets,
    columns=['K-means'] + list(model_configs.keys())
)

# Store detailed results for later analysis
detailed_results = {}

# Evaluate each dataset
for dataset_name in datasets:
    print(f"\n\nDataset: {dataset_name}")
    print("-" * (9 + len(dataset_name)))
    
    # Create data loader for this dataset
    data_loader = create_data_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        total_samples=total_samples,
        device=device
    )
    
    # Get dataset SNR info
    sample_batch = next(iter(data_loader))
    if 'snr_values' in sample_batch and sample_batch['snr_values'] is not None:
        snr_values = sample_batch['snr_values'].cpu().numpy()
        print(f"  SNR range: {snr_values.min():.1f} - {snr_values.max():.1f} dB")
    
    # Evaluate K-means baseline
    print("\n  K-means Baseline:")
    kmeans_results = evaluate_kmeans_baseline(dataset_name, data_loader, device)
    if kmeans_results:
        avg_loss = kmeans_results['avg_wasserstein']
        std_loss = kmeans_results['std_wasserstein']
        results_table.loc[dataset_name, 'K-means'] = f"{avg_loss:.4f} ± {std_loss:.4f}"
        print(f"    Average Wasserstein: {avg_loss:.4f} ± {std_loss:.4f}")
        
        if dataset_name not in detailed_results:
            detailed_results[dataset_name] = {}
        detailed_results[dataset_name]['kmeans'] = kmeans_results
    
    # Evaluate each model
    print("\n  Neural Models:")
    for model_key, model_config in model_configs.items():
        model_results = evaluate_model_on_dataset(
            model_key, model_config, dataset_name, data_loader, device
        )
        
        if model_results:
            avg_loss = model_results['avg_wasserstein']
            std_loss = model_results['std_wasserstein']
            results_table.loc[dataset_name, model_key] = f"{avg_loss:.4f} ± {std_loss:.4f}"
            print(f"    {model_config['name']}: {avg_loss:.4f} ± {std_loss:.4f}")
            
            if dataset_name not in detailed_results:
                detailed_results[dataset_name] = {}
            detailed_results[dataset_name][model_key] = model_results

print("\n\n" + "=" * 70)
print("Performance Summary Table")
print("=" * 70)
print("\nAverage Wasserstein Distance (mean ± std):")
print(results_table.to_string())

# Save results table to CSV
csv_path = tutorial_output_dir / "snr_performance_table.csv"
results_table.to_csv(csv_path)
print(f"\nSaved performance table to: {csv_path}")

# Save detailed results
detailed_results_path = tutorial_output_dir / "detailed_evaluation_results.pt"
torch.save(detailed_results, detailed_results_path)
print(f"Saved detailed results to: {detailed_results_path}")

# Create log wasserstein table
print("\n\nAverage Log Wasserstein Distance Table:")
print("-" * 40)
log_results_table = pd.DataFrame(
    index=datasets,
    columns=['K-means'] + list(model_configs.keys())
)

for dataset in datasets:
    for col in log_results_table.columns:
        if col == 'K-means':
            model_key = 'kmeans'
        else:
            model_key = col
            
        if dataset in detailed_results and model_key in detailed_results[dataset]:
            avg_log = detailed_results[dataset][model_key]['avg_log_wasserstein']
            std_log = detailed_results[dataset][model_key]['std_log_wasserstein']
            log_results_table.loc[dataset, col] = f"{avg_log:.4f} ± {std_log:.4f}"

print("\nAverage Log Wasserstein Distance (mean ± std):")
print(log_results_table.to_string())

# Save log wasserstein table
log_csv_path = tutorial_output_dir / "snr_performance_log_wasserstein_table.csv"
log_results_table.to_csv(log_csv_path)
print(f"\nSaved log wasserstein table to: {log_csv_path}")

# Create a cleaner numeric table for analysis
print("\n\nNumeric Performance Table (Average Wasserstein):")
print("-" * 60)

numeric_table = pd.DataFrame(
    index=datasets,
    columns=['K-means'] + list(model_configs.keys()),
    dtype=float
)

for dataset in datasets:
    for col in numeric_table.columns:
        if col == 'K-means':
            model_key = 'kmeans'
        else:
            model_key = col
            
        if dataset in detailed_results and model_key in detailed_results[dataset]:
            numeric_table.loc[dataset, col] = detailed_results[dataset][model_key]['avg_wasserstein']

print(numeric_table.round(4).to_string())

# Save numeric table
numeric_csv_path = tutorial_output_dir / "snr_performance_numeric.csv"
numeric_table.to_csv(numeric_csv_path)
print(f"\nSaved numeric table to: {numeric_csv_path}")

# Create numeric log wasserstein table
print("\n\nNumeric Performance Table (Average Log Wasserstein):")
print("-" * 60)

log_numeric_table = pd.DataFrame(
    index=datasets,
    columns=['K-means'] + list(model_configs.keys()),
    dtype=float
)

for dataset in datasets:
    for col in log_numeric_table.columns:
        if col == 'K-means':
            model_key = 'kmeans'
        else:
            model_key = col
            
        if dataset in detailed_results and model_key in detailed_results[dataset]:
            log_numeric_table.loc[dataset, col] = detailed_results[dataset][model_key]['avg_log_wasserstein']

print(log_numeric_table.round(4).to_string())

# Save log numeric table
log_numeric_csv_path = tutorial_output_dir / "snr_performance_log_numeric.csv"
log_numeric_table.to_csv(log_numeric_csv_path)
print(f"\nSaved log numeric table to: {log_numeric_csv_path}")

# Compute improvement over K-means (Wasserstein)
print("\n\nImprovement over K-means Baseline (%) - Wasserstein:")
print("-" * 52)

improvement_table = pd.DataFrame(
    index=datasets,
    columns=list(model_configs.keys()),
    dtype=float
)

for dataset in datasets:
    kmeans_score = numeric_table.loc[dataset, 'K-means']
    for model_key in model_configs.keys():
        model_score = numeric_table.loc[dataset, model_key]
        if pd.notna(kmeans_score) and pd.notna(model_score):
            # Since lower is better, improvement = (kmeans - model) / kmeans * 100
            improvement = (kmeans_score - model_score) / abs(kmeans_score) * 100
            improvement_table.loc[dataset, model_key] = improvement

print(improvement_table.round(2).to_string())

# Save improvement table
improvement_csv_path = tutorial_output_dir / "snr_performance_improvement.csv"
improvement_table.to_csv(improvement_csv_path)
print(f"\nSaved improvement table to: {improvement_csv_path}")

# Compute improvement over K-means (Log Wasserstein)
print("\n\nImprovement over K-means Baseline (%) - Log Wasserstein:")
print("-" * 56)

log_improvement_table = pd.DataFrame(
    index=datasets,
    columns=list(model_configs.keys()),
    dtype=float
)

for dataset in datasets:
    kmeans_score = log_numeric_table.loc[dataset, 'K-means']
    for model_key in model_configs.keys():
        model_score = log_numeric_table.loc[dataset, model_key]
        if pd.notna(kmeans_score) and pd.notna(model_score):
            # Since lower is better, improvement = (kmeans - model) / abs(kmeans) * 100
            improvement = (kmeans_score - model_score) / abs(kmeans_score) * 100
            log_improvement_table.loc[dataset, model_key] = improvement

print(log_improvement_table.round(2).to_string())

# Save log improvement table
log_improvement_csv_path = tutorial_output_dir / "snr_performance_log_improvement.csv"
log_improvement_table.to_csv(log_improvement_csv_path)
print(f"\nSaved log improvement table to: {log_improvement_csv_path}")

print("\n" + "=" * 70)
print("SNR Performance Evaluation Complete")
print("=" * 70)

# Summary statistics
print("\n\nKey Findings:")
print("-" * 13)

# Best model per dataset (Wasserstein)
print("\n### Wasserstein Distance Rankings ###")
for dataset in datasets:
    row = numeric_table.loc[dataset]
    best_model = row.idxmin()
    best_score = row.min()
    print(f"\n{dataset}:")
    print(f"  Best model: {best_model} (Wasserstein: {best_score:.4f})")
    
    # Show ranking
    ranked = row.sort_values()
    print("  Full ranking:")
    for i, (model, score) in enumerate(ranked.items(), 1):
        if pd.notna(score):
            print(f"    {i}. {model}: {score:.4f}")

# Best model per dataset (Log Wasserstein)
print("\n\n### Log Wasserstein Distance Rankings ###")
for dataset in datasets:
    row = log_numeric_table.loc[dataset]
    best_model = row.idxmin()
    best_score = row.min()
    print(f"\n{dataset}:")
    print(f"  Best model: {best_model} (Log Wasserstein: {best_score:.4f})")
    
    # Show ranking
    ranked = row.sort_values()
    print("  Full ranking:")
    for i, (model, score) in enumerate(ranked.items(), 1):
        if pd.notna(score):
            print(f"    {i}. {model}: {score:.4f}")

print("\n" + "=" * 70)

# Additional evaluation on diverse_snr_moderate dataset for plotting
print("\n\nAdditional Evaluation for Average Performance Plots")
print("=" * 51)

# Set plotting style
set_plotting_style()

def evaluate_models_for_plotting(model_configs, device):
    """Evaluate all models on diverse_snr_moderate dataset for plotting."""
    
    diverse_dataset = "diverse_snr_moderate"
    batch_size = 1
    total_samples = 4096
    
    print(f"\nEvaluating models on {diverse_dataset} dataset")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    
    # Create data loader
    data_loader = create_data_loader(
        dataset_name=diverse_dataset,
        batch_size=batch_size,
        total_samples=total_samples,
        device=device
    )
    
    # Store all results
    all_results = {}
    
    # Evaluate K-means baseline
    print("\nEvaluating K-means baseline...")
    kmeans_cache_path = tutorial_output_dir / f"{diverse_dataset}_kmeans_full_results.pt"
    
    if kmeans_cache_path.exists():
        print(f"  Loading cached K-means results from: {kmeans_cache_path}")
        all_results['kmeans'] = torch.load(kmeans_cache_path, weights_only=False)
    else:
        # Use evaluate_kmeans_baseline function
        kmeans_results = evaluate_kmeans_baseline(diverse_dataset, data_loader, device)
        if kmeans_results:
            all_results['kmeans'] = kmeans_results
            # Save the full results with different name to avoid overwriting simple cache
            torch.save(kmeans_results, kmeans_cache_path)
            print(f"  Saved K-means results to: {kmeans_cache_path}")
    
    # Evaluate each model
    for model_key, model_config in model_configs.items():
        cache_path = tutorial_output_dir / f"{diverse_dataset}_{model_key}_full_results.pt"
        
        if cache_path.exists():
            print(f"\nLoading cached {model_config['name']} results...")
            all_results[model_key] = torch.load(cache_path, weights_only=False)
        else:
            print(f"\nEvaluating {model_config['name']}...")
            model_results = evaluate_model_on_dataset(
                model_key, model_config, diverse_dataset, data_loader, device
            )
            if model_results:
                all_results[model_key] = model_results
                # Save the full results with different name
                torch.save(model_results, cache_path)
                print(f"  Saved results to: {cache_path}")
    
    return all_results

def compute_average_performance_vs_snr(results, model_configs, snr_bins=10, save_path=None):
    """Compute and plot average performance as a function of SNR."""
    print("\nCreating Average Performance vs SNR Plot")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define SNR bins
    snr_min, snr_max = 3.0, 15.0
    bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Colors for different models
    colors = {
        'kmeans': 'orange',
        'baseline_16_layers': 'blue',
        'baseline_32_layers': 'red',
        'baseline_64_layers': 'green',
        'no_flow_16_layers': 'purple'
    }
    
    # Plot K-means baseline
    if 'kmeans' in results and results['kmeans'] is not None:
        log_wass = results['kmeans']['log_wasserstein']
        snr_db = results['kmeans']['snr_db']
        
        # Compute average in each bin
        avg_wass = []
        
        for i in range(len(bin_edges) - 1):
            mask = (snr_db >= bin_edges[i]) & (snr_db < bin_edges[i + 1])
            if np.any(mask):
                # Convert to actual Wasserstein distance
                avg_wass.append(np.exp(log_wass[mask].mean()))
            else:
                avg_wass.append(np.nan)
        
        avg_wass = np.array(avg_wass)
        valid_mask = ~np.isnan(avg_wass)
        
        ax.plot(bin_centers[valid_mask], avg_wass[valid_mask], 
                color=colors['kmeans'], marker='o', markersize=6, 
                linewidth=2, label='K-means')
    
    # Plot each model
    for model_key, model_config in model_configs.items():
        if model_key in results and results[model_key] is not None:
            log_wass = results[model_key]['log_wasserstein']
            snr_db = results[model_key]['snr_db']
            
            # Compute average in each bin
            avg_wass = []
            
            for i in range(len(bin_edges) - 1):
                mask = (snr_db >= bin_edges[i]) & (snr_db < bin_edges[i + 1])
                if np.any(mask):
                    # Convert to actual Wasserstein distance
                    avg_wass.append(np.exp(log_wass[mask].mean()))
                else:
                    avg_wass.append(np.nan)
            
            avg_wass = np.array(avg_wass)
            valid_mask = ~np.isnan(avg_wass)
            
            label = model_config['name']
            color = colors.get(model_key, 'black')
            
            ax.plot(bin_centers[valid_mask], avg_wass[valid_mask], 
                    color=color, linestyle='-', linewidth=2,
                    marker='o', markersize=6, label=label)
            
            print(f"\n{label}:")
            print(f"  Average Wasserstein range: [{np.nanmin(avg_wass):.4f}, {np.nanmax(avg_wass):.4f}]")
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Average Wasserstein Distance')
    ax.set_title('Model Performance vs SNR')
    ax.set_xlim(snr_min - 0.5, snr_max + 0.5)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nSaved average performance plot to: {save_path}")
    
    return fig

# Run the additional evaluation
diverse_results = evaluate_models_for_plotting(model_configs, device)

# Save all diverse dataset results
diverse_results_path = tutorial_output_dir / "diverse_dataset_full_results.pt"
torch.save(diverse_results, diverse_results_path)
print(f"\nSaved all diverse dataset results to: {diverse_results_path}")

# Create average performance plot
avg_perf_plot = compute_average_performance_vs_snr(
    diverse_results,
    model_configs,
    snr_bins=10,
    save_path=tutorial_output_dir / "average_loss_vs_snr.png"
)

# Also create LOESS plots similar to script 11
def plot_loess_conditional_expectations(results, model_configs, save_path=None):
    """Create plot showing LOESS conditional expectations."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    print("\nCreating LOESS Conditional Expectation Plot")
    print("-" * 44)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Colors for different models
    colors = {
        'kmeans': 'orange',
        'baseline_16_layers': 'blue',
        'baseline_32_layers': 'red',
        'baseline_64_layers': 'green',
        'no_flow_16_layers': 'purple'
    }
    
    loess_results = {}
    
    # Compute LOESS for each model
    for model_key in ['kmeans'] + list(model_configs.keys()):
        if model_key in results and results[model_key] is not None:
            log_wass = results[model_key]['log_wasserstein']
            snr_db = results[model_key]['snr_db']
            
            # Skip if no SNR data
            if len(snr_db) == 0:
                continue
            
            # Determine label
            if model_key == 'kmeans':
                label = 'K-means'
            else:
                label = model_configs[model_key]['name']
            
            # Use fixed frac = 0.1
            frac = 0.1
            
            # Grid for smooth curves
            snr_grid = np.linspace(snr_db.min(), snr_db.max(), 50)
            
            # Compute LOESS fit
            loess_fit = lowess(log_wass, snr_db, frac=frac, return_sorted=True)
            conditional_exp = np.interp(snr_grid, loess_fit[:, 0], loess_fit[:, 1])
            
            loess_results[model_key] = {
                'snr_grid': snr_grid,
                'conditional_exp': conditional_exp,
                'frac': frac
            }
            
            color = colors.get(model_key, 'black')
            
            # Plot conditional expectation curve (convert from log to actual Wasserstein)
            ax.plot(snr_grid, np.exp(conditional_exp), color=color, linewidth=2.5, 
                    label=f'{label} (frac={frac:.3f})')
            
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

# Create LOESS plot
loess_plot, loess_results = plot_loess_conditional_expectations(
    diverse_results,
    model_configs,
    save_path=tutorial_output_dir / "loess_conditional_expectations.png"
)

# Save LOESS results
loess_results_path = tutorial_output_dir / "loess_results.pt"
torch.save(loess_results, loess_results_path)
print(f"Saved LOESS results to: {loess_results_path}")

# Create scatter plot of all results
def plot_wasserstein_distributions(results, model_configs, save_path=None):
    """Create scatter plot of Wasserstein distributions."""
    print("\nCreating Wasserstein Distribution Plot")
    print("-" * 39)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colors and markers for different models
    colors = ['orange', 'blue', 'red', 'green', 'purple']
    markers = ['v', 'o', 's', '^', 'D']
    
    # Plot each model's results
    model_keys = ['kmeans'] + list(model_configs.keys())
    for i, model_key in enumerate(model_keys):
        if model_key in results and results[model_key] is not None:
            log_wass = results[model_key]['log_wasserstein']
            snr_db = results[model_key]['snr_db']
            
            # Convert to actual Wasserstein distance
            wass = np.exp(log_wass)
            
            # Determine label
            if model_key == 'kmeans':
                label = 'K-means'
            else:
                label = model_configs[model_key]['name']
            
            ax.scatter(snr_db, wass, c=colors[i], marker=markers[i], 
                      alpha=0.6, s=20, label=label)
    
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

# Create distribution plot
dist_plot = plot_wasserstein_distributions(
    diverse_results,
    model_configs,
    save_path=tutorial_output_dir / "wasserstein_distributions.png"
)

print("\n" + "=" * 70)
print("SNR Performance Evaluation with Plots Complete")
print("=" * 70)