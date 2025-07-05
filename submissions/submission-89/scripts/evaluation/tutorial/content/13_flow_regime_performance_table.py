"""
GMM Model Evaluation Tutorial - Part 13: Flow Regime Performance Table
====================================================================

This script evaluates model performance with different flow regimes:
- baseline_16_layers with direct flow
- baseline_16_layers with fractional flow  
- fractional_16_layers with direct flow
- fractional_16_layers with fractional flow

Creates a performance table showing average Wasserstein distance for each
model-dataset-flow_regime combination across different SNR datasets.
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 13: Flow Regime Performance Table")
print("=" * 70)
print()

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import copy

# Add project root to path if needed
project_root = '/Users/azimin/Programming/gmm-v4/'
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
project_path = Path(project_root)
output_dir = project_path / 'output'
experiment_base_dir = output_dir / 'final_experiments'
tutorial_output_dir = project_path / 'scripts/evaluation/tutorial/output/flow_regime_performance'
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

# Define base models and flow regimes to evaluate
base_models = {
    "baseline_16_layers": {"name": "Baseline 16L", "path": "baseline_16_layers"},
    "fractional_16_layers": {"name": "Fractional 16L", "path": "fractional_16_layers"},
}

flow_regimes = ["direct", "fractional"]

def patch_model_flow_regime(model, flow_regime):
    """
    Patch a model to use a specific flow distribution mode.
    
    Args:
        model: The model to patch
        flow_regime: 'direct' or 'fractional'
        
    Returns:
        Patched model (deep copy with modified flow settings)
    """
    # Create deep copy to avoid modifying original
    patched_model = copy.deepcopy(model)
    
    # Apply flow regime patch
    if hasattr(patched_model, 'transformer') and hasattr(patched_model.transformer, 'flow_distribution_mode'):
        print(f"    Patching flow_distribution_mode: {patched_model.transformer.flow_distribution_mode} → {flow_regime}")
        patched_model.transformer.flow_distribution_mode = flow_regime
    else:
        print(f"    Warning: flow_distribution_mode not found in model, cannot patch")
    
    return patched_model

def get_cache_path(dataset_name, model_key, flow_regime):
    """Get cache file path for evaluation results."""
    return tutorial_output_dir / f"{dataset_name}_{model_key}_{flow_regime}_results.pt"

def evaluate_model_with_flow_regime(model_key, model_config, dataset_name, data_loader, flow_regime, device):
    """Evaluate a single model with a specific flow regime on a dataset, with caching."""
    cache_path = get_cache_path(dataset_name, model_key, flow_regime)
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Evaluating {model_config['name']} ({flow_regime} flow) on {dataset_name}...")
    
    try:
        # Load model
        model_path = experiment_base_dir / model_config["path"]
        model, _ = load_model_from_experiment(model_path, load_best=False, device=device)
        
        # Patch model with flow regime
        patched_model = patch_model_flow_regime(model, flow_regime)
        
        # Evaluate dataset
        eval_results_list = evaluate_dataset(
            patched_model, 
            data_loader,
            kmeans_on_inputs=False,
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
        print(f"  Error evaluating {model_config['name']} with {flow_regime} flow: {e}")
        return None

def evaluate_kmeans_baseline(dataset_name, data_loader, device):
    """Evaluate K-means baseline on a dataset, with caching."""
    cache_path = tutorial_output_dir / f"{dataset_name}_kmeans_results.pt"
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached K-means results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Computing K-means baseline for {dataset_name}...")
    
    try:
        # Use any model just to get K-means results
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
print("\nEvaluating Models with Different Flow Regimes")
print("=" * 46)

# Create column names for the 4 model combinations + K-means
column_names = ['K-means']
for model_key in base_models.keys():
    for flow_regime in flow_regimes:
        # Create descriptive column names
        if model_key == 'baseline_16_layers':
            if flow_regime == 'direct':
                col_name = 'Uniform → Uniform'
            else:
                col_name = 'Uniform → Unit'
        elif model_key == 'fractional_16_layers':
            if flow_regime == 'direct':
                col_name = 'Unit → Uniform'
            else:
                col_name = 'Unit → Unit'
        else:
            col_name = f"{base_models[model_key]['name']} ({flow_regime})"
        column_names.append(col_name)

# Initialize results tables
results_table = pd.DataFrame(
    index=datasets,
    columns=column_names
)

log_results_table = pd.DataFrame(
    index=datasets,
    columns=column_names
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
        avg_log_loss = kmeans_results['avg_log_wasserstein']
        std_log_loss = kmeans_results['std_log_wasserstein']
        
        results_table.loc[dataset_name, 'K-means'] = f"{avg_loss:.4f} ± {std_loss:.4f}"
        log_results_table.loc[dataset_name, 'K-means'] = f"{avg_log_loss:.4f} ± {std_log_loss:.4f}"
        print(f"    Average Wasserstein: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"    Average Log Wasserstein: {avg_log_loss:.4f} ± {std_log_loss:.4f}")
        
        if dataset_name not in detailed_results:
            detailed_results[dataset_name] = {}
        detailed_results[dataset_name]['kmeans'] = kmeans_results
    
    # Evaluate each model with each flow regime
    print("\n  Neural Models with Flow Regimes:")
    for model_key, model_config in base_models.items():
        for flow_regime in flow_regimes:
            model_results = evaluate_model_with_flow_regime(
                model_key, model_config, dataset_name, data_loader, flow_regime, device
            )
            
            if model_results:
                avg_loss = model_results['avg_wasserstein']
                std_loss = model_results['std_wasserstein']
                avg_log_loss = model_results['avg_log_wasserstein']
                std_log_loss = model_results['std_log_wasserstein']
                
                # Use consistent column naming
                if model_key == 'baseline_16_layers':
                    if flow_regime == 'direct':
                        col_name = 'Uniform → Uniform'
                    else:
                        col_name = 'Uniform → Unit'
                elif model_key == 'fractional_16_layers':
                    if flow_regime == 'direct':
                        col_name = 'Unit → Uniform'
                    else:
                        col_name = 'Unit → Unit'
                else:
                    col_name = f"{model_config['name']} ({flow_regime})"
                    
                results_table.loc[dataset_name, col_name] = f"{avg_loss:.4f} ± {std_loss:.4f}"
                log_results_table.loc[dataset_name, col_name] = f"{avg_log_loss:.4f} ± {std_log_loss:.4f}"
                
                print(f"    {col_name}:")
                print(f"      Wasserstein: {avg_loss:.4f} ± {std_loss:.4f}")
                print(f"      Log Wasserstein: {avg_log_loss:.4f} ± {std_log_loss:.4f}")
                
                if dataset_name not in detailed_results:
                    detailed_results[dataset_name] = {}
                detailed_results[dataset_name][f"{model_key}_{flow_regime}"] = model_results

print("\n\n" + "=" * 70)
print("Performance Summary Tables")
print("=" * 70)

print("\nAverage Wasserstein Distance (mean ± std):")
print(results_table.to_string())

print("\n\nAverage Log Wasserstein Distance (mean ± std):")
print(log_results_table.to_string())

# Save results tables to CSV
csv_path = tutorial_output_dir / "flow_regime_wasserstein_table.csv"
results_table.to_csv(csv_path)
print(f"\nSaved Wasserstein table to: {csv_path}")

log_csv_path = tutorial_output_dir / "flow_regime_log_wasserstein_table.csv"
log_results_table.to_csv(log_csv_path)
print(f"Saved Log Wasserstein table to: {log_csv_path}")

# Save detailed results
detailed_results_path = tutorial_output_dir / "detailed_evaluation_results.pt"
torch.save(detailed_results, detailed_results_path)
print(f"Saved detailed results to: {detailed_results_path}")

# Create cleaner numeric tables for analysis
print("\n\nNumeric Performance Tables")
print("-" * 26)

# Wasserstein numeric table
print("\nAverage Wasserstein Distance:")
numeric_table = pd.DataFrame(
    index=datasets,
    columns=column_names,
    dtype=float
)

for dataset in datasets:
    # K-means
    if dataset in detailed_results and 'kmeans' in detailed_results[dataset]:
        numeric_table.loc[dataset, 'K-means'] = detailed_results[dataset]['kmeans']['avg_wasserstein']
    
    # Model combinations
    for model_key in base_models.keys():
        for flow_regime in flow_regimes:
            # Use consistent column naming
            if model_key == 'baseline_16_layers':
                if flow_regime == 'direct':
                    col_name = 'Uniform → Uniform'
                else:
                    col_name = 'Uniform → Unit'
            elif model_key == 'fractional_16_layers':
                if flow_regime == 'direct':
                    col_name = 'Unit → Uniform'
                else:
                    col_name = 'Unit → Unit'
            else:
                col_name = f"{base_models[model_key]['name']} ({flow_regime})"
            result_key = f"{model_key}_{flow_regime}"
            if dataset in detailed_results and result_key in detailed_results[dataset]:
                numeric_table.loc[dataset, col_name] = detailed_results[dataset][result_key]['avg_wasserstein']

print(numeric_table.round(4).to_string())

# Log Wasserstein numeric table
print("\n\nAverage Log Wasserstein Distance:")
log_numeric_table = pd.DataFrame(
    index=datasets,
    columns=column_names,
    dtype=float
)

for dataset in datasets:
    # K-means
    if dataset in detailed_results and 'kmeans' in detailed_results[dataset]:
        log_numeric_table.loc[dataset, 'K-means'] = detailed_results[dataset]['kmeans']['avg_log_wasserstein']
    
    # Model combinations
    for model_key in base_models.keys():
        for flow_regime in flow_regimes:
            # Use consistent column naming
            if model_key == 'baseline_16_layers':
                if flow_regime == 'direct':
                    col_name = 'Uniform → Uniform'
                else:
                    col_name = 'Uniform → Unit'
            elif model_key == 'fractional_16_layers':
                if flow_regime == 'direct':
                    col_name = 'Unit → Uniform'
                else:
                    col_name = 'Unit → Unit'
            else:
                col_name = f"{base_models[model_key]['name']} ({flow_regime})"
            result_key = f"{model_key}_{flow_regime}"
            if dataset in detailed_results and result_key in detailed_results[dataset]:
                log_numeric_table.loc[dataset, col_name] = detailed_results[dataset][result_key]['avg_log_wasserstein']

print(log_numeric_table.round(4).to_string())

# Save numeric tables
numeric_csv_path = tutorial_output_dir / "flow_regime_wasserstein_numeric.csv"
numeric_table.to_csv(numeric_csv_path)
print(f"\nSaved numeric Wasserstein table to: {numeric_csv_path}")

log_numeric_csv_path = tutorial_output_dir / "flow_regime_log_wasserstein_numeric.csv"
log_numeric_table.to_csv(log_numeric_csv_path)
print(f"Saved numeric Log Wasserstein table to: {log_numeric_csv_path}")

# Compute improvement over K-means
print("\n\nImprovement over K-means Baseline (%) - Wasserstein:")
print("-" * 52)

improvement_table = pd.DataFrame(
    index=datasets,
    columns=[col for col in column_names if col != 'K-means'],
    dtype=float
)

for dataset in datasets:
    kmeans_score = numeric_table.loc[dataset, 'K-means']
    for col in improvement_table.columns:
        model_score = numeric_table.loc[dataset, col]
        if pd.notna(kmeans_score) and pd.notna(model_score):
            # Since lower is better, improvement = (kmeans - model) / kmeans * 100
            improvement = (kmeans_score - model_score) / abs(kmeans_score) * 100
            improvement_table.loc[dataset, col] = improvement

print(improvement_table.round(2).to_string())

# Save improvement table
improvement_csv_path = tutorial_output_dir / "flow_regime_improvement.csv"
improvement_table.to_csv(improvement_csv_path)
print(f"\nSaved improvement table to: {improvement_csv_path}")

# Compute improvement over K-means (Log Wasserstein)
print("\n\nImprovement over K-means Baseline (%) - Log Wasserstein:")
print("-" * 56)

log_improvement_table = pd.DataFrame(
    index=datasets,
    columns=[col for col in column_names if col != 'K-means'],
    dtype=float
)

for dataset in datasets:
    kmeans_score = log_numeric_table.loc[dataset, 'K-means']
    for col in log_improvement_table.columns:
        model_score = log_numeric_table.loc[dataset, col]
        if pd.notna(kmeans_score) and pd.notna(model_score):
            # Since lower is better, improvement = (kmeans - model) / abs(kmeans) * 100
            improvement = (kmeans_score - model_score) / abs(kmeans_score) * 100
            log_improvement_table.loc[dataset, col] = improvement

print(log_improvement_table.round(2).to_string())

# Save log improvement table
log_improvement_csv_path = tutorial_output_dir / "flow_regime_log_improvement.csv"
log_improvement_table.to_csv(log_improvement_csv_path)
print(f"\nSaved log improvement table to: {log_improvement_csv_path}")

print("\n" + "=" * 70)
print("Flow Regime Performance Evaluation Complete")
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

# Flow regime comparison
print("\n\n### Flow Regime Analysis ###")

print("\nWasserstein Distance Comparison:")
print("-" * 32)
for model_key in base_models.keys():
    if model_key == 'baseline_16_layers':
        print(f"\nUniform flow model:")
    elif model_key == 'fractional_16_layers':
        print(f"\nUnit flow model:")
    else:
        print(f"\n{base_models[model_key]['name']}:")
    for dataset in datasets:
        # Use consistent column naming
        if model_key == 'baseline_16_layers':
            direct_col = 'Uniform → Uniform'
            frac_col = 'Uniform → Unit'
        elif model_key == 'fractional_16_layers':
            direct_col = 'Unit → Uniform'
            frac_col = 'Unit → Unit'
        else:
            direct_col = f"{base_models[model_key]['name']} (direct)"
            frac_col = f"{base_models[model_key]['name']} (fractional)"
        
        direct_score = numeric_table.loc[dataset, direct_col]
        frac_score = numeric_table.loc[dataset, frac_col]
        
        if pd.notna(direct_score) and pd.notna(frac_score):
            diff = frac_score - direct_score
            better = "uniform" if diff > 0 else "unit"
            print(f"  {dataset}: {better} regime is better by {abs(diff):.4f}")

print("\n\nLog Wasserstein Distance Comparison:")
print("-" * 36)
for model_key in base_models.keys():
    if model_key == 'baseline_16_layers':
        print(f"\nUniform flow model:")
    elif model_key == 'fractional_16_layers':
        print(f"\nUnit flow model:")
    else:
        print(f"\n{base_models[model_key]['name']}:")
    for dataset in datasets:
        # Use consistent column naming
        if model_key == 'baseline_16_layers':
            direct_col = 'Uniform → Uniform'
            frac_col = 'Uniform → Unit'
        elif model_key == 'fractional_16_layers':
            direct_col = 'Unit → Uniform'
            frac_col = 'Unit → Unit'
        else:
            direct_col = f"{base_models[model_key]['name']} (direct)"
            frac_col = f"{base_models[model_key]['name']} (fractional)"
        
        direct_score = log_numeric_table.loc[dataset, direct_col]
        frac_score = log_numeric_table.loc[dataset, frac_col]
        
        if pd.notna(direct_score) and pd.notna(frac_score):
            diff = frac_score - direct_score
            better = "uniform" if diff > 0 else "unit"
            print(f"  {dataset}: {better} regime is better by {abs(diff):.4f}")

print("\n" + "=" * 70)

# Additional evaluation on diverse_snr_moderate dataset for plotting
print("\n\nAdditional Evaluation for Average Performance Plots")
print("=" * 51)

# Set plotting style
set_plotting_style()

def evaluate_models_for_plotting(base_models, flow_regimes, device):
    """Evaluate all model combinations on diverse_snr_moderate dataset for plotting."""
    
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
        kmeans_results = evaluate_kmeans_baseline(diverse_dataset, data_loader, device)
        if kmeans_results:
            all_results['kmeans'] = kmeans_results
            # Save the full results
            torch.save(kmeans_results, kmeans_cache_path)
            print(f"  Saved K-means results to: {kmeans_cache_path}")
    
    # Evaluate each model with each flow regime
    for model_key, model_config in base_models.items():
        for flow_regime in flow_regimes:
            result_key = f"{model_key}_{flow_regime}"
            cache_path = tutorial_output_dir / f"{diverse_dataset}_{result_key}_full_results.pt"
            
            if cache_path.exists():
                print(f"\nLoading cached {model_config['name']} ({flow_regime}) results...")
                all_results[result_key] = torch.load(cache_path, weights_only=False)
            else:
                print(f"\nEvaluating {model_config['name']} ({flow_regime})...")
                model_results = evaluate_model_with_flow_regime(
                    model_key, model_config, diverse_dataset, data_loader, flow_regime, device
                )
                if model_results:
                    all_results[result_key] = model_results
                    # Save the full results
                    torch.save(model_results, cache_path)
                    print(f"  Saved results to: {cache_path}")
    
    return all_results

def compute_average_performance_vs_snr(results, base_models, flow_regimes, snr_bins=10, save_path=None):
    """Compute and plot average performance as a function of SNR for flow regime models."""
    print("\nCreating Average Performance vs SNR Plot")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define SNR bins
    snr_min, snr_max = 3.0, 15.0
    bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Colors for different models - all using solid lines with distinguishable colors
    colors = {
        'kmeans': 'orange',
        'baseline_16_layers_direct': 'blue',
        'baseline_16_layers_fractional': 'cyan',
        'fractional_16_layers_direct': 'red',
        'fractional_16_layers_fractional': 'magenta'
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
                linewidth=2, label='K-means (inputs)')
    
    # Plot each model combination
    for model_key, model_config in base_models.items():
        for flow_regime in flow_regimes:
            result_key = f"{model_key}_{flow_regime}"
            
            if result_key in results and results[result_key] is not None:
                log_wass = results[result_key]['log_wasserstein']
                snr_db = results[result_key]['snr_db']
                
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
                
                # Create more descriptive labels
                if model_key == 'baseline_16_layers':
                    if flow_regime == 'direct':
                        label = 'Uniform → Uniform'
                    else:
                        label = 'Uniform → Unit'
                elif model_key == 'fractional_16_layers':
                    if flow_regime == 'direct':
                        label = 'Unit → Uniform'
                    else:
                        label = 'Unit → Unit'
                else:
                    label = f"{model_config['name']} ({flow_regime})"
                
                color = colors.get(result_key, 'black')
                
                ax.plot(bin_centers[valid_mask], avg_wass[valid_mask], 
                        color=color, linestyle='-', linewidth=2,
                        marker='o', markersize=6, label=label)
                
                print(f"\n{label}:")
                print(f"  Average Wasserstein range: [{np.nanmin(avg_wass):.4f}, {np.nanmax(avg_wass):.4f}]")
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Average Wasserstein Distance')
    ax.set_title('Flow Regime Model Performance vs SNR')
    ax.set_xlim(snr_min - 0.5, snr_max + 0.5)
    ax.set_yscale('log')  # Use log scale for y-axis
    format_axis_with_grid(ax)
    format_legend(ax)
    
    if save_path:
        save_figure(fig, save_path)
        print(f"\nSaved average performance plot to: {save_path}")
    
    return fig

# Run the additional evaluation
diverse_results = evaluate_models_for_plotting(base_models, flow_regimes, device)

# Save all diverse dataset results
diverse_results_path = tutorial_output_dir / "diverse_dataset_full_results.pt"
torch.save(diverse_results, diverse_results_path)
print(f"\nSaved all diverse dataset results to: {diverse_results_path}")

# Create average performance plot
avg_perf_plot = compute_average_performance_vs_snr(
    diverse_results,
    base_models,
    flow_regimes,
    snr_bins=10,
    save_path=tutorial_output_dir / "flow_regime_avg_performance_vs_snr.png"
)

print("\n" + "=" * 70)
print("Flow Regime Performance Evaluation with Plots Complete")
print("=" * 70)