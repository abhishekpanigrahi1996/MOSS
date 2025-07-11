"""
GMM Model Evaluation Tutorial - Part 16: Sample Size Performance Table
=====================================================================

This script evaluates baseline_16_layers model performance with different 
numbers of points per GMM sample:
- 100 points
- 500 points
- 1000 points
- 2000 points

This shows how sample size affects model performance across different SNR levels.
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 16: Sample Size Performance")
print("=" * 70)
print()

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

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

# No need for additional imports - create_data_loader handles sample size

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('/mount/Storage/gmm-v4/output')
experiment_base_dir = output_dir / 'final_experiments'
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output/sample_size_performance')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configuration
base_datasets = ["high_snr_fixed", "average_snr_fixed", "low_snr_fixed"]
sample_sizes = [100, 500, 1000, 2000]
batch_size = 32  # Batch size for evaluation
total_samples = 4096  # Total number of GMM samples to evaluate

print("Evaluation Configuration")
print("-" * 24)
print(f"Base datasets: {', '.join(base_datasets)}")
print(f"Sample sizes: {sample_sizes}")
print(f"Total GMM samples per config: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Device: {device}")
print()

# Model to evaluate
model_name = "baseline_16_layers"
model_path = experiment_base_dir / model_name


def get_cache_path(dataset_name, num_points, metric_type):
    """Get cache file path for evaluation results."""
    return tutorial_output_dir / f"{dataset_name}_{num_points}pts_{metric_type}_results.pt"

def evaluate_model_with_sample_size(model, dataset_name, num_points, batch_size, total_samples, device):
    """Evaluate model on dataset with specific number of points per sample."""
    cache_path = get_cache_path(dataset_name, num_points, "model")
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Evaluating on {dataset_name} with {num_points} points per sample...")
    
    try:
        # Create data loader with specific number of points
        data_loader = create_data_loader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            total_samples=total_samples,
            points_per_gmm=num_points,
            device=device,
            fixed_data=True,
            base_seed=42
        )
        
        # Evaluate dataset
        eval_results_list = evaluate_dataset(
            model, 
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
            'std_log_wasserstein': np.std(all_log_wass),
            'num_points': num_points
        }
        
        # Cache results
        torch.save(results, cache_path)
        print(f"  Cached results to: {cache_path}")
        
        return results
        
    except Exception as e:
        print(f"  Error evaluating with {num_points} points: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_kmeans_baseline(dataset_name, num_points, batch_size, total_samples, device):
    """Evaluate K-means baseline with specific number of points per sample."""
    cache_path = get_cache_path(dataset_name, num_points, "kmeans")
    
    # Try to load cached results
    if cache_path.exists():
        print(f"  Loading cached K-means results from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"  Computing K-means baseline for {dataset_name} with {num_points} points...")
    
    try:
        # Create data loader with specific number of points
        data_loader = create_data_loader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            total_samples=total_samples,
            points_per_gmm=num_points,
            device=device,
            fixed_data=True,
            base_seed=42
        )
        
        # Load any model to use for K-means evaluation
        temp_model_path = experiment_base_dir / "baseline_16_layers"
        temp_model, _ = load_model_from_experiment(temp_model_path, load_best=False, device=device)
        
        # Evaluate with K-means only
        eval_results_list = evaluate_dataset(
            temp_model, 
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
            'std_log_wasserstein': np.std(all_log_kmeans_wass),
            'num_points': num_points
        }
        
        # Cache results
        torch.save(results, cache_path)
        print(f"  Cached K-means results to: {cache_path}")
        
        return results
        
    except Exception as e:
        print(f"  Error computing K-means with {num_points} points: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main evaluation
print(f"\nLoading {model_name} model...")
model, _ = load_model_from_experiment(model_path, load_best=False, device=device)
print(f"Model loaded successfully")

print("\nEvaluating Model with Different Sample Sizes")
print("=" * 45)

# Create column names
column_names = ['K-means'] + [f'{pts} points' for pts in sample_sizes]

# Initialize results tables for each dataset
all_results = {}

for dataset_name in base_datasets:
    print(f"\n\nDataset: {dataset_name}")
    print("=" * (9 + len(dataset_name)))
    
    # Initialize tables for this dataset
    results_table = pd.DataFrame(
        index=[f'{pts} points' for pts in sample_sizes],
        columns=['K-means', model_name]
    )
    
    log_results_table = pd.DataFrame(
        index=[f'{pts} points' for pts in sample_sizes],
        columns=['K-means', model_name]
    )
    
    # Store detailed results
    detailed_results = {}
    
    # Evaluate for each sample size
    for num_points in sample_sizes:
        print(f"\n  Sample size: {num_points} points")
        print("  " + "-" * (13 + len(str(num_points))))
        
        row_name = f'{num_points} points'
        
        # Evaluate K-means baseline
        kmeans_results = evaluate_kmeans_baseline(
            dataset_name, num_points, batch_size, total_samples, device
        )
        
        if kmeans_results:
            avg_loss = kmeans_results['avg_wasserstein']
            std_loss = kmeans_results['std_wasserstein']
            avg_log_loss = kmeans_results['avg_log_wasserstein']
            std_log_loss = kmeans_results['std_log_wasserstein']
            
            results_table.loc[row_name, 'K-means'] = f"{avg_loss:.4f} ± {std_loss:.4f}"
            log_results_table.loc[row_name, 'K-means'] = f"{avg_log_loss:.4f} ± {std_log_loss:.4f}"
            print(f"    K-means - Wasserstein: {avg_loss:.4f} ± {std_loss:.4f}")
            print(f"    K-means - Log Wasserstein: {avg_log_loss:.4f} ± {std_log_loss:.4f}")
            
            if row_name not in detailed_results:
                detailed_results[row_name] = {}
            detailed_results[row_name]['kmeans'] = kmeans_results
        
        # Evaluate model
        model_results = evaluate_model_with_sample_size(
            model, dataset_name, num_points, batch_size, total_samples, device
        )
        
        if model_results:
            avg_loss = model_results['avg_wasserstein']
            std_loss = model_results['std_wasserstein']
            avg_log_loss = model_results['avg_log_wasserstein']
            std_log_loss = model_results['std_log_wasserstein']
            
            results_table.loc[row_name, model_name] = f"{avg_loss:.4f} ± {std_loss:.4f}"
            log_results_table.loc[row_name, model_name] = f"{avg_log_loss:.4f} ± {std_log_loss:.4f}"
            print(f"    {model_name} - Wasserstein: {avg_loss:.4f} ± {std_loss:.4f}")
            print(f"    {model_name} - Log Wasserstein: {avg_log_loss:.4f} ± {std_log_loss:.4f}")
            
            if row_name not in detailed_results:
                detailed_results[row_name] = {}
            detailed_results[row_name]['model'] = model_results
    
    # Store results for this dataset
    all_results[dataset_name] = {
        'wasserstein_table': results_table,
        'log_wasserstein_table': log_results_table,
        'detailed_results': detailed_results
    }
    
    # Print tables for this dataset
    print(f"\n\n{dataset_name} - Wasserstein Distance (mean ± std):")
    print(results_table.to_string())
    
    print(f"\n\n{dataset_name} - Log Wasserstein Distance (mean ± std):")
    print(log_results_table.to_string())
    
    # Save CSV files for this dataset
    csv_path = tutorial_output_dir / f"{dataset_name}_wasserstein_table.csv"
    results_table.to_csv(csv_path)
    print(f"\nSaved {dataset_name} Wasserstein table to: {csv_path}")
    
    log_csv_path = tutorial_output_dir / f"{dataset_name}_log_wasserstein_table.csv"
    log_results_table.to_csv(log_csv_path)
    print(f"Saved {dataset_name} Log Wasserstein table to: {log_csv_path}")

# Create summary across all datasets
print("\n\n" + "=" * 70)
print("Summary Across All Datasets")
print("=" * 70)

# Create combined numeric tables
for metric in ['wasserstein', 'log_wasserstein']:
    print(f"\n\n### {metric.replace('_', ' ').title()} Summary ###")
    
    # Create multi-index dataframe
    summary_data = []
    for dataset in base_datasets:
        for num_points in sample_sizes:
            row_name = f'{num_points} points'
            detailed = all_results[dataset]['detailed_results'][row_name]
            
            row_data = {
                'Dataset': dataset,
                'Sample Size': num_points,
                'K-means': detailed['kmeans'][f'avg_{metric}'],
                model_name: detailed['model'][f'avg_{metric}'],
                'Improvement (%)': ((detailed['kmeans'][f'avg_{metric}'] - 
                                   detailed['model'][f'avg_{metric}']) / 
                                  abs(detailed['kmeans'][f'avg_{metric}']) * 100)
            }
            summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    pivot_table = summary_df.pivot_table(
        index='Sample Size',
        columns='Dataset',
        values=['K-means', model_name, 'Improvement (%)']
    )
    
    print(pivot_table.round(4).to_string())
    
    # Save summary
    summary_path = tutorial_output_dir / f"summary_{metric}.csv"
    pivot_table.to_csv(summary_path)
    print(f"\nSaved {metric} summary to: {summary_path}")

# Analyze effect of sample size
print("\n\n### Sample Size Effect Analysis ###")
print("-" * 35)

for dataset in base_datasets:
    print(f"\n{dataset}:")
    
    # Extract values for analysis
    kmeans_wass = []
    model_wass = []
    kmeans_log_wass = []
    model_log_wass = []
    
    for num_points in sample_sizes:
        row_name = f'{num_points} points'
        detailed = all_results[dataset]['detailed_results'][row_name]
        kmeans_wass.append(detailed['kmeans']['avg_wasserstein'])
        model_wass.append(detailed['model']['avg_wasserstein'])
        kmeans_log_wass.append(detailed['kmeans']['avg_log_wasserstein'])
        model_log_wass.append(detailed['model']['avg_log_wasserstein'])
    
    # Calculate relative changes
    print("\n  Wasserstein distance change from 100 to 2000 points:")
    kmeans_change = (kmeans_wass[-1] - kmeans_wass[0]) / kmeans_wass[0] * 100
    model_change = (model_wass[-1] - model_wass[0]) / model_wass[0] * 100
    print(f"    K-means: {kmeans_change:+.1f}%")
    print(f"    {model_name}: {model_change:+.1f}%")
    
    print("\n  Log Wasserstein distance change from 100 to 2000 points:")
    kmeans_log_change = (kmeans_log_wass[-1] - kmeans_log_wass[0]) / abs(kmeans_log_wass[0]) * 100
    model_log_change = (model_log_wass[-1] - model_log_wass[0]) / abs(model_log_wass[0]) * 100
    print(f"    K-means: {kmeans_log_change:+.1f}%")
    print(f"    {model_name}: {model_log_change:+.1f}%")

print("\n" + "=" * 70)
print("Sample Size Performance Evaluation Complete")
print("=" * 70)

# Save all results
all_results_path = tutorial_output_dir / "all_results.pt"
torch.save(all_results, all_results_path)
print(f"\nSaved all results to: {all_results_path}")