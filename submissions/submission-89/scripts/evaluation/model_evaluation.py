"""
GMM Model Evaluation Module

This module provides tools and functions for evaluating GMM-based models, 
visualizing predictions, and analyzing metrics like entropy and Wasserstein distance.
"""
import torch
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from scipy.special import digamma
from scipy import stats
from pathlib import Path
import os
from datetime import datetime

# Add project root to path
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Import project-specific modules
from config import get_data_config
from data.loaders import GMMDataLoader
from losses.wasserstein import wasserstein_loss, compute_wasserstein_distance
# Import evaluation utilities
# from model_evaluation_utils import load_model_from_experiment
from kmeans_evaluation import (
    apply_kmeans_to_batch,
    evaluate_kmeans_on_dataloader,
    evaluate_model_with_kmeans
)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define output directory
output_dir = Path('/mount/Storage/gmm-v4/output')
output_folder = Path('/mount/Storage/gmm-v4/output/figures')
output_folder.mkdir(exist_ok=True, parents=True)

def load_model_from_experiment(experiment_dir, load_best=False, device='cuda'):
    """
    Load a model from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        load_best: Whether to load best model or latest model
        device: Device to load model on
        
    Returns:
        tuple: (model, config)
    """
    # This is a stub - the actual implementation would depend on your model loading logic
    # For testing we'll just return None values
    return None, None

# Define consistent plotting styles
def setup_plotting_style():
    """Configure global plotly styling for consistent visuals"""
    # Define color palettes
    COLOR_PALETTE = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
        'input': 'rgba(70, 130, 180, 0.3)',   # Steel blue with transparency
        'predictions': 'rgba(220, 20, 60, 0.7)' # Crimson with transparency
    }
    
    # Define model configurations with consistent styling
    MODEL_CONFIGS = [
        {"name": "16 Layers", "path": "baseline_16_layers", "color": COLOR_PALETTE['primary'][0], "dash": None, "layers": 16},
        {"name": "32 Layers", "path": "baseline_32_layers", "color": COLOR_PALETTE['primary'][1], "dash": 'dash', "layers": 32},
        {"name": "64 Layers", "path": "baseline_64_layers", "color": COLOR_PALETTE['primary'][2], "dash": 'dot', "layers": 64},
        {"name": "Simple 16 layers", "path": "simple_16_layers", "color": COLOR_PALETTE['primary'][3], "dash": None, "layers": 16},
        {"name": "Hard 16 layers", "path": "hard_16_layers", "color": COLOR_PALETTE['primary'][4], "dash": None, "layers": 16}
    ]
    
    # Common layout settings
    LAYOUT_TEMPLATE = {
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'title': {'font': {'size': 16}, 'x': 0.5, 'xanchor': 'center'},
        'legend': {
            'font': {'size': 12},
            'orientation': 'v',
            'yanchor': 'top',
            'y': 0.99,
            'xanchor': 'right',
            'x': 0.99,
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'rgba(0, 0, 0, 0.2)',
            'borderwidth': 1
        },
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'width': 900,  # Default width to fit notebook
        'height': 500, # Default height
        'xaxis': {
            'showgrid': True,
            'gridcolor': 'rgba(230, 230, 230, 1)',
            'gridwidth': 1,
            'title_font': {'size': 14},
            'zeroline': True,
            'zerolinecolor': 'rgba(200, 200, 200, 1)',
            'zerolinewidth': 1
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': 'rgba(230, 230, 230, 1)',
            'gridwidth': 1,
            'title_font': {'size': 14},
            'zeroline': True,
            'zerolinecolor': 'rgba(200, 200, 200, 1)',
            'zerolinewidth': 1
        }
    }
    
    return COLOR_PALETTE, MODEL_CONFIGS, LAYOUT_TEMPLATE

# Initialize the plotting styles
COLOR_PALETTE, MODEL_CONFIGS, LAYOUT_TEMPLATE = setup_plotting_style()

def get_flow_prediction(model):
    """Create a function to extract flow prediction from a model.
    
    Args:
        model: The trained GMM model with a flow predictor
        
    Returns:
        flow_prediction: Function that takes SNR values and returns flow speeds
    """
    FP = model.transformer.flow_predictor
    FP.eval()
    def flow_prediction(x):
        if isinstance(x, (int, float)):
            x = torch.tensor([x])
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        x = x.to(device)
        target = {'snr_db': x}
        return FP(targets=target, inputs=None)
    return flow_prediction

def evaluate_model_with_multiple_conditions(model, data, snr_values=None, flow_speeds=None, device='cuda', batch_size=None):
    """
    Evaluate a GMM model on data points with corresponding SNR values or flow speeds.
    
    Args:
        model: The loaded model (ClusterPredictionModel)
        data: Input data tensor of shape [S, N, d] where:
              - S is number of scenarios (SNR values or flow speeds)
              - N is number of points per scenario
              - d is dimension of each point
        snr_values: Optional tensor of SNR values in dB, shape [S]
        flow_speeds: Optional tensor of flow speeds, shape [S] or [S, num_layers]
        device: Device to run evaluation on ('cuda' or 'cpu')
        batch_size: Optional batch size for processing scenarios
                   If None, process all scenarios in one batch
        
    Returns:
        Dictionary containing:
        - 'input': Original input data [S, N, d]
        - 'predictions': Model predictions [S, N, d]
        - 'snr_values': SNR values used (if provided) [S]
        - 'flow_speeds': Flow speeds used (if provided or computed) [S] or [S, num_layers]
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Move data to device
    data = data.to(device)
    
    # Get dimensions
    S, N, d = data.shape
    
    # Check if we have either SNR values or flow speeds
    if snr_values is None and flow_speeds is None:
        raise ValueError("Either snr_values or flow_speeds must be provided")
    
    # Move tensors to device and verify shapes
    use_flow_speeds = flow_speeds is not None
    
    if use_flow_speeds:
        # Flow speeds take precedence if provided
        flow_speeds = flow_speeds.to(device) if isinstance(flow_speeds, torch.Tensor) else torch.tensor(flow_speeds, device=device)
        if flow_speeds.dim() == 1:
            if len(flow_speeds) != S:
                raise ValueError(f"Number of flow speeds ({len(flow_speeds)}) must match number of scenarios in data ({S})")
        elif flow_speeds.dim() == 2:
            if flow_speeds.shape[0] != S:
                raise ValueError(f"Number of flow speeds ({flow_speeds.shape[0]}) must match number of scenarios in data ({S})")
    else:
        # Use SNR values
        snr_values = snr_values.to(device) if isinstance(snr_values, torch.Tensor) else torch.tensor(snr_values, device=device)
        if len(snr_values) != S:
            raise ValueError(f"Number of SNR values ({len(snr_values)}) must match number of scenarios in data ({S})")
    
    # Initialize storage for predictions and computed flow speeds
    predictions = torch.zeros_like(data)
    computed_flow_speeds = []
    
    # Process in batches if specified, otherwise all at once
    if batch_size is None:
        batch_size = S
    
    for i in range(0, S, batch_size):
        # Get current batch
        end_idx = min(i + batch_size, S)
        batch_data = data[i:end_idx]
        
        # Run model inference
        with torch.no_grad():
            if use_flow_speeds:
                # Use flow speeds directly
                batch_flow = flow_speeds[i:end_idx]
                batch_pred = model(batch_data, flow_speed=batch_flow)
                computed_flow_speeds.append(batch_flow.detach().cpu())
            else:
                # Use SNR values
                batch_snr = snr_values[i:end_idx]
                targets = {'snr_db': batch_snr}
                batch_pred = model(batch_data, targets=targets)
                
                # Capture computed flow speeds
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'compute_flow_speed'):
                    flow = model.transformer.compute_flow_speed(targets=targets, x=batch_data)
                    computed_flow_speeds.append(flow.detach().cpu())
                else:
                    assert False, "Model doesn't have compute_flow_speed method"
            
            # Store predictions
            predictions[i:end_idx] = batch_pred
    
    # Combine flow speeds if captured
    have_computed_flows = len(computed_flow_speeds) > 0
    if have_computed_flows:
        computed_flow_speeds = torch.cat(computed_flow_speeds, dim=0)
    
    # Return results
    results = {
        'input': data.detach().cpu(),
        'predictions': predictions.detach().cpu()
    }
    
    if snr_values is not None:
        results['snr_values'] = snr_values.detach().cpu()
        
    # Handle flow speeds
    if use_flow_speeds:
        results['flow_speeds'] = flow_speeds.detach().cpu()
    elif have_computed_flows:
        results['flow_speeds'] = computed_flow_speeds
        
    return results

def plot_flow_speed_comparison(model_configs):
    """Create a dual plot showing flow speed and total flow vs SNR for different models.
    
    Args:
        model_configs: List of model configuration dictionaries
        
    Returns:
        fig: Plotly figure with the dual plots
    """
    # Create data for plotting
    snr_db = torch.linspace(3, 15, 100)
    snr_db_np = snr_db.cpu().numpy()
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Flow Speed vs SNR', 'Total Flow vs SNR'],
                        horizontal_spacing=0.08)
    
    # Dictionary to store flow speeds for total flow calculation
    flow_speeds = {}
    
    # Load each model and get predictions
    for model_config in model_configs:
        try:
            # Load model
            model_path = output_dir / 'final_experiments' / model_config["path"]
            model, config = load_model_from_experiment(model_path, load_best=False, device=device)
            
            # Get flow prediction
            predictor = get_flow_prediction(model)
            flow_speed = predictor(snr_db).detach().cpu().numpy()
            
            # Store flow speed for total flow calculation
            flow_speeds[model_config["name"]] = flow_speed
            
            # Plot flow speed on the first subplot
            fig.add_trace(
                go.Scatter(
                    x=snr_db_np, 
                    y=flow_speed, 
                    name=model_config["name"],
                    line=dict(color=model_config["color"], dash=model_config["dash"], width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Calculate and plot total flow (flow_speed * num_layers) on the second subplot
            total_flow = flow_speed * model_config["layers"]
            fig.add_trace(
                go.Scatter(
                    x=snr_db_np, 
                    y=total_flow, 
                    name=model_config["name"],
                    line=dict(color=model_config["color"], dash=model_config["dash"], width=2),
                    showlegend=False  # Don't repeat legend entries
                ),
                row=1, col=2
            )
            
        except Exception as e:
            print(f"Error loading model {model_config['path']}: {e}")
    
    # Update axes labels and titles
    fig.update_xaxes(title_text='SNR (dB)', row=1, col=1)
    fig.update_xaxes(title_text='SNR (dB)', row=1, col=2)
    fig.update_yaxes(title_text='Flow Speed', row=1, col=1)
    fig.update_yaxes(title_text='Total Flow (Speed × Layers)', row=1, col=2)
    
    # Apply consistent grid styling to both subplots
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(200, 200, 200, 1)',
        zerolinewidth=1,
        row=1, col=1
    )
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(200, 200, 200, 1)',
        zerolinewidth=1,
        row=1, col=2
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(200, 200, 200, 1)',
        zerolinewidth=1,
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(200, 200, 200, 1)',
        zerolinewidth=1,
        row=1, col=2
    )
    
    # Create a copy of the layout template without conflicting attributes
    layout_settings = {k: v for k, v in LAYOUT_TEMPLATE.items() 
                      if k not in ['legend', 'width', 'height']}
    
    # Apply styling
    fig.update_layout(
        **layout_settings,
        width=1100,  # Wider to accommodate two plots
        height=450,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )
    )
    
    return fig

def get_cached_evaluation(experiment_dir, data_loader, device='cuda', force_recompute=False, 
                          run_kmeans_on_predictions=True, dataset_name='unknown'):
    """
    Get evaluation results for a model, loading from cache if available or computing and caching if not.
    
    Args:
        experiment_dir: Path to experiment directory
        data_loader: GMMDataLoader instance
        device: Device to run evaluation on ('cuda' or 'cpu')
        force_recompute: If True, recompute even if cache exists
        run_kmeans_on_predictions: Whether to run K-means on model predictions
        dataset_name: Name of the dataset being evaluated (used for caching)
        
    Returns:
        tuple: (model, config, results) where results is the evaluation data
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(experiment_dir, 'evaluation_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path - include data loader parameters in filename
    cache_suffix = "_with_pred_kmeans" if run_kmeans_on_predictions else ""
    cache_filename = f"eval_{dataset_name}_bs{data_loader.batch_size}_ns{data_loader.num_samples}{cache_suffix}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Load model
    model, config = load_model_from_experiment(experiment_dir, load_best=False, device=device)
    
    # Check if cache exists and load it
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached evaluation from {cache_path}")
        # Use numpy's savez/load which is more reliable for large numerical data
        cached_data = np.load(cache_path, allow_pickle=True)
        
        # Extract metadata as a python object
        metadata = cached_data['metadata'].item()
        evaluation_timestamp = metadata.get('evaluation_timestamp', 'unknown date')
        dataset_in_cache = metadata.get('dataset_name', 'unknown')
        print(f"Cache loaded successfully. Dataset: {dataset_in_cache}, Evaluated on: {evaluation_timestamp}")
        
        # Convert arrays to a results dict structure
        results = {
            'results': [],
            'evaluation_timestamp': evaluation_timestamp,
            'dataset_name': dataset_in_cache
        }
        
        # Get the number of batches from the cache
        num_batches = metadata.get('num_batches', 0)
        
        # Load each batch result
        for i in range(num_batches):
            batch_result = {}
            # Extract data for this batch
            for key in cached_data.keys():
                if key.startswith(f'batch{i}_'):
                    field_name = key[len(f'batch{i}_'):]
                    batch_result[field_name] = cached_data[key]
            
            if batch_result:  # Only add if we found data
                results['results'].append(batch_result)
        
        return model, config, results
    
    # If we get here, either cache doesn't exist or force_recompute is True
    print(f"Computing evaluation for {experiment_dir} with dataset {dataset_name}...")
    
    # Run evaluation with K-means on predictions if requested
    results = evaluate_model_with_kmeans(
        model=model, 
        data_loader=data_loader, 
        device=device,
        run_kmeans_on_predictions=run_kmeans_on_predictions
    )
    
    # Check if the model provided predictions for K-means
    has_pred_kmeans = any('pred_kmeans_wasserstein_distance' in batch for batch in results)
    print(f"Results include K-means on predictions: {has_pred_kmeans}")
    
    # Add timestamp and metadata
    results_with_meta = {
        'results': results,
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset_name': dataset_name
    }
    
    # Create metadata dict
    metadata = {
        'evaluation_timestamp': results_with_meta['evaluation_timestamp'],
        'num_batches': len(results_with_meta['results']),
        'run_kmeans_on_predictions': run_kmeans_on_predictions,
        'has_pred_kmeans': has_pred_kmeans,
        'dataset_name': dataset_name
    }
    
    # Prepare data to save
    save_dict = {'metadata': metadata}
    
    # Add each batch result with a unique key
    for i, batch_result in enumerate(results_with_meta['results']):
        for key, value in batch_result.items():
            # Process based on type
            if isinstance(value, (np.ndarray, np.number, float, int, bool)):
                save_dict[f'batch{i}_{key}'] = value
            elif isinstance(value, torch.Tensor):
                save_dict[f'batch{i}_{key}'] = value.detach().cpu().numpy()
            # Skip other types that can't be serialized
    
    # Save using numpy's savez
    np.savez(cache_path, **save_dict)
    print(f"Evaluation complete and cached to {cache_path}")
    
    return model, config, results_with_meta

def knn_entropy_bias_reduced_torch(X, k=3, device="cpu", B=1, subsample="random", seed=None, eps_min=1e-12):
    """
    Bias-reduced Kozachenko–Leonenko entropy estimator (2 H(n) – mean_b H_b(n/2)).
    Works on CPU or CUDA. Returns entropy in **nats**.
    
    Args:
        X: Input tensor of shape [n, d] where n is number of points and d is dimension
        k: Number of nearest neighbors to use (default: 3)
        device: Device to run computation on ('cpu' or 'cuda')
        B: Number of half-samples to average over (default: 1)
        subsample: Method for subsampling ('random' or 'first_half')
        seed: Random seed for reproducibility
        eps_min: Minimum distance to avoid numerical issues
        
    Returns:
        float: Estimated entropy value
    """
    X = X.to(device, dtype=torch.float32)
    n, d = X.shape
    if n < 2 * k + 2:
        raise ValueError("Need at least 2·k + 2 points.")

    log_Vd = (d / 2) * math.log(math.pi) - math.lgamma(1 + d / 2)

    def H_knn(data):
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

def plot_comparison_metrics(flow_or_snr_values, entropy_values, wasserstein_values, 
                          x_label, title_prefix, entropy_ci=None, wasserstein_ci=None):
    """
    Create a dual-axis plot showing entropy and Wasserstein distance metrics.
    
    Args:
        flow_or_snr_values: Array of flow speeds or SNR values (x-axis)
        entropy_values: Array of entropy values
        wasserstein_values: Array of Wasserstein distance values
        x_label: Label for x-axis ('Flow Speed' or 'SNR (dB)')
        title_prefix: Prefix for the plot title
        entropy_ci: Optional tuple of (lower, upper) confidence intervals for entropy
        wasserstein_ci: Optional tuple of (lower, upper) confidence intervals for Wasserstein
        
    Returns:
        fig: Plotly figure with the dual-axis plot
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add entropy trace on primary y-axis
    entropy_trace = go.Scatter(
        x=flow_or_snr_values,
        y=entropy_values,
        name="Entropy",
        mode="lines+markers",
        line=dict(color=COLOR_PALETTE['primary'][0], width=2),
        marker=dict(size=8)
    )
    fig.add_trace(entropy_trace, secondary_y=False)
    
    # Add confidence interval for entropy if provided
    if entropy_ci is not None:
        entropy_ci_lower, entropy_ci_upper = entropy_ci
        fig.add_trace(
            go.Scatter(
                x=flow_or_snr_values,
                y=entropy_ci_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=flow_or_snr_values,
                y=entropy_ci_lower,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 255, 0.2)',
                fill='tonexty',
                name='95% CI (Entropy)',
            ),
            secondary_y=False
        )
    
    # Add Wasserstein distance trace on secondary y-axis
    wasserstein_trace = go.Scatter(
        x=flow_or_snr_values,
        y=wasserstein_values,
        name="Wasserstein Distance",
        mode="lines+markers",
        line=dict(color=COLOR_PALETTE['primary'][1], width=2),
        marker=dict(size=8, symbol="square")
    )
    fig.add_trace(wasserstein_trace, secondary_y=True)
    
    # Add confidence interval for Wasserstein if provided
    if wasserstein_ci is not None:
        wasserstein_ci_lower, wasserstein_ci_upper = wasserstein_ci
        fig.add_trace(
            go.Scatter(
                x=flow_or_snr_values,
                y=wasserstein_ci_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=flow_or_snr_values,
                y=wasserstein_ci_lower,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='95% CI (Wasserstein)',
            ),
            secondary_y=True
        )
    
    # Create a copy of the layout template without conflicting settings
    # Remove 'title' from layout_settings to avoid conflict
    layout_settings = {k: v for k, v in LAYOUT_TEMPLATE.items() 
                      if k not in ['legend', 'title', 'width', 'height']}
    
    # Apply consistent styling
    fig.update_layout(
        width=900,  # Standard width for notebook display
        height=500,
        **layout_settings,
        title={
            'text': f"{title_prefix} vs {x_label}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_label,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )
    )
    
    # Set y-axis titles with consistent color scheme
    fig.update_yaxes(
        title_text="Entropy", 
        secondary_y=False, 
        color=COLOR_PALETTE['primary'][0],
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1
    )
    
    fig.update_yaxes(
        title_text="Wasserstein Distance", 
        secondary_y=True, 
        color=COLOR_PALETTE['primary'][1],
        showgrid=False  # No grid on secondary y-axis to avoid visual clutter
    )
    
    # Apply consistent x-axis styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 1)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(200, 200, 200, 1)',
        zerolinewidth=1
    )
    
    return fig

def analyze_model_with_fixed_snr(model, snr_db=8.0, num_flow_speeds=10, points_per_test=1000, num_tests=10):
    """
    Analyze a model's performance with a fixed SNR across different flow speeds.
    
    Args:
        model: The trained GMM model
        snr_db: Fixed SNR value in dB
        num_flow_speeds: Number of different flow speeds to test
        points_per_test: Number of points per test
        num_tests: Number of independent tests for statistical analysis
        
    Returns:
        tuple: (flow_speeds, entropy_results, wasserstein_results)
    """
    # Create a range of flow speeds to evaluate
    flow_speeds = torch.linspace(0.1, 1.0, num_flow_speeds).cuda()
    flow_numpy = flow_speeds.cpu().numpy()
    
    # Configure data generator for fixed SNR
    test_config = get_data_config("low_snr_fixed")
    test_config.sample_count_distribution = {'type': 'fixed', 'value': points_per_test * num_tests}
    test_config.snr_db_params = {'type': 'fixed', 'value': snr_db}
    
    # Create data loader with appropriate sample size
    data_loader = GMMDataLoader(
        config_dict=test_config.model_dump(),
        batch_size=1,
        num_samples=1,
        device='cuda',
        fixed_data=True,
        loader_id=f"flow_analysis_snr{snr_db}"
    )
    
    # Get a batch with all points
    batch_data = next(iter(data_loader))
    input_points = batch_data[0][0]  # [total_points, dim]
    centers = batch_data[1]['centers'][0]  # [n_clusters, dim]
    labels = batch_data[1]['labels'][0]  # [total_points]
    
    # Split into test sets
    split_input_points = torch.split(input_points, points_per_test)
    split_labels = torch.split(labels, points_per_test)
    
    print(f"Split data into {len(split_input_points)} subsets of {len(split_input_points[0])} points each")
    
    # For each flow speed, compute metrics across all test sets
    entropy_values = []
    entropy_stds = []
    wasserstein_values = []
    wasserstein_stds = []
    entropy_ci_low = []
    entropy_ci_high = []
    wasserstein_ci_low = []
    wasserstein_ci_high = []
    
    # For each flow speed
    for i, flow in enumerate(flow_speeds):
        print(f"Processing flow speed = {flow.item():.2f}")
        
        # For each test set
        batch_entropies = []
        batch_wasserstein = []
        
        for j in range(num_tests):
            # Get this test set
            subset_input = split_input_points[j]
            subset_labels = split_labels[j]
            
            # Run model inference
            with torch.no_grad():
                subset_predictions = model(
                    subset_input.unsqueeze(0).to(device),  # Add batch dim
                    flow_speed=flow.unsqueeze(0)          # Add batch dim
                )[0]  # Remove batch dim
                
                # Calculate metrics
                entropy = knn_entropy_bias_reduced_torch(
                    subset_predictions,
                    k=5,
                    device='cpu',
                    subsample='random',
                    B=3,
                    seed=42 + j
                )
                batch_entropies.append(entropy)
                
                # Calculate Wasserstein distance
                pred_batch = subset_predictions.unsqueeze(0)  # Add batch dim
                labels_batch = subset_labels.unsqueeze(0)     # Add batch dim
                centers_batch = centers.unsqueeze(0)          # Add batch dim
                
                wasserstein_dist = wasserstein_loss(
                    predictions=pred_batch.to(device),
                    labels=labels_batch.to(device),
                    positions=centers_batch.to(device),
                    implementation="pot",
                    algorithm="exact",
                    reduction="mean"
                ).item()
                batch_wasserstein.append(wasserstein_dist)
        
        # Calculate statistics
        entropy_mean = np.mean(batch_entropies)
        entropy_std = np.std(batch_entropies)
        wasserstein_mean = np.mean(batch_wasserstein)
        wasserstein_std = np.std(batch_wasserstein)
        
        # Calculate confidence intervals
        t_value = stats.t.ppf(0.975, num_tests - 1)  # 95% CI
        entropy_error = t_value * entropy_std / np.sqrt(num_tests)
        wasserstein_error = t_value * wasserstein_std / np.sqrt(num_tests)
        
        # Store results
        entropy_values.append(entropy_mean)
        entropy_stds.append(entropy_std)
        wasserstein_values.append(wasserstein_mean)
        wasserstein_stds.append(wasserstein_std)
        
        # Store confidence intervals
        entropy_ci_low.append(entropy_mean - entropy_error)
        entropy_ci_high.append(entropy_mean + entropy_error)
        wasserstein_ci_low.append(wasserstein_mean - wasserstein_error)
        wasserstein_ci_high.append(wasserstein_mean + wasserstein_error)
        
        print(f"  Entropy: {entropy_mean:.4f} ± {entropy_error:.4f}")
        print(f"  Wasserstein: {wasserstein_mean:.4f} ± {wasserstein_error:.4f}")
    
    # Create visualization
    metrics_fig = plot_comparison_metrics(
        flow_numpy, 
        entropy_values, 
        wasserstein_values, 
        x_label="Flow Speed", 
        title_prefix="Entropy and Wasserstein Distance",
        entropy_ci=(entropy_ci_low, entropy_ci_high),
        wasserstein_ci=(wasserstein_ci_low, wasserstein_ci_high)
    )
    
    # Add SNR information to title
    metrics_fig.update_layout(title={
        'text': f"Entropy and Wasserstein Distance vs Flow Speed (SNR={snr_db}dB)",
        'x': 0.5,
        'xanchor': 'center'
    })
    
    # Show the figure
    metrics_fig.show()
    
    # Save the figure
    metrics_fig.write_image(output_folder / f'flow_metrics_snr{int(snr_db)}.png', scale=2)
    metrics_fig.write_html(output_folder / f'flow_metrics_snr{int(snr_db)}.html')
    
    # Create correlation plot
    corr_data = pd.DataFrame({
        'Entropy': entropy_values,
        'Wasserstein': wasserstein_values,
        'Flow_Speed': flow_numpy,
        'Entropy_Std': entropy_stds,
        'Wasserstein_Std': wasserstein_stds
    })
    
    # Create figure with Plotly Express
    corr_fig = px.scatter(
        corr_data,
        x='Entropy', 
        y='Wasserstein', 
        color='Flow_Speed',
        color_continuous_scale='plasma',
        error_x=dict(array=entropy_stds),
        error_y=dict(array=wasserstein_stds),
        labels={
            'Entropy': 'Entropy Estimate', 
            'Wasserstein': 'Wasserstein Distance', 
            'Flow_Speed': 'Flow Speed'
        },
        title=f'Relationship Between Entropy and Wasserstein Distance (SNR={snr_db}dB)'
    )
    
    # Create a copy of the layout template without conflicting settings
    layout_settings = {k: v for k, v in LAYOUT_TEMPLATE.items() if k not in ['legend', 'title', 'width', 'height']}
    
    # Update layout with consistent styling
    corr_fig.update_layout(
        **layout_settings,
        width=900,
        height=600,
        title={
            'text': f'Relationship Between Entropy and Wasserstein Distance (SNR={snr_db}dB)',
            'font': {'size': 16},
            'x': 0.5,
            'xanchor': 'center'
        },
        coloraxis_colorbar={
            'title': 'Flow Speed',
            'y': 0.99,
            'x': 0.99,
            'len': 0.8
        },
        xaxis={
            'title': 'Entropy Estimate',
            'showgrid': True,
            'gridcolor': 'rgba(230, 230, 230, 1)',
            'gridwidth': 1,
            'zeroline': True,
            'zerolinecolor': 'rgba(200, 200, 200, 1)',
            'zerolinewidth': 1
        },
        yaxis={
            'title': 'Wasserstein Distance',
            'showgrid': True,
            'gridcolor': 'rgba(230, 230, 230, 1)',
            'gridwidth': 1,
            'zeroline': True,
            'zerolinecolor': 'rgba(200, 200, 200, 1)',
            'zerolinewidth': 1
        }
    )
    
    # Show and save the correlation plot
    corr_fig.show()
    corr_fig.write_image(output_folder / f'correlation_snr{int(snr_db)}.png', scale=2)
    corr_fig.write_html(output_folder / f'correlation_snr{int(snr_db)}.html')
    
    # Return results for further analysis
    return flow_numpy, {
        'entropy_values': entropy_values,
        'entropy_stds': entropy_stds,
        'entropy_ci_low': entropy_ci_low,
        'entropy_ci_high': entropy_ci_high,
        'figure': metrics_fig,
        'correlation_figure': corr_fig
    }, {
        'wasserstein_values': wasserstein_values,
        'wasserstein_stds': wasserstein_stds,
        'wasserstein_ci_low': wasserstein_ci_low,
        'wasserstein_ci_high': wasserstein_ci_high
    }

def analyze_model_with_multiple_k_values(model, data, snr_values=None, flow_speeds=None, device='cuda', k_values=[1, 3, 5, 7]):
    """
    Evaluate entropy estimates with different k-nn values for robustness analysis.
    
    Args:
        model: The trained GMM model
        data: Input data tensor
        snr_values: Optional tensor of SNR values in dB
        flow_speeds: Optional tensor of flow speeds
        device: Device to run evaluation on
        k_values: List of k values to use for entropy estimation
        
    Returns:
        Dictionary containing entropy estimates for each k value
    """
    # Get model predictions
    eval_results = evaluate_model_with_multiple_conditions(
        model=model,
        data=data,
        snr_values=snr_values,
        flow_speeds=flow_speeds,
        device=device
    )
    
    # Initialize storage for entropy values
    entropy_by_k = {k: [] for k in k_values}
    
    # Get parameters for x-axis (either SNR or flow speed)
    if snr_values is not None:
        x_values = snr_values.cpu().numpy()
        x_label = "SNR (dB)"
    else:
        x_values = flow_speeds.cpu().numpy()
        x_label = "Flow Speed"
    
    # Calculate entropy for each condition and each k value
    for i in range(len(eval_results['predictions'])):
        predictions = eval_results['predictions'][i]
        
        # Calculate entropy with different k values
        for k in k_values:
            entropy = knn_entropy_bias_reduced_torch(
                predictions, k=k, device='cpu', subsample='random', B=3, seed=42
            )
            entropy_by_k[k].append(entropy)
    
    # Create a figure to compare entropy estimates with different k values
    fig = go.Figure()
    
    # Add a trace for each k value
    for k in k_values:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=entropy_by_k[k],
                mode='lines+markers',
                name=f'k={k}',
                line=dict(width=2),
                marker=dict(size=8)
            )
        )
    
    # Update layout
    layout_settings = {k: v for k, v in LAYOUT_TEMPLATE.items() 
                      if k not in ['legend', 'title', 'width', 'height']}
    
    fig.update_layout(
        **layout_settings,
        title={
            'text': "Entropy Estimation with Different k-NN Values",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_label,
        yaxis_title="Entropy Estimate",
        width=900,
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return entropy_by_k, fig

def main():
    """Main entry point for testing functions"""
    print("GMM Model Evaluation Module")
    print("This script contains evaluation utilities for GMM models.")
    print("Functions are not executed directly - import this module to use them.")
    
if __name__ == "__main__":
    main()