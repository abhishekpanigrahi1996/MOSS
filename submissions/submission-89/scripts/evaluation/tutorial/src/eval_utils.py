"""
Evaluation utilities for GMM models.
"""
import torch
import numpy as np
import math
from scipy.special import digamma
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from fast_pytorch_kmeans import KMeans
from losses.wasserstein import wasserstein_loss, compute_wasserstein_distance
from losses.utils import compute_weights_from_labels

def get_flow_prediction(model):
    """
    Extract flow predictor from model and return a callable function.
    
    Args:
        model: The model containing a transformer with flow_predictor
        
    Returns:
        callable: Function that takes SNR values and returns flow predictions.
        If flow_predictor is not defined for the model, returns a function that always returns 1.
    """
    # Check if model has transformer and flow_predictor
    has_flow_predictor = (hasattr(model, 'transformer') and 
                          hasattr(model.transformer, 'flow_predictor'))
    
    if has_flow_predictor:
        FP = model.transformer.flow_predictor
        FP.eval()
        
        def flow_prediction(x):
            device = next(model.parameters()).device
            if isinstance(x, (int, float)):
                x = torch.tensor([x])
            elif not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            
            x = x.to(device)
            target = {'snr_db': x}
            return FP(targets=target, inputs=None)
    else:
        # Return default function that always returns 1
        def flow_prediction(x):
            device = next(model.parameters()).device
            if isinstance(x, (int, float)):
                x = torch.tensor([x])
            elif not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            
            # Return tensor of ones with appropriate shape
            x = x.to(device)
            return torch.ones_like(x)
    
    return flow_prediction


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


def run_kmeans(data, n_clusters, device='cuda'):
    """
    Run KMeans on input data.
    
    Args:
        data: Input points of shape [batch_size, num_samples, dim]
        n_clusters: Number of clusters (must be provided)
        device: Device to run on
        
    Returns:
        Dictionary with centers and labels
    """
    if n_clusters is None:
        raise ValueError("n_clusters must be provided for KMeans clustering")
    
    # Move data to specified device
    data = data.to(device)
    
    # Get dimensions
    batch_size, num_samples, dim = data.shape
    
    # Initialize storage for results
    centers = torch.zeros((batch_size, n_clusters, dim), device=device)
    labels = torch.zeros((batch_size, num_samples), dtype=torch.long, device=device)
    
    # Process each batch item separately
    for i in range(batch_size):
        # Get current batch item
        batch_data = data[i]
        
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels[i] = kmeans.fit_predict(batch_data)
        centers[i] = kmeans.centroids
    
    return {
        'centers': centers,
        'labels': labels
    }


def compute_metrics(results, metrics_list=None):
    """
    Compute metrics based on evaluation results.
    
    Supported metrics:
    - 'entropy': Statistical entropy of the predicted distribution
    - 'log_wasserstein': Log Wasserstein distance between predictions and ground truth
    - 'log_kmeans_wasserstein': Log Wasserstein distance between ground truth and KMeans centers from inputs
    - 'log_pred_kmeans_wasserstein': Log Wasserstein distance between ground truth and KMeans centers from predictions
    
    Args:
        results: Evaluation results dictionary (single batch) or list of dictionaries (multiple batches)
                Must include 'predictions' and the required data for each requested metric:
                - 'entropy': requires 'predictions'
                - 'log_wasserstein': requires 'predictions' and 'targets' with 'centers' and 'labels'
                - 'log_kmeans_wasserstein': requires 'targets' with 'centers' and 'kmeans_results'
                - 'log_pred_kmeans_wasserstein': requires 'targets' with 'centers' and 'pred_kmeans_results'
        metrics_list: List of metric names to compute, or None for defaults
        
    Returns:
        Updated results with metrics added
        
    Raises:
        ValueError: If required data for any requested metric is missing
    """
    # Default metrics if none specified
    if metrics_list is None:
        metrics_list = [
            'entropy',                     # Entropy of predictions
            'log_wasserstein',             # Wasserstein distance between predictions and ground truth
            'log_kmeans_wasserstein',      # Distance between ground truth targets and KMeans centers from input
            'log_pred_kmeans_wasserstein'  # Distance between ground truth targets and KMeans centers from predictions
        ]
    
    # Handle list of result dictionaries (multiple batches)
    if isinstance(results, list):
        return [compute_metrics(batch_result, metrics_list) for batch_result in results]
    
    # Initialize metrics dictionary if not present
    if 'metrics' not in results:
        results['metrics'] = {}
    
    device = results['predictions'].device
    batch_size = results['predictions'].shape[0]
    
    # 1. Compute entropy if requested
    if 'entropy' in metrics_list:
        if 'predictions' not in results:
            raise ValueError("entropy metric requires predictions")
            
        entropy_values = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            entropy = knn_entropy_bias_reduced_torch(
                results['predictions'][i], k=5, device=device, 
                subsample='random', B=3, seed=42
            )
            entropy_values[i] = entropy
        results['metrics']['entropy'] = entropy_values
    
    # 2. Compute log_wasserstein if requested
    if 'log_wasserstein' in metrics_list:
        if 'predictions' not in results:
            raise ValueError("log_wasserstein metric requires predictions")
        if 'targets' not in results or 'centers' not in results['targets'] or 'labels' not in results['targets']:
            raise ValueError("log_wasserstein metric requires predictions and targets with 'centers' and 'labels'")
            
        # Use wasserstein_loss which properly handles computing weights from labels
        wasserstein_dist = wasserstein_loss(
            predictions=results['predictions'],
            labels=results['targets']['labels'],
            positions=results['targets']['centers'],
            implementation="pot",
            algorithm="exact",
            reduction="none"
        )
        
        # Log transform values
        results['metrics']['log_wasserstein'] = torch.log(torch.clamp_min(wasserstein_dist, 1e-10))
    
    # 3. Compute log_kmeans_wasserstein if requested
    if 'log_kmeans_wasserstein' in metrics_list:
        # Check targets
        if 'targets' not in results or 'centers' not in results['targets'] or 'labels' not in results['targets']:
            raise ValueError("log_kmeans_wasserstein metric requires targets with 'centers' and 'labels'")
        
        # Check kmeans_results
        if 'kmeans_results' not in results or 'centers' not in results['kmeans_results'] or 'labels' not in results['kmeans_results']:
            raise ValueError("log_kmeans_wasserstein metric requires kmeans_results with 'centers' and 'labels'")
        
        # Using wasserstein_loss for computing proper weighted Wasserstein distance
        wasserstein_dist = compute_wasserstein_distance(
            results['targets']['centers'],
            results['kmeans_results']['centers'],
            # Get weights based on label counts from targets and kmeans
            compute_weights_from_labels(results['targets']['labels'], results['targets']['centers'].shape[1]),
            compute_weights_from_labels(results['kmeans_results']['labels'], results['kmeans_results']['centers'].shape[1]),
            implementation="pot",
            algorithm="exact",
            reduction="none"
        )
            
        # Log transform values
        results['metrics']['log_kmeans_wasserstein'] = torch.log(torch.clamp_min(wasserstein_dist, 1e-10))
    
    # 4. Compute log_pred_kmeans_wasserstein if requested
    if 'log_pred_kmeans_wasserstein' in metrics_list:
        # Check targets
        if 'targets' not in results or 'centers' not in results['targets'] or 'labels' not in results['targets']:
            raise ValueError("log_pred_kmeans_wasserstein metric requires targets with 'centers' and 'labels'")
        
        # Check pred_kmeans_results
        if 'pred_kmeans_results' not in results or 'centers' not in results['pred_kmeans_results'] or 'labels' not in results['pred_kmeans_results']:
            raise ValueError("log_pred_kmeans_wasserstein metric requires pred_kmeans_results with 'centers' and 'labels'")
            
        # Using wasserstein_loss for computing proper weighted Wasserstein distance
        wasserstein_dist = compute_wasserstein_distance(
            results['targets']['centers'],
            results['pred_kmeans_results']['centers'],
            # Get weights based on label counts from targets and pred_kmeans
            compute_weights_from_labels(results['targets']['labels'], results['targets']['centers'].shape[1]),
            compute_weights_from_labels(results['pred_kmeans_results']['labels'], results['pred_kmeans_results']['centers'].shape[1]),
            implementation="pot",
            algorithm="exact",
            reduction="none"
        )
            
        # Log transform values
        results['metrics']['log_pred_kmeans_wasserstein'] = torch.log(torch.clamp_min(wasserstein_dist, 1e-10))
    
    return results


def evaluate(model, data, snr=None, flow_speed=None, 
             kmeans_on_inputs=False, kmeans_on_predictions=False,
             n_clusters=None, metrics=None, device='cuda', 
             targets=None, **kwargs):
    """
    Core evaluation function with unified interface.
    
    Args:
        model: The model to evaluate (can be None if only running KMeans)
        data: Input data tensor [batch_size, num_samples, dim] (single batch only)
        snr: Optional SNR values to use (torch.Tensor, shape=[batch_size])
        flow_speed: Optional flow speed values (torch.Tensor, shape=[batch_size] or [batch_size, num_layers])
        kmeans_on_inputs: Whether to run KMeans on input data
        kmeans_on_predictions: Whether to run KMeans on model predictions
        n_clusters: Number of clusters for KMeans. Required if kmeans_on_inputs or kmeans_on_predictions is True.
                   Typically extracted from targets['centers'].shape[1]
        metrics: List of metrics to compute or None for defaults
        device: Device to run evaluation on
        targets: Optional ground truth targets for metrics calculation (dict with 'centers', 'labels')
                Required for certain metrics like log_wasserstein when not using KMeans
        **kwargs: Additional evaluation options
        
    Returns:
        A single evaluation results dictionary
    """
    # Ensure model is in eval mode if provided
    if model is not None:
        model.eval()
    
    # Move data to device
    inputs = data.to(device)
    
    # Initialize results dictionary
    results = {
        'inputs': inputs.detach(),
        'snr_values': snr.to(device) if snr is not None else None,
        'flow_speeds': flow_speed.to(device) if flow_speed is not None else None
    }

    # Add targets to results if provided
    if targets is not None:
        # Create targets dictionary in results and copy contents
        results['targets'] = {}
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                results['targets'][key] = value.to(device)
            else:
                results['targets'][key] = value
    
    # Only proceed with model evaluation if a model is provided
    if model is not None:
        # Check if we have either SNR values or flow speeds
        if snr is None and flow_speed is None:
            raise ValueError("Either snr or flow_speed must be provided when using a model")
        
        # Determine if we need to get flow_speed from SNR or use it directly
        # Logic: if flow_speed is provided, use it; otherwise, get it from SNR
        if flow_speed is not None:
            # We have direct flow_speed, validate batch size
            input_batch_size = inputs.shape[0]
            flow_batch_size = flow_speed.shape[0] if hasattr(flow_speed, 'shape') else 1
            
            if input_batch_size != flow_batch_size:
                if flow_batch_size == 1:
                    # If we have a single flow value, repeat it for all inputs
                    flow_speed = flow_speed.repeat(input_batch_size)
                    results['flow_speeds'] = flow_speed.to(device)
                elif input_batch_size == 1:
                    # If we have a single input batch, repeat it for all flow values
                    inputs = inputs.repeat(flow_batch_size, 1, 1)
                    results['inputs'] = inputs.detach()
                else:
                    # If both have multiple elements but don't match, raise an exception
                    raise ValueError(f"Batch size mismatch: inputs shape={inputs.shape} has batch size {input_batch_size}, but flow_speed shape={flow_speed.shape} has batch size {flow_batch_size}. These must match.")
        else:
            # We need to get flow_speed from SNR
            # First validate SNR batch size
            input_batch_size = inputs.shape[0]
            snr_batch_size = snr.shape[0] if hasattr(snr, 'shape') else 1
            
            if input_batch_size != snr_batch_size:
                if snr_batch_size == 1:
                    # If we have a single SNR value, repeat it for all inputs
                    snr = snr.repeat(input_batch_size)
                    results['snr_values'] = snr.to(device)
                elif input_batch_size == 1:
                    # If we have a single input batch, repeat it for all SNR values
                    inputs = inputs.repeat(snr_batch_size, 1, 1)
                    results['inputs'] = inputs.detach()
                else:
                    # If both have multiple elements but don't match, raise an exception
                    raise ValueError(f"Batch size mismatch: inputs shape={inputs.shape} has batch size {input_batch_size}, but SNR values shape={snr.shape} has batch size {snr_batch_size}. These must match.")
            
            # Get flow predictor from model
            flow_predictor = get_flow_prediction(model)
            
            # Use the flow predictor to convert SNR to flow speeds
            flow_speed = flow_predictor(snr.to(device))
            
            # Update results
            results['flow_speeds'] = flow_speed
        
        # Run model inference
        with torch.no_grad():
            # Always use flow_speeds (either provided directly or derived from SNR)
            predictions = model(inputs, flow_speed=flow_speed.to(device))
        
        # Store predictions
        results['predictions'] = predictions
    else:
        # If no model is provided, we can only run KMeans
        if not (kmeans_on_inputs or kmeans_on_predictions):
            raise ValueError("When no model is provided, at least one of kmeans_on_inputs or kmeans_on_predictions must be True")
        if kmeans_on_predictions:
            raise ValueError("Cannot run kmeans_on_predictions when no model is provided")
    
    # Run KMeans if requested
    if kmeans_on_inputs:
        # Check if we have n_clusters
        if n_clusters is None:
            # Try to get n_clusters from targets if available
            if targets is not None and 'centers' in targets:
                n_clusters = targets['centers'].shape[1]
            else:
                raise ValueError("n_clusters must be provided when kmeans_on_inputs=True")
        results['kmeans_results'] = run_kmeans(results['inputs'], n_clusters=n_clusters, device=device)
    
    if kmeans_on_predictions:
        # Check if we have n_clusters
        if n_clusters is None:
            # Try to get n_clusters from targets if available
            if targets is not None and 'centers' in targets:
                n_clusters = targets['centers'].shape[1]
            else:
                raise ValueError("n_clusters must be provided when kmeans_on_predictions=True")
        results['pred_kmeans_results'] = run_kmeans(results['predictions'], n_clusters=n_clusters, device=device)
    
    # Compute metrics if requested
    if metrics is not None:
        results = compute_metrics(results, metrics)
    
    return results


def evaluate_dataset(model, data_loader, 
                    kmeans_on_inputs=False, kmeans_on_predictions=False,
                    metrics=None, device='cuda', **kwargs):
    """
    Evaluate a model on an entire dataset.
    
    Args:
        model: The model to evaluate
        data_loader: A DataLoader instance providing batches of data
        kmeans_on_inputs: Whether to run KMeans on input data
        kmeans_on_predictions: Whether to run KMeans on model predictions
        metrics: List of metrics to compute or None for defaults
        device: Device to run evaluation on
        **kwargs: Additional evaluation options
        
    Returns:
        A list of evaluation result dictionaries (one per batch)
    """
    # Initialize list to store batch results
    results = []
    
    # Process each batch
    try:
        from tqdm import tqdm
        batch_iterator = tqdm(data_loader, desc="Processing batches", leave=False)
    except ImportError:
        batch_iterator = data_loader
        
    for batch in batch_iterator:
        # Get inputs and targets from DataLoader - always use tuple format
        inputs = batch[0]
        targets_data = batch[1]
        
        # Extract SNR from targets - required
        if 'snr_db' in targets_data:
            snr = targets_data['snr_db']
        else:
            # SNR is required
            raise ValueError("Data loader targets must contain 'snr_db'")
        
        # Extract n_clusters from the targets for KMeans if needed
        n_clusters = None
        if kmeans_on_inputs or kmeans_on_predictions:
            if 'centers' in targets_data:
                n_clusters = targets_data['centers'].shape[1]
            else:
                raise ValueError("Cannot determine n_clusters for KMeans from targets")
        
        # Prepare targets dictionary for evaluation metrics
        targets = {}
        
        # Copy relevant fields from dict
        for key in ['centers', 'labels', 'snr_db']:
            if key in targets_data:
                targets[key] = targets_data[key]
        
        # Evaluate this batch
        batch_results = evaluate(
            model=model, 
            data=inputs, 
            snr=snr, 
            flow_speed=None,  # We're using SNR from the dataset
            kmeans_on_inputs=kmeans_on_inputs,
            kmeans_on_predictions=kmeans_on_predictions,
            n_clusters=n_clusters,
            metrics=metrics,
            device=device,
            targets=targets,
            **kwargs
        )
        
        results.append(batch_results)
    
    return results


def evaluate_with_snr(model, data, snr_values, 
                     kmeans_on_inputs=False, kmeans_on_predictions=False,
                     n_clusters=None, batch_size=None, metrics=None, device='cuda',
                     targets=None, **kwargs):
    """
    Evaluate model with a single input data point duplicated across multiple SNR values.
    
    Args:
        model: Model to evaluate
        data: Input data tensor, either [num_samples, dim] or [1, num_samples, dim]
        snr_values: SNR values to use (torch.Tensor of shape [n_values])
        kmeans_on_inputs: Whether to run KMeans on input data
        kmeans_on_predictions: Whether to run KMeans on model predictions
        n_clusters: Number of clusters for KMeans. Required if kmeans_on_inputs or kmeans_on_predictions is True.
        batch_size: Optional batch size for evaluation (defaults to full size)
        metrics: List of metrics to compute
        device: Device to run on
        targets: Optional ground truth targets for metrics calculation
        **kwargs: Additional evaluation options
        
    Returns:
        Evaluation results where the batch dimension corresponds to different SNR values
    """
    # Ensure data has batch dimension
    if data.dim() == 2:
        data = data.unsqueeze(0)  # Add batch dimension if needed
    
    # If batch_size is None, use the full length of snr_values
    if batch_size is None:
        batch_size = len(snr_values)
    
    results_list = []
    n_values = len(snr_values)
    
    # Process in batches
    for i in range(0, n_values, batch_size):
        # Get current batch of SNR values
        batch_snr_values = snr_values[i:i+batch_size]
        curr_batch_size = len(batch_snr_values)
        
        # Duplicate input data for this batch
        batch_data = data.repeat(curr_batch_size, 1, 1)
        
        # Prepare batch targets if provided
        batch_targets = None
        if targets is not None:
            batch_targets = {}
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    # Handle different dimensions for different target components
                    if key == 'centers':  # [n_clusters, dim]
                        batch_targets[key] = value.unsqueeze(0).repeat(curr_batch_size, 1, 1)
                    elif key == 'labels':  # [num_samples]
                        batch_targets[key] = value.unsqueeze(0).repeat(curr_batch_size, 1)
                    else:  # Other tensor types
                        batch_targets[key] = value.repeat(curr_batch_size)
                else:
                    batch_targets[key] = value
        
        # Evaluate this batch
        batch_results = evaluate(
            model, batch_data, snr=batch_snr_values, 
            kmeans_on_inputs=kmeans_on_inputs, 
            kmeans_on_predictions=kmeans_on_predictions,
            n_clusters=n_clusters, metrics=metrics, device=device,
            targets=batch_targets, **kwargs
        )
        results_list.append(batch_results)
    
    # Return single result if we processed everything in one batch
    if len(results_list) == 1:
        return results_list[0]
    
    # Otherwise return list of batch results
    return results_list


def evaluate_with_flow(model, data, flow_speeds, 
                      kmeans_on_inputs=False, kmeans_on_predictions=False,
                      n_clusters=None, batch_size=None, metrics=None, device='cuda',
                      targets=None, **kwargs):
    """
    Evaluate model with a single input data point duplicated across multiple flow speeds.
    
    Args:
        model: Model to evaluate
        data: Input data tensor, either [num_samples, dim] or [1, num_samples, dim]
        flow_speeds: Flow speeds to use (torch.Tensor of shape [n_values] or [n_values, num_layers])
        kmeans_on_inputs: Whether to run KMeans on input data
        kmeans_on_predictions: Whether to run KMeans on model predictions
        n_clusters: Number of clusters for KMeans. Required if kmeans_on_inputs or kmeans_on_predictions is True.
        batch_size: Optional batch size for evaluation (defaults to full size)
        metrics: List of metrics to compute
        device: Device to run on
        targets: Optional ground truth targets for metrics calculation
        **kwargs: Additional evaluation options
        
    Returns:
        Evaluation results where the batch dimension corresponds to different flow speeds
    """
    # Ensure data has batch dimension
    if data.dim() == 2:
        data = data.unsqueeze(0)  # Add batch dimension if needed
    
    # Determine number of values
    n_values = flow_speeds.shape[0]
    
    # If batch_size is None, use the full length of flow_speeds
    if batch_size is None:
        batch_size = n_values
    
    results_list = []
    
    # Process in batches
    for i in range(0, n_values, batch_size):
        # Get current batch of flow speeds
        batch_flow_speeds = flow_speeds[i:i+batch_size]
        curr_batch_size = len(batch_flow_speeds)
        
        # Duplicate input data for this batch
        batch_data = data.repeat(curr_batch_size, 1, 1)
        
        # Prepare batch targets if provided
        batch_targets = None
        if targets is not None:
            batch_targets = {}
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    # Handle different dimensions for different target components
                    if key == 'centers':  # [n_clusters, dim]
                        batch_targets[key] = value.unsqueeze(0).repeat(curr_batch_size, 1, 1)
                    elif key == 'labels':  # [num_samples]
                        batch_targets[key] = value.unsqueeze(0).repeat(curr_batch_size, 1)
                    else:  # Other tensor types
                        batch_targets[key] = value.repeat(curr_batch_size)
                else:
                    batch_targets[key] = value
        
        # Evaluate this batch
        batch_results = evaluate(
            model, batch_data, flow_speed=batch_flow_speeds, 
            kmeans_on_inputs=kmeans_on_inputs, 
            kmeans_on_predictions=kmeans_on_predictions,
            n_clusters=n_clusters, metrics=metrics, device=device,
            targets=batch_targets, **kwargs
        )
        results_list.append(batch_results)
    
    # Return single result if we processed everything in one batch
    if len(results_list) == 1:
        return results_list[0]
    
    # Otherwise return list of batch results
    return results_list 