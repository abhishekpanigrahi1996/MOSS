"""
K-means clustering evaluation for GMM models.

This module provides functions for applying K-means clustering to GMM data
and comparing it with the true clusters.
"""
import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
import numpy as np
from fast_pytorch_kmeans import KMeans

from model_evaluation_utils import load_model_from_experiment
from losses.wasserstein import compute_wasserstein_distance
from losses import create_wasserstein_loss

logger = logging.getLogger(__name__)


def apply_kmeans_to_batch(
    points: torch.Tensor, 
    targets: Dict[str, torch.Tensor],
    n_clusters: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply K-means clustering to a batch of data.
    
    Args:
        points: Points tensor of shape (batch_size, num_points, dim)
        targets: Dictionary with centers, labels, SNR values, etc.
        n_clusters: Number of clusters for K-means (if None, uses number of centers in targets)
        device: Device to run K-means on
        
    Returns:
        Dictionary with tensored results for the entire batch:
        - points: Original points (batch_size, num_points, dim)
        - centers: True centers (batch_size, n_clusters, dim)
        - kmeans_centers: K-means centers (batch_size, n_clusters, dim)
        - kmeans_labels: K-means labels (batch_size, num_points)
        - wasserstein_distance: Wasserstein distances (batch_size,)
        - log_wasserstein_distance: Log of Wasserstein distances (batch_size,)
        - snr_db: SNR values if available (batch_size,)
    """
    # Get device from points if not provided
    if device is None:
        device = points.device
    
    # Extract batch size and ensure points are on the right device
    batch_size = points.shape[0]
    num_points = points.shape[1]
    points = points.to(device)
    
    # Move targets to device
    for key in targets:
        if isinstance(targets[key], torch.Tensor):
            targets[key] = targets[key].to(device)
    
    # Get centers and SNR
    centers = targets['centers']
    
    # Determine number of clusters for each sample
    if n_clusters is None:
        # Use number of true centers from targets
        n_clusters = centers.shape[1]
    
    # Initialize tensors to store results
    kmeans_centers_batch = torch.zeros(batch_size, n_clusters, points.shape[2], device=device)
    kmeans_labels_batch = torch.zeros(batch_size, num_points, dtype=torch.long, device=device)
    wasserstein_distances = torch.zeros(batch_size, device=device)
    log_wasserstein_distances = torch.zeros(batch_size, device=device)
    
    # Process each sample in the batch
    for i in range(batch_size):
        # Get sample points
        sample_points = points[i]  # Shape: [num_points, dim]
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        kmeans_labels = kmeans.fit_predict(sample_points)
        kmeans_centers = kmeans.centroids
        
        # Store K-means results
        kmeans_centers_batch[i] = kmeans_centers
        kmeans_labels_batch[i] = kmeans_labels
        
        # Calculate Wasserstein distance between true centers and K-means centers
        true_centers_tensor = centers[i].unsqueeze(0)  # [1, n_clusters, dim]
        kmeans_centers_tensor = kmeans_centers.unsqueeze(0)  # [1, n_clusters, dim]
        
        # Compute weights for true centers if labels are available
        labels = targets['labels'][i]
        labels_counts = torch.bincount(labels.long(), minlength=n_clusters)
        true_centers_weights = (labels_counts / labels_counts.sum()).unsqueeze(0)  # [1, n_clusters]
        
        # K-means labels for weight calculation
        kmeans_counts = torch.bincount(kmeans_labels.long(), minlength=n_clusters)
        kmeans_weights = (kmeans_counts / kmeans_counts.sum()).unsqueeze(0)  # [1, n_clusters]
        
        # Compute Wasserstein distances using exact algorithm with POT
        w_distance = compute_wasserstein_distance(
            true_centers_tensor, 
            kmeans_centers_tensor,
            x_weights=true_centers_weights,
            y_weights=kmeans_weights,
            implementation="pot",
            algorithm="exact"
        ).item()
        
        # Store Wasserstein distance
        wasserstein_distances[i] = w_distance
        
        # Compute log of Wasserstein distance
        log_w_distance = np.log(w_distance) if w_distance > 0 else -np.inf
        log_wasserstein_distances[i] = log_w_distance
    
    # Create result dictionary with tensors
    results = {
        "points": points,
        "centers": centers,
        "kmeans_centers": kmeans_centers_batch,
        "kmeans_labels": kmeans_labels_batch,
        "kmeans_wasserstein_distance": wasserstein_distances,
        "kmeans_log_wasserstein_distance": log_wasserstein_distances,
    }
    
    # Add labels if available
    if 'labels' in targets:
        results["labels"] = targets['labels']
    
    # Add SNR if available
    if 'snr_db' in targets:
        results["snr_db"] = targets['snr_db']
    
    return results


def evaluate_kmeans_on_dataloader(
    data_loader: DataLoader,
    n_clusters: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply K-means clustering to all batches in a data loader.
    
    Args:
        data_loader: DataLoader providing batches
        n_clusters: Number of clusters for K-means
        device: Device to run K-means on
        
    Returns:
        Dictionary with tensored results for all samples:
        - points: Original points (total_samples, num_points, dim)
        - centers: True centers (total_samples, n_clusters, dim)
        - kmeans_centers: K-means centers (total_samples, n_clusters, dim)
        - kmeans_labels: K-means labels (total_samples, num_points)
        - wasserstein_distance: Wasserstein distances (total_samples,)
        - log_wasserstein_distance: Log of Wasserstein distances (total_samples,)
        - snr_db: SNR values if available (total_samples,)
    """
    # Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_results = []
    
    # Create progress bar
    pbar = tqdm(
        total=len(data_loader), 
        desc="Processing batches", 
        unit="batch"
    )
    
    # Process each batch
    start_time = time.time()
    sample_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Handle different batch formats
        if isinstance(batch, tuple) and len(batch) == 2:
            # Tuple format: (points, targets)
            points, targets = batch
        elif isinstance(batch, dict):
            # Dictionary format, extract points and use the dict as targets
            points = batch['points']
            targets = batch
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
            
        # Apply K-means to this batch
        result = apply_kmeans_to_batch(
            points=points, 
            targets=targets, 
            n_clusters=n_clusters, 
            device=device
        )
        
        # Store batch results
        batch_results.append(result)
        
        # Update sample count
        sample_count += points.shape[0]
        
        # Update progress bar
        pbar.update(1)
        elapsed = time.time() - start_time
        samples_per_sec = sample_count / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "samples": sample_count,
            "samples/s": f"{samples_per_sec:.2f}"
        })
    
    # Close progress bar
    pbar.close()
    

    
    return batch_results


def evaluate_model_with_kmeans(
    model: torch.nn.Module,
    data_loader: DataLoader, 
    n_clusters: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    run_kmeans_on_predictions: bool = True  # New parameter to control whether to run K-means on predictions
) -> Dict[str, torch.Tensor]:
    """
    Evaluate a model and apply K-means clustering to compare results.
    Also optionally applies K-means to model predictions if available.
    
    Args:
        model: Trained GMM model
        data_loader: DataLoader providing evaluation data
        n_clusters: Number of clusters for K-means
        device: Device to run evaluation on
        run_kmeans_on_predictions: Whether to run K-means on model predictions
        
    Returns:
        Dictionary with tensored results for all samples including:
        - Original points and K-means results
        - Model predicted centers
        - K-means results on model predictions (if available)
    """
    # Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is in evaluation mode and on the right device
    model.eval()
    model = model.to(device)
    
    batch_results = []
    
    # Create progress bar
    pbar = tqdm(
        total=len(data_loader), 
        desc="Evaluating model", 
        unit="batch"
    )
    
    # Process each batch
    start_time = time.time()
    sample_count = 0
    loss_fn = create_wasserstein_loss(algorithm="exact", backend="pot", reduction="none", use_true_weights=False)
    
    for batch_idx, batch in enumerate(data_loader):
        # Handle different batch formats
        if isinstance(batch, tuple) and len(batch) == 2:
            # Tuple format: (points, targets)
            points, targets = batch
        elif isinstance(batch, dict):
            # Dictionary format, extract points and use the dict as targets
            points = batch['points']
            targets = batch
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
            
        # Ensure points are on the correct device
        points = points.to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(points, targets)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                predicted_centers = outputs.get('centers', None)
                # Check if the model outputs predicted points as well
                predicted_points = outputs.get('points', None)
            elif isinstance(outputs, torch.Tensor):
                predicted_centers = outputs
                predicted_points = None
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")
        
        # Apply K-means to this batch of original points
        result = apply_kmeans_to_batch(
            points=points, 
            targets=targets, 
            n_clusters=n_clusters, 
            device=device
        )
        
        # Add model's predicted centers and calculate wasserstein distance
        if predicted_centers is not None:
            wasserstein_distance = loss_fn(predicted_centers, targets)
            
            # Ensure values are on CPU before converting to numpy
            if isinstance(wasserstein_distance, torch.Tensor):
                wasserstein_distance = wasserstein_distance.detach().cpu()
                
            result['wasserstein_distance'] = wasserstein_distance
            result['log_wasserstein_distance'] = np.log(wasserstein_distance + 1e-12)
            result['predicted_centers'] = predicted_centers
        
        # Optionally apply K-means to model's predicted points
        if run_kmeans_on_predictions and predicted_centers is not None:
            # Apply K-means to predicted points
            pred_result = apply_kmeans_to_batch(
                points=predicted_centers,
                targets=targets,
                n_clusters=n_clusters,
                device=device
            )
            
            # Add prediction K-means results to the main result with 'pred_' prefix
            result['pred_kmeans_centers'] = pred_result['kmeans_centers']
            result['pred_kmeans_labels'] = pred_result['kmeans_labels']
            result['pred_kmeans_wasserstein_distance'] = pred_result['kmeans_wasserstein_distance']
            result['pred_kmeans_log_wasserstein_distance'] = pred_result['kmeans_log_wasserstein_distance']
        
        # Store batch results
        batch_results.append(result)
        
        # Update sample count
        sample_count += points.shape[0]
        
        # Update progress bar
        pbar.update(1)
        elapsed = time.time() - start_time
        samples_per_sec = sample_count / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "samples": sample_count,
            "samples/s": f"{samples_per_sec:.2f}"
        })
    
    # Close progress bar
    pbar.close()
    
    # Process batch results to ensure they're serializable
    # This avoids issues when pickling tensors
    processed_results = []
    for result in batch_results:
        # Create a new dict with CPU numpy values instead of tensors
        processed_result = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                # Convert tensors to numpy arrays
                processed_result[key] = value.detach().cpu().numpy()
            else:
                processed_result[key] = value
        processed_results.append(processed_result)
    
    # Return processed batch results
    return processed_results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage:
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GMM model with K-means clustering")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--num-clusters", type=int, default=None, help="Number of clusters for K-means")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on")
    parser.add_argument("--load-best", action="store_true", help="Load best model instead of latest")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.experiment_dir}")
    model, config = load_model_from_experiment(args.experiment_dir, args.load_best, args.device)
    
    # Create data loader from config
    from data.loaders.data_loader import GMMDataLoader
    
    # Determine number of samples
    if args.num_samples is None:
        num_samples = 1000
    else:
        num_samples = args.num_samples
    
    logger.info(f"Creating data loader with {num_samples} samples and batch size {args.batch_size}")
    data_loader = GMMDataLoader(
        config_dict=config.data.model_dump(),
        batch_size=args.batch_size,
        num_samples=num_samples,
        device=args.device,
        fixed_data=True,  # Use fixed data for reproducibility
        loader_id="eval"
    )
    
    # Run evaluation
    logger.info("Starting model evaluation with K-means")
    results = evaluate_model_with_kmeans(
        model=model,
        data_loader=data_loader,
        n_clusters=args.num_clusters,
        device=args.device
    )
    
    # Print summary
    print(f"\nProcessed {results['points'].shape[0]} samples")
    wasserstein_distances = results['wasserstein_distance'].cpu().numpy()
    print("Wasserstein distances statistics:")
    print(f"  Min: {np.min(wasserstein_distances):.6f}")
    print(f"  Max: {np.max(wasserstein_distances):.6f}")
    print(f"  Mean: {np.mean(wasserstein_distances):.6f}")
    print(f"  Std: {np.std(wasserstein_distances):.6f}")
    
    # Save results if requested
    if args.save_results:
        import torch
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.experiment_dir) / f"kmeans_evaluation_{timestamp}.pt"
        
        torch.save(results, output_path)
        print(f"Results saved to {output_path}")