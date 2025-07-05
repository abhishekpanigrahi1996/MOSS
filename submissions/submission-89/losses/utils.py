"""
Utility functions for the loss calculations.

This module contains common validation and utility functions used by the various
loss implementations.
"""

import torch
from typing import Dict, Optional, Tuple
import numpy as np


def validate_point_tensors(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> None:
    """Validate that two point tensors have matching shapes."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Target shape {targets.shape} must match prediction shape {predictions.shape}")
    
    if len(predictions.shape) != 3:
        raise ValueError(f"Expected 3D tensor [batch_size, seq_len, dim], got {predictions.shape}")


def validate_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    require_uniform_weights_and_equal_points: bool = False,
    implementation_name: str = "This implementation"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validate and normalize input tensors for Wasserstein distance computation.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
        require_uniform_weights_and_equal_points: If True, enforces both uniform weights and equal point counts
        implementation_name: Name of the implementation for error messages
    
    Returns:
        Tuple of (x, y, x_weights, y_weights) with proper shapes and normalization
    """
    # Check tensor dimensions
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected 3D tensors, got x.dim()={x.dim()}, y.dim()={y.dim()}")
    
    # Extract tensor dimensions
    batch_size, n_points_x, dim = x.shape
    _, n_points_y, d_y = y.shape
    device = x.device
    
    # Check batch size and dimensionality
    if batch_size != y.shape[0]:
        raise ValueError(f"Batch sizes must match: {batch_size} vs {y.shape[0]}")
    if dim != d_y:
        raise ValueError(f"Point dimensionality must match: {dim} vs {d_y}")
    
    # Check for equal point counts if required
    if require_uniform_weights_and_equal_points and n_points_x != n_points_y:
        raise ValueError(f"{implementation_name} requires equal number of points, "
                         f"got {n_points_x} and {n_points_y}.")
    
    # Handle weights for x
    if x_weights is None:
        x_weights = torch.ones(batch_size, n_points_x, device=device) / n_points_x
    else:
        # Validate weight tensor dimensions
        if x_weights.dim() != 2 or x_weights.shape[0] != batch_size or x_weights.shape[1] != n_points_x:
            raise ValueError(f"Invalid x_weights shape: {x_weights.shape}, expected: ({batch_size}, {n_points_x})")
        if torch.any(x_weights < 0):
            raise ValueError("x_weights must be non-negative")
        
        # Normalize weights to sum to 1
        x_weights = x_weights / x_weights.sum(dim=1, keepdim=True)
    
    # Handle weights for y
    if y_weights is None:
        y_weights = torch.ones(batch_size, n_points_y, device=device) / n_points_y
    else:
        # Validate weight tensor dimensions
        if y_weights.dim() != 2 or y_weights.shape[0] != batch_size or y_weights.shape[1] != n_points_y:
            raise ValueError(f"Invalid y_weights shape: {y_weights.shape}, expected: ({batch_size}, {n_points_y})")
        if torch.any(y_weights < 0):
            raise ValueError("y_weights must be non-negative")
        
        # Normalize weights to sum to 1
        y_weights = y_weights / y_weights.sum(dim=1, keepdim=True)
    
    # Check for uniform weights if required
    if require_uniform_weights_and_equal_points and not has_uniform_weights(x_weights, y_weights):
        raise ValueError(f"{implementation_name} does not support non-uniform weights.")
    
    return x, y, x_weights, y_weights


def validate_cluster_inputs(
    predicted_points: torch.Tensor,
    cluster_centers: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = None
) -> None:
    """
    Validate inputs for label-based Wasserstein distance calculation.
    
    Args:
        predicted_points: Predicted points of shape [batch_size, seq_len, dim]
        cluster_centers: True cluster centers of shape [batch_size, n_clusters, dim]
        labels: Cluster labels of shape [batch_size, seq_len]
        n_clusters: Optional number of clusters (if not provided, inferred from cluster_centers)
    """
    # Check tensor dimensions
    if predicted_points.dim() != 3 or cluster_centers.dim() != 3 or labels.dim() != 2:
        raise ValueError("Invalid tensor dimensions")
    
    batch_size, seq_len, dim = predicted_points.shape
    
    # Check batch sizes and dimensions match
    if (cluster_centers.shape[0] != batch_size or labels.shape[0] != batch_size or
            labels.shape[1] != seq_len or predicted_points.shape[2] != cluster_centers.shape[2]):
        raise ValueError("Mismatched tensor shapes")
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = cluster_centers.shape[1]
    
    # Check label values
    max_label = labels.max().item()
    if max_label >= n_clusters:
        raise ValueError(f"Invalid label value {max_label}, max valid is {n_clusters-1}")
    
    # Check all labels are valid
    if not ((labels >= 0) & (labels < n_clusters)).all():
        raise ValueError(f"Labels must be in range [0, {n_clusters-1}]")


def map_cluster_centers_to_points(
    data: torch.Tensor, 
    cluster_centers: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Map cluster centers to each point based on the point's cluster label.

    Args:
        data: Data points of shape [batch_size, seq_len, data_dim]
        cluster_centers: Cluster centers of shape [batch_size, n_clusters, data_dim]
        labels: Cluster labels for each point of shape [batch_size, seq_len]

    Returns:
        Centers for each point with shape [batch_size, seq_len, data_dim]
    """
    batch_size, seq_len = labels.shape
    device = data.device
    
    # Validate inputs
    validate_cluster_inputs(data, cluster_centers, labels)
    
    # Create batch indices for the entire batch
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, seq_len)
    
    # Gather the centers for all points
    return cluster_centers[batch_indices, labels]


def has_uniform_weights(
    x_weights: torch.Tensor,
    y_weights: torch.Tensor
) -> bool:
    """
    Check if weights are uniform (all equal to 1/n_points).
    
    Args:
        x_weights: Weights for x points of shape [batch_size, n_points_x]
        y_weights: Weights for y points of shape [batch_size, n_points_y]
        
    Returns:
        True if both x_weights and y_weights are uniform, False otherwise
    """
    # Check if x_weights are uniform
    n_points_x = x_weights.shape[1]
    uniform_x = torch.ones_like(x_weights) / n_points_x
    if not torch.allclose(x_weights, uniform_x, rtol=1e-5, atol=1e-5):
        return False
    
    # Check if y_weights are uniform
    n_points_y = y_weights.shape[1]
    uniform_y = torch.ones_like(y_weights) / n_points_y
    if not torch.allclose(y_weights, uniform_y, rtol=1e-5, atol=1e-5):
        return False
    
    return True


def compute_weights_from_labels(
    labels: torch.Tensor,  # [batch_size, seq_len]
    n_clusters: int
) -> torch.Tensor:  # [batch_size, n_clusters]
    """
    Compute weights for each cluster based on label frequency.
    
    Args:
        labels: Cluster labels for each point of shape [batch_size, seq_len]
        n_clusters: Total number of clusters
        
    Returns:
        Normalized cluster weights of shape [batch_size, n_clusters]
    """
    batch_size = labels.shape[0]
    device = labels.device
    
    # Create one-hot encoding of labels
    # [batch_size, seq_len, n_clusters]
    labels_one_hot = torch.zeros(batch_size, labels.shape[1], n_clusters, 
                              device=device).scatter_(2, labels.unsqueeze(2), 1)
    
    # Sum over sequence dimension to get counts per cluster
    # [batch_size, n_clusters]
    cluster_weights = labels_one_hot.sum(dim=1)
    
    # Normalize weights to sum to 1
    return cluster_weights / cluster_weights.sum(dim=1, keepdim=True)


def compute_metrics(
    predictions: torch.Tensor, 
    point_targets: torch.Tensor,
    data: torch.Tensor,
    labels: torch.Tensor,
    cluster_centers: torch.Tensor,
    algorithm: str = "sinkhorn",
    epsilon: float = 0.01,
    max_iterations: int = 1000
) -> Dict[str, float]:
    """
    Compute metrics for evaluating predictions.
    
    Args:
        predictions: Predicted values [batch_size, seq_len, dim]
        point_targets: Target values [batch_size, seq_len, dim]
        data: Original data points [batch_size, seq_len, data_dim]
        labels: Point labels [batch_size, seq_len]
        cluster_centers: Cluster centers [batch_size, n_clusters, dim]
        algorithm: Which algorithm to use for Wasserstein distance
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        
    Returns:
        Dictionary of computed metrics
    """
    # Import here to avoid circular imports
    from .mse import mse_loss_direct, mse_loss_with_labels
    from .wasserstein import wasserstein_loss_direct, wasserstein_loss
    
    metrics = {}
    
    # MSE between predictions and point targets
    metrics["mse"] = mse_loss_direct(predictions, point_targets).item()
    metrics["mse_labels"] = mse_loss_with_labels(predictions, cluster_centers, labels).item()
    
    try:
        # Wasserstein distances
        metrics["wasserstein"] = wasserstein_loss_direct(
            predictions, point_targets, algorithm=algorithm, 
            epsilon=epsilon, max_iterations=max_iterations
        ).item()
        
        # Use the new unified interface for labels-based loss
        metrics["wasserstein_labels"] = wasserstein_loss(
            predictions, labels, cluster_centers, algorithm=algorithm,
            epsilon=epsilon, max_iterations=max_iterations
        ).item()
    except Exception as e:
        metrics["wasserstein_error"] = str(e)
    
    return metrics