"""
Wasserstein distance utilities and loss functions.

This module provides functions to compute Wasserstein distances between
point clouds and implements them as PyTorch loss functions.

The Wasserstein distance (also known as Earth Mover's Distance) measures
the minimum "work" required to transform one probability distribution into another.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

# Import from reorganized implementation
from .wasserstein_impl import (
    get_exact_implementation,
    get_regularized_implementation,
    list_available_implementations
)
from .wasserstein_impl.backends import _has_jax, _has_pot, _has_scipy, get_available_backends
from .utils import validate_inputs, validate_cluster_inputs, compute_weights_from_labels


def compute_wasserstein_distance(
    x: torch.Tensor,  # shape [batch_size, n_points_x, dim]
    y: torch.Tensor,  # shape [batch_size, n_points_y, dim]
    x_weights: Optional[torch.Tensor] = None,  # shape [batch_size, n_points_x]
    y_weights: Optional[torch.Tensor] = None,  # shape [batch_size, n_points_y]
    implementation: Literal["auto", "pot", "scipy", "jax"] = "auto",
    algorithm: Literal["auto", "exact", "sinkhorn"] = "auto",
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    reduction: Literal["none", "mean", "sum"] = "none",
    **kwargs
) -> torch.Tensor:
    """
    Compute Wasserstein distance between two batches of point clouds.
    
    This function provides a unified interface to different Wasserstein distance
    implementations and algorithms. It automatically selects the best implementation
    based on the data and available libraries.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
        implementation: Implementation to use:
                       - "auto": Automatically select based on data and available libraries
                       - "pot": Python Optimal Transport implementation (supports different point counts and weights)
                       - "scipy": SciPy implementation (faster for equal counts, no weights support)
                       - "jax": JAX implementation (fastest on GPU, requires JAX)
        algorithm: Algorithm to use:
                  - "auto": Automatically select based on data size
                  - "exact": Linear programming-based exact solution
                  - "sinkhorn": Regularized optimal transport with Sinkhorn algorithm
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        Wasserstein distances for each batch, with optional reduction
    """
    # Validate and normalize inputs
    x, y, x_weights, y_weights = validate_inputs(x, y, x_weights, y_weights)
    
    # Auto-select algorithm if needed
    if algorithm == "auto":
        # For small point clouds, exact algorithm is faster
        if x.shape[1] <= 50 and y.shape[1] <= 50:
            algorithm = "exact"
        else:
            algorithm = "sinkhorn"
    
    # Auto-select implementation if needed
    if implementation == "auto":
        available = get_available_backends()
        
        # Check if we need to handle different point counts or non-uniform weights
        equal_counts = x.shape[1] == y.shape[1]
        uniform_x = torch.allclose(x_weights, torch.ones_like(x_weights) / x.shape[1])
        uniform_y = torch.allclose(y_weights, torch.ones_like(y_weights) / y.shape[1])
        uniform_weights = uniform_x and uniform_y
        
        # Select implementation based on requirements and availability
        if _has_jax:
            implementation = "jax"
        elif _has_pot:
            implementation = "pot"
        elif _has_scipy and algorithm == "exact" and equal_counts and uniform_weights:
            implementation = "scipy"
        else:
            raise ImportError(
                "No implementation available for the requested method. "
                f"Available implementations: {', '.join([k for k, v in available.items() if v])}"
            )
    
    # Check requirements for specific implementations
    equal_counts = x.shape[1] == y.shape[1]
    uniform_x = torch.allclose(x_weights, torch.ones_like(x_weights) / x.shape[1])
    uniform_y = torch.allclose(y_weights, torch.ones_like(y_weights) / y.shape[1])
    uniform_weights = uniform_x and uniform_y
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate implementation requirements
    if implementation == "scipy" and algorithm == "exact" and (not equal_counts or not uniform_weights):
        raise ValueError(f"SciPy exact implementation requires equal point counts and uniform weights. "
                        f"Got point counts: {x.shape[1]} and {y.shape[1]}, uniform weights: {uniform_weights}. "
                        f"Use POT implementation for different point counts or non-uniform weights.")

    if implementation == "jax" and algorithm == "exact" and not equal_counts:
        raise ValueError(f"JAX exact implementation requires equal point counts. "
                        f"Got point counts: {x.shape[1]} and {y.shape[1]}. "
                        f"Use POT implementation for different point counts.")
    
    # Get the appropriate implementation
    if algorithm == "exact":
        impl = get_exact_implementation(implementation)
        logger.debug(f"Using exact algorithm with {implementation} backend")
    elif algorithm == "sinkhorn":
        impl = get_regularized_implementation(
            implementation, 
            epsilon=epsilon, 
            max_iterations=max_iterations
        )
        logger.debug(f"Using sinkhorn algorithm with {implementation} backend, epsilon={epsilon}")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'exact' or 'sinkhorn'")
    
    # Compute distance - don't catch errors, let them propagate
    distances = impl(x, y, x_weights, y_weights)
    
    # Apply reduction
    if reduction == 'none':
        return distances
    elif reduction == 'mean':
        return torch.mean(distances)
    elif reduction == 'sum':
        return torch.sum(distances)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def wasserstein_loss_direct(
    x_points: torch.Tensor,  # shape [batch_size, n_points_x, dim]
    y_points: torch.Tensor,  # shape [batch_size, n_points_y, dim]
    implementation: Literal["auto", "pot", "scipy", "jax"] = "auto",
    algorithm: Literal["auto", "exact", "sinkhorn"] = "auto",
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    reduction: Literal["mean", "sum", "none"] = "mean",
    **kwargs
) -> torch.Tensor:
    """
    Compute Wasserstein loss directly between two batches of point clouds.
    
    Args:
        x_points: First batch of points of shape [batch_size, n_points_x, dim]
        y_points: Second batch of points of shape [batch_size, n_points_y, dim]
        implementation: Which implementation to use
        algorithm: Which algorithm to use
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        Wasserstein loss (reduced according to reduction parameter)
    """
    return compute_wasserstein_distance(
        x_points, y_points,
        implementation=implementation, 
        algorithm=algorithm, 
        epsilon=epsilon,
        max_iterations=max_iterations,
        reduction=reduction,
        **kwargs
    )


def wasserstein_loss(
    predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
    labels: torch.Tensor,      # shape [batch_size, seq_len]
    positions: torch.Tensor,   # shape [batch_size, n_positions, dim]
    implementation: Literal["auto", "pot", "scipy", "jax"] = "auto",
    algorithm: Literal["auto", "exact", "sinkhorn"] = "auto",
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    reduction: Literal["mean", "sum", "none"] = "mean",
    **kwargs
) -> torch.Tensor:
    """
    Compute Wasserstein loss between predicted points and target positions.
    
    Args:
        predictions: Predicted points of shape [batch_size, seq_len, dim]
        labels: Target labels of shape [batch_size, seq_len]
        positions: Target positions of shape [batch_size, n_positions, dim]
        implementation: Implementation to use ('auto', 'pot', 'scipy', 'jax')
        algorithm: Algorithm to use ('auto', 'exact', 'sinkhorn')
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        Wasserstein loss (reduced according to reduction parameter)
    """
    # Validate inputs
    validate_cluster_inputs(predictions, positions, labels)
    
    # Compute class weights by counting labels
    positions_weights = compute_weights_from_labels(labels, positions.shape[1])
    
    # Compute Wasserstein distance between predictions and positions
    return compute_wasserstein_distance(
        predictions,
        positions,
        None,              # No prediction weights (uniform)
        positions_weights, # Weighted by class frequency
        implementation=implementation,
        algorithm=algorithm,
        epsilon=epsilon,
        max_iterations=max_iterations,
        reduction=reduction,
        **kwargs
    )


class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss for GMM models.
    
    This module computes the Wasserstein distance between predicted points
    and target positions based on the provided labels.
    """
    
    def __init__(
        self,
        implementation: Literal["auto", "pot", "scipy", "jax"] = "auto",
        algorithm: Literal["auto", "exact", "sinkhorn"] = "auto",
        epsilon: float = 0.01,
        max_iterations: int = 10000,
        reduction: Literal["mean", "sum", "none"] = "mean",
        **kwargs
    ):
        """
        Initialize the Wasserstein loss.
        
        Args:
            implementation: Implementation to use ('auto', 'pot', 'scipy', 'jax')
            algorithm: Algorithm to use ('auto', 'exact', 'sinkhorn')
            epsilon: Regularization parameter for Sinkhorn algorithm
            max_iterations: Maximum number of iterations for Sinkhorn algorithm
            reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        """
        super().__init__()
        self.implementation = implementation
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.reduction = reduction
        self.kwargs = kwargs
    
    def forward(
        self,
        predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
        labels: torch.Tensor,      # shape [batch_size, seq_len]
        positions: torch.Tensor    # shape [batch_size, n_positions, dim]
    ) -> torch.Tensor:
        """
        Compute Wasserstein loss between predicted points and target positions.
        
        Args:
            predictions: Predicted points of shape [batch_size, seq_len, dim]
            labels: Target labels of shape [batch_size, seq_len]
            positions: Target positions of shape [batch_size, n_positions, dim]
            
        Returns:
            Wasserstein loss
        """
        return wasserstein_loss(
            predictions,
            labels,
            positions,
            implementation=self.implementation,
            algorithm=self.algorithm,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
            reduction=self.reduction,
            **self.kwargs
        )


# Legacy classes have been removed in favor of the unified WassersteinLoss implementation


def wasserstein_distance_matrix(
    point_sets: torch.Tensor,  # [n_sets, n_points, dim]
    weights: Optional[torch.Tensor] = None,  # [n_sets, n_points]
    implementation: Literal["auto", "pot", "scipy", "jax"] = "auto",
    algorithm: Literal["auto", "exact", "sinkhorn"] = "auto",
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    **kwargs
) -> torch.Tensor:  # [n_sets, n_sets]
    """
    Compute pairwise Wasserstein distances between multiple point sets.
    
    This function computes the distance between each pair of point sets in the input,
    returning a distance matrix.
    
    Args:
        point_sets: Tensor of multiple point sets of shape [n_sets, n_points, dim]
        weights: Optional weights for points of shape [n_sets, n_points]
        implementation: Which implementation to use
        algorithm: Which algorithm to use
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        
    Returns:
        Distance matrix of shape [n_sets, n_sets]
    """
    n_sets = point_sets.shape[0]
    device = point_sets.device
    
    # Initialize distance matrix
    distance_matrix = torch.zeros((n_sets, n_sets), device=device)
    
    # Compute distances for each pair
    for i in range(n_sets):
        for j in range(i+1, n_sets):
            # Extract the point sets
            x = point_sets[i:i+1]  # Add batch dim: [1, n_points, dim]
            y = point_sets[j:j+1]  # Add batch dim: [1, n_points, dim]
            
            # Extract weights if provided
            x_weights = weights[i:i+1] if weights is not None else None
            y_weights = weights[j:j+1] if weights is not None else None
            
            # Compute distance
            distance = compute_wasserstein_distance(
                x, y, x_weights, y_weights,
                implementation=implementation, 
                algorithm=algorithm, 
                epsilon=epsilon,
                max_iterations=max_iterations,
                **kwargs
            ).item()
            
            # Fill symmetric matrix
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix