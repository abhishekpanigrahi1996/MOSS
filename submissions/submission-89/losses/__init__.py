"""
Loss and metric functions for the model.

This module provides various loss and metric functions for training and evaluation,
including:
- MSE loss
- Wasserstein distance-based losses with various implementations
- Energy distance-based losses
- Utility functions for validation
- Factory functions for unified loss interface
"""

# Import MSE losses 
from .mse import mse_loss, MSELoss
from .mse import mse_loss_direct, mse_loss_with_labels

# Import Energy losses
from .energy import energy_loss, EnergyLoss
from .energy import energy_distance, energy_loss_direct

# Import Wasserstein interface
from .wasserstein import wasserstein_loss, WassersteinLoss
from .wasserstein import (
    compute_wasserstein_distance,
    wasserstein_loss_direct,
    wasserstein_distance_matrix
)

# Import backend availability
from .wasserstein_impl.backends import _has_pot, _has_scipy, _has_jax
from .wasserstein_impl.backends import get_available_backends

# Import utility functions
from .utils import validate_inputs, compute_weights_from_labels

# Unified interface factory functions 

def create_mse_loss():
    """
    Create a unified MSE loss function.
    
    Returns:
        A function that accepts (predictions, targets) where targets can be:
        - Dictionary with 'centers' and 'labels' keys
        - Tuple of (centers, labels)
    """
    def unified_mse_loss(predictions, targets):
        # Unpack targets
        if isinstance(targets, dict) and 'centers' in targets and 'labels' in targets:
            centers, labels = targets['centers'], targets['labels']
        elif isinstance(targets, tuple) and len(targets) == 2:
            centers, labels = targets
        else:
            raise ValueError("Targets must be a dict with 'centers' and 'labels' keys, or a tuple (centers, labels)")
        
        # Call with unpacked parameters
        return mse_loss(predictions, labels, centers)
    
    return unified_mse_loss

def create_energy_loss():
    """
    Create a unified energy loss function.
    
    Returns:
        A function that accepts (predictions, targets) where targets can be:
        - Dictionary with 'centers' and 'labels' keys
        - Tuple of (centers, labels)
    """
    def unified_energy_loss(predictions, targets):
        # Unpack targets
        if isinstance(targets, dict) and 'centers' in targets and 'labels' in targets:
            centers, labels = targets['centers'], targets['labels']
        elif isinstance(targets, tuple) and len(targets) == 2:
            centers, labels = targets
        else:
            raise ValueError("Targets must be a dict with 'centers' and 'labels' keys, or a tuple (centers, labels)")
        
        # Call with unpacked parameters
        return energy_loss(predictions, labels, centers)
    
    return unified_energy_loss

def create_wasserstein_loss(algorithm="exact", epsilon=0.01, max_iterations=100, backend="pot", reduction="mean", use_true_weights=True):
    """
    Create a configured Wasserstein loss function.
    
    Args:
        algorithm: "exact" or "sinkhorn"
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum iterations for Sinkhorn algorithm
        backend: "pot", "scipy", or "jax"
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        use_true_weights: If True, use mixture weights from the model for center weighting
        
    Returns:
        A function that accepts (predictions, targets) where targets is a dictionary 
        with 'centers' and 'labels' keys.
    """
    def unified_wasserstein_loss(predictions, targets, reduction=reduction):
        # Unpack targets - targets is always a dict
        centers = targets['centers']
        labels = targets['labels']
        weights = targets.get('weights', None) if use_true_weights else None
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if we need to handle special cases for exact implementations with uniform weights requirement
        needs_expansion = False
        
        # SciPy exact requires equal points and uniform weights
        if algorithm == "exact" and backend == "scipy":
            needs_expansion = True
            logger.debug("Using expanded centers with SciPy exact implementation")
        
        # JAX exact requires equal points (can handle non-uniform weights)
        if algorithm == "exact" and backend == "jax" and predictions.shape[1] != centers.shape[1]:
            needs_expansion = True
            logger.debug("Using expanded centers with JAX exact implementation")
        
        if needs_expansion:
            # Import here to avoid circular imports
            from .utils import map_cluster_centers_to_points
            
            # Expand centers to match predictions by mapping each label to its corresponding center
            expanded_centers = map_cluster_centers_to_points(predictions, centers, labels)
            
            logger.debug(f"Original centers shape: {centers.shape}, expanded centers shape: {expanded_centers.shape}")
            
            # Use wasserstein_loss_direct which directly compares point clouds
            return wasserstein_loss_direct(
                predictions,
                expanded_centers,
                implementation=backend,
                algorithm=algorithm,
                epsilon=epsilon,
                max_iterations=max_iterations,
                reduction=reduction
            )
        
        # For POT implementation and when use_true_weights is enabled and weights are available
        if weights is not None and use_true_weights:
            # Use weights from the targets when calling compute_wasserstein_distance
            return compute_wasserstein_distance(
                predictions,
                centers,
                x_weights=None,  # No prediction weights (uniform)
                y_weights=weights,  # Use GMM mixture weights from params
                implementation=backend,
                algorithm=algorithm,
                epsilon=epsilon,
                max_iterations=max_iterations,
                reduction=reduction
            )
        else:
            # Standard case without custom weights
            return wasserstein_loss(
                predictions, 
                labels, 
                centers, 
                algorithm=algorithm,
                epsilon=epsilon,
                max_iterations=max_iterations,
                implementation=backend,
                reduction=reduction
            )
    return unified_wasserstein_loss

# Define publicly exposed symbols
__all__ = [
    # Unified interface factory functions
    'create_mse_loss',
    'create_energy_loss',
    'create_wasserstein_loss',
    
    # Original loss interfaces
    'mse_loss',
    'MSELoss',
    'energy_loss',
    'EnergyLoss', 
    'wasserstein_loss',
    'WassersteinLoss',
    
    # Computing functions
    'energy_distance',
    'compute_wasserstein_distance',
    
    # Backward compatibility - MSE
    'mse_loss_direct',
    'mse_loss_with_labels',
    
    # Backward compatibility - Energy
    'energy_loss_direct',
    'energy_distance',
    
    # Wasserstein direct interfaces
    'wasserstein_loss_direct',
    'wasserstein_distance_matrix',
    'get_available_backends',
    
    # Utilities
    'validate_inputs',
    'compute_weights_from_labels',
    
    # Availability flags
    '_has_pot',
    '_has_scipy',
    '_has_jax'
]