"""
Utility functions for Wasserstein distance calculations.

This module provides common utility functions used across different
Wasserstein distance implementations, such as input validation.
"""

import torch
import numpy as np


def validate_inputs(
    x, y, x_weights=None, y_weights=None, 
    require_uniform_weights_and_equal_points=False,
    implementation_name=None
):
    """
    Validate input tensors for Wasserstein distance calculations.
    
    Args:
        x: First point cloud tensor [batch_size, n_points_x, dim]
        y: Second point cloud tensor [batch_size, n_points_y, dim]
        x_weights: Optional weights for first point cloud [batch_size, n_points_x]
        y_weights: Optional weights for second point cloud [batch_size, n_points_y]
        require_uniform_weights_and_equal_points: If True, requires both point clouds
            to have the same number of points and uniform weights.
        implementation_name: Optional name of implementation for error messages.
    
    Returns:
        Tuple of validated (x, y, x_weights, y_weights)
    
    Raises:
        ValueError: If inputs don't meet the required conditions.
    """
    # Basic shape validation
    if len(x.shape) != 3:
        raise ValueError(f"Expected x to have shape [batch_size, n_points, dim], got {x.shape}")
    if len(y.shape) != 3:
        raise ValueError(f"Expected y to have shape [batch_size, n_points, dim], got {y.shape}")
    
    # Ensure same batch size and dimensionality
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Batch sizes must match: {x.shape[0]} != {y.shape[0]}")
    if x.shape[2] != y.shape[2]:
        raise ValueError(f"Point dimensions must match: {x.shape[2]} != {y.shape[2]}")
    
    # Equal points constraint check
    if require_uniform_weights_and_equal_points and x.shape[1] != y.shape[1]:
        impl_msg = f" for {implementation_name}" if implementation_name else ""
        raise ValueError(f"Number of points must match{impl_msg}: {x.shape[1]} != {y.shape[1]}")
    
    # Handle weights
    batch_size, n_points_x, dim = x.shape
    n_points_y = y.shape[1]
    
    # Process x_weights
    if x_weights is None:
        # Create uniform weights
        x_weights = torch.ones(batch_size, n_points_x, device=x.device) / n_points_x
    else:
        # Validate shape
        if x_weights.shape != (batch_size, n_points_x):
            raise ValueError(f"Expected x_weights to have shape ({batch_size}, {n_points_x}), got {x_weights.shape}")
        
        # Normalize weights to sum to 1 per batch
        x_weights = x_weights / x_weights.sum(dim=1, keepdim=True)
    
    # Process y_weights
    if y_weights is None:
        # Create uniform weights
        y_weights = torch.ones(batch_size, n_points_y, device=y.device) / n_points_y
    else:
        # Validate shape
        if y_weights.shape != (batch_size, n_points_y):
            raise ValueError(f"Expected y_weights to have shape ({batch_size}, {n_points_y}), got {y_weights.shape}")
        
        # Normalize weights to sum to 1 per batch
        y_weights = y_weights / y_weights.sum(dim=1, keepdim=True)
    
    # Check for specific implementation constraints
    if require_uniform_weights_and_equal_points:
        # Check uniform weights
        uniform_x = torch.allclose(x_weights, torch.ones_like(x_weights) / n_points_x)
        uniform_y = torch.allclose(y_weights, torch.ones_like(y_weights) / n_points_y)
        
        if not (uniform_x and uniform_y):
            impl_msg = f" for {implementation_name}" if implementation_name else ""
            raise ValueError(f"Uniform weights required{impl_msg}. Please use an implementation that supports non-uniform weights.")
    
    return x, y, x_weights, y_weights


def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        tensor: PyTorch tensor to convert
    
    Returns:
        NumPy array
    """
    if tensor is None:
        return None
    
    # Detach if tensor requires grad
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    # Convert to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy
    return tensor.numpy()


def numpy_to_tensor(array, device=None):
    """
    Convert a NumPy array to a PyTorch tensor.
    
    Args:
        array: NumPy array to convert
        device: Optional target device
    
    Returns:
        PyTorch tensor
    """
    if array is None:
        return None
    
    # Convert to tensor
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.tensor(array)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def compute_cost_matrix(x, y):
    """
    Compute pairwise squared Euclidean distance matrix between two point sets.
    
    Args:
        x: First point cloud tensor [batch_size, n_points_x, dim]
        y: Second point cloud tensor [batch_size, n_points_y, dim]
    
    Returns:
        Cost matrix [batch_size, n_points_x, n_points_y]
    """
    batch_size, n_x, dim = x.shape
    n_y = y.shape[1]
    
    # Reshape for broadcasting
    x_expanded = x.unsqueeze(2)  # [batch_size, n_x, 1, dim]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, n_y, dim]
    
    # Compute squared distances
    distances = ((x_expanded - y_expanded) ** 2).sum(dim=3)  # [batch_size, n_x, n_y]
    
    return distances 