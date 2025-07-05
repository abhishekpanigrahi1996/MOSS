"""
Implementation of the Wasserstein distance using SciPy.

Exact Wasserstein distance using SciPy's linear_sum_assignment.
Requires equal point counts and uniform weights.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

try:
    import scipy
    from scipy.optimize import linear_sum_assignment
    _has_scipy = True
except ImportError:
    _has_scipy = False
    import warnings
    warnings.warn("SciPy not found. To use SciPy-based Wasserstein distances, install it with: pip install scipy")

# Import from utils
from ..utils import validate_inputs
from .base import ExactWassersteinBase, register_exact_implementation


class ScipyExactWasserstein(ExactWassersteinBase):
    """
    Exact Wasserstein distance implementation using SciPy.
    
    Uses SciPy's linear_sum_assignment (Hungarian algorithm).
    Requires equal point counts and uniform weights.
    """
    
    def __init__(self):
        """Initialize the SciPy-based exact Wasserstein distance module."""
        super().__init__()
        
        if not _has_scipy:
            raise ImportError(
                "SciPy not found. To use SciPy-based Wasserstein distances, "
                "install it with: pip install scipy"
            )
    
    def forward(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor] = None,
        y_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute exact Wasserstein distance between batches of point clouds.
        
        Requires equal point counts and uniform weights.
        
        Args:
            x: First batch of points [batch_size, n_points, dim]
            y: Second batch of points [batch_size, n_points, dim]
            x_weights: Weights for x points [batch_size, n_points]
            y_weights: Weights for y points [batch_size, n_points]
            
        Returns:
            Exact Wasserstein distances [batch_size]
        """
        # Validate and normalize inputs - SciPy implementation requires equal counts and uniform weights
        x, y, x_weights, y_weights = validate_inputs(
            x, y, x_weights, y_weights, 
            require_uniform_weights_and_equal_points=True,
            implementation_name="SciPy linear_sum_assignment"
        )
        
        # Use torch.cdist for more efficient distance computation
        cost_matrices = torch.cdist(x, y, p=2).pow(2)  # Squared Euclidean distance
        
        # Process each batch individually
        batch_size = x.shape[0]
        device = x.device
        wasserstein_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Get the cost matrix for this batch
            cost_matrix = cost_matrices[i].detach().cpu().numpy()
            
            # Compute the optimal assignment using the Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Extract costs directly from the original cost matrix on GPU
            # Maintain gradient flow through cost matrix
            row_indices_tensor = torch.tensor(row_indices, device=device, dtype=torch.long)
            col_indices_tensor = torch.tensor(col_indices, device=device, dtype=torch.long)
            
            # Compute mean cost for the optimal assignment
            # Since weights are uniform, we can simply average the costs
            optimal_costs = cost_matrices[i][row_indices_tensor, col_indices_tensor]
            wasserstein_distances[i] = optimal_costs.mean()
        
        return wasserstein_distances


# For backward compatibility
def compute_wasserstein_scipy(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Compute Wasserstein distance using SciPy's implementation.
    
    This function provides backward compatibility with the old API.
    
    Args:
        x: First batch of points of shape [batch_size, n_points, dim]
        y: Second batch of points of shape [batch_size, n_points, dim]
        x_weights: Optional weights for x points
        y_weights: Optional weights for y points
        
    Returns:
        Wasserstein distances for each batch of shape [batch_size,]
    """
    # Create the implementation and use it
    implementation = ScipyExactWasserstein()
    return implementation(x, y, x_weights, y_weights)


# Register the implementation if SciPy is available
if _has_scipy:
    register_exact_implementation("scipy", ScipyExactWasserstein)