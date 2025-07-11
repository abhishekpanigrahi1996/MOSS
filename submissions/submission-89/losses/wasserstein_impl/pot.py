"""
Implementation of the Wasserstein distance using the Python Optimal Transport (POT) library.

This module provides both exact (linear programming) and approximate (Sinkhorn)
implementations of the Wasserstein distance for batched point clouds.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

try:
    import ot
    _has_pot = True
except ImportError:
    _has_pot = False
    import warnings
    warnings.warn(
        "POT (Python Optimal Transport) package not found. "
        "To use POT-based Wasserstein distances, install it with: pip install POT"
    )

# Import from utils
from ..utils import validate_inputs
from .base import ExactWassersteinBase, RegularizedWassersteinBase, register_exact_implementation, register_regularized_implementation


class PotExactWasserstein(ExactWassersteinBase):
    """
    Exact Wasserstein distance implementation using the POT library.
    
    Uses POT library's EMD solver with PyTorch tensors.
    """
    
    def __init__(self):
        """Initialize the POT-based exact Wasserstein distance module."""
        super().__init__()
        
        if not _has_pot:
            raise ImportError(
                "POT (Python Optimal Transport) package not found. "
                "To use POT-based Wasserstein distances, install it with: pip install POT"
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
        
        Args:
            x: First batch of points [batch_size, n_points_x, dim]
            y: Second batch of points [batch_size, n_points_y, dim]
            x_weights: Weights for x points [batch_size, n_points_x]
            y_weights: Weights for y points [batch_size, n_points_y]
            
        Returns:
            Exact Wasserstein distances [batch_size]
        """
        # Validate and normalize inputs
        x, y, x_weights, y_weights = validate_inputs(
            x, y, x_weights, y_weights, 
            require_uniform_weights_and_equal_points=False,
            implementation_name="POT Exact OT"
        )
        
        # Use torch.cdist for more efficient distance computation
        cost_matrices = torch.cdist(x, y, p=2).pow(2)  # Squared Euclidean distance
        
        # Process each batch individually
        batch_size = x.shape[0]
        device = x.device
        wasserstein_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Get tensors for this batch
            x_w = x_weights[i]
            y_w = y_weights[i]
            cost_matrix = cost_matrices[i]
            
            # Compute the optimal transport plan using ot.emd
            # POT's ot.emd can work directly with PyTorch tensors
            transport_plan = ot.emd(x_w, y_w, cost_matrix)
            
            # Compute the Wasserstein distance using the transport plan
            distance = torch.sum(transport_plan * cost_matrix)
            wasserstein_distances[i] = distance
        
        return wasserstein_distances


class PotRegularizedWasserstein(RegularizedWassersteinBase):
    """
    Regularized Wasserstein distance implementation using the POT library.
    
    Uses Sinkhorn algorithm via POT library with PyTorch tensors.
    """
    
    def __init__(self, epsilon: float = 0.1, max_iterations: int = 1000):
        """
        Initialize the POT-based regularized Wasserstein distance module.
        
        Args:
            epsilon: Regularization parameter for Sinkhorn algorithm
            max_iterations: Maximum number of iterations for Sinkhorn algorithm
        """
        super().__init__(epsilon=epsilon, max_iterations=max_iterations)
        
        if not _has_pot:
            raise ImportError(
                "POT (Python Optimal Transport) package not found. "
                "To use POT-based Wasserstein distances, install it with: pip install POT"
            )
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor] = None,
        y_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regularized Wasserstein distance between batches of point clouds.
        
        Args:
            x: First batch of points [batch_size, n_points_x, dim]
            y: Second batch of points [batch_size, n_points_y, dim]
            x_weights: Weights for x points [batch_size, n_points_x]
            y_weights: Weights for y points [batch_size, n_points_y]
            
        Returns:
            Regularized Wasserstein distances [batch_size]
        """
        # Validate and normalize inputs
        x, y, x_weights, y_weights = validate_inputs(
            x, y, x_weights, y_weights, 
            require_uniform_weights_and_equal_points=False,
            implementation_name="POT Sinkhorn"
        )
        
        # Use torch.cdist for more efficient distance computation
        cost_matrices = torch.cdist(x, y, p=2).pow(2)  # Squared Euclidean distance
        
        # Apply the log weight adjustment to the cost matrices at batch level
        # Create masks for weights that are greater than a small threshold
        x_log_mask = x_weights > 1e-10
        y_log_mask = y_weights > 1e-10
        
        # Initialize log weights tensors
        x_log_weights = torch.zeros_like(x_weights)
        y_log_weights = torch.zeros_like(y_weights)
        
        # Compute log only for weights above threshold
        x_log_weights[x_log_mask] = torch.log(torch.clamp(x_weights[x_log_mask], min=1e-10))
        y_log_weights[y_log_mask] = torch.log(torch.clamp(y_weights[y_log_mask], min=1e-10))
        
        # Reshape for broadcasting to cost_matrices dimensions
        # [batch_size, n_points_x, 1] + [batch_size, 1, n_points_y]
        x_log_weights_reshaped = x_log_weights.unsqueeze(-1)
        y_log_weights_reshaped = y_log_weights.unsqueeze(1)
        
        # Apply subtraction: C_ij -= eps * (log(a_i) + log(b_j))
        cost_matrices -= self.epsilon * (x_log_weights_reshaped + y_log_weights_reshaped)
        
        # Process each batch individually
        batch_size = x.shape[0]
        device = x.device
        wasserstein_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Get tensors for this batch
            x_w = x_weights[i]
            y_w = y_weights[i]
            cost_matrix = cost_matrices[i]
            
            # Use log-domain Sinkhorn for better numerical stability
            transport_plan = ot.bregman.sinkhorn_log(
                x_w, y_w, cost_matrix, 
                reg=self.epsilon,
                numItermax=self.max_iterations,
                stopThr=1e-3
            )
            
            # Compute the transport cost
            transport_cost = torch.sum(transport_plan * cost_matrix)
            
            # Compute entropy term with better numerical stability
            # Use a safe version of xlogy that clamps very small values
            # transport_plan can have very small values, leading to numerical issues
            min_threshold = 1e-10
            
            # Mask out very small values in transport plan to avoid numerical issues
            valid_mask = transport_plan > min_threshold
            masked_plan = transport_plan.clone()
            masked_plan[~valid_mask] = min_threshold  # Set very small values to threshold
            
            # Compute entropy term only for valid entries
            entropy_term = self.epsilon * torch.sum(torch.xlogy(masked_plan, masked_plan))
            
            # For regularized OT, we should include the entropy term
            # Compute <P, C> + ε * KL(P||µ⊗ν)
            wasserstein_distances[i] = transport_cost + entropy_term
        
        return wasserstein_distances


# Register the implementations if POT is available
if _has_pot:
    register_exact_implementation("pot", PotExactWasserstein)
    register_regularized_implementation("pot", PotRegularizedWasserstein)


# Backward compatibility functions

def compute_wasserstein_exact(
    x: torch.Tensor, 
    y: torch.Tensor, 
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Backward compatibility function for the exact Wasserstein distance.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
    
    Returns:
        Wasserstein distances for each batch of shape [batch_size,]
    """
    implementation = PotExactWasserstein()
    return implementation(x, y, x_weights, y_weights)


def compute_wasserstein_sinkhorn(
    x: torch.Tensor, 
    y: torch.Tensor, 
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    **kwargs
) -> torch.Tensor:
    """
    Backward compatibility function for the Sinkhorn Wasserstein distance.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
    
    Returns:
        Wasserstein distances for each batch of shape [batch_size,]
    """
    implementation = PotRegularizedWasserstein(
        epsilon=epsilon,
        max_iterations=max_iterations
    )
    return implementation(x, y, x_weights, y_weights)


def compute_wasserstein_pot(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    algorithm: Literal["exact", "sinkhorn"] = "sinkhorn",
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    **kwargs
) -> torch.Tensor:
    """
    Backward compatibility function for the POT Wasserstein distance.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
        algorithm: Algorithm to use ("exact" or "sinkhorn")
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        
    Returns:
        Wasserstein distances for each batch of shape [batch_size,]
    """
    if algorithm == "exact":
        return compute_wasserstein_exact(x, y, x_weights, y_weights, **kwargs)
    else:  # algorithm == "sinkhorn"
        return compute_wasserstein_sinkhorn(
            x, y, x_weights, y_weights,
            epsilon=epsilon,
            max_iterations=max_iterations,
            **kwargs
        )