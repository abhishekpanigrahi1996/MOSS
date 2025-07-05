"""
Energy distance loss implementation.

The energy distance is a statistical distance between probability distributions.
It's defined as:
    E(P, Q) = 2E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

where X, X' ~ P and Y, Y' ~ P are independent random variables, and ||·|| is a norm.
This implementation uses the Euclidean norm (p=2 by default).

Unlike Wasserstein distance, energy distance doesn't require solving an optimization
problem, making it computationally efficient, especially for large point clouds.
"""

import torch
from typing import Optional, Literal, Tuple

# Import utilities
from .utils import validate_inputs, validate_cluster_inputs, compute_weights_from_labels


def energy_distance(
    x: torch.Tensor,  # shape [batch_size, n_points_x, dim]
    y: torch.Tensor,  # shape [batch_size, n_points_y, dim]
    x_weights: Optional[torch.Tensor] = None,  # shape [batch_size, n_points_x]
    y_weights: Optional[torch.Tensor] = None,  # shape [batch_size, n_points_y]
    p: float = 2.0,  # Power parameter for the distance (p=2 for Euclidean)
    squared: bool = True,  # Whether to return squared energy distance
    reduction: Literal["none", "mean", "sum"] = "none"
) -> torch.Tensor:
    """
    Compute the energy distance between batches of point clouds.
    
    Energy distance is a statistical distance between probability distributions.
    It can be computed using weighted samples from each distribution.
    
    Args:
        x: First batch of points of shape [batch_size, n_points_x, dim]
        y: Second batch of points of shape [batch_size, n_points_y, dim]
        x_weights: Optional weights for x points of shape [batch_size, n_points_x]
        y_weights: Optional weights for y points of shape [batch_size, n_points_y]
        p: Power parameter for distance calculation (p=2 for Euclidean)
        squared: If True, return squared energy distance
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
    
    Returns:
        Energy distances for each batch, with optional reduction
    """
    # Validate and normalize inputs
    x, y, x_weights, y_weights = validate_inputs(x, y, x_weights, y_weights)
    
    # Compute pairwise distances for all batches at once
    # 1. Pairwise distances between x and y: ||x_i - y_j||
    # [batch_size, n_points_x, n_points_y]
    xy_dist = torch.cdist(x, y, p=p)
    
    # 2. Pairwise distances within x: ||x_i - x_j||
    # [batch_size, n_points_x, n_points_x]
    xx_dist = torch.cdist(x, x, p=p)
    
    # 3. Pairwise distances within y: ||y_i - y_j||
    # [batch_size, n_points_y, n_points_y]
    yy_dist = torch.cdist(y, y, p=p)
    
    # 4. Compute weighted average of distances for all batches
    # Create weight matrices: [batch_size, n_points_x, n_points_y]
    x_weights_expanded = x_weights.unsqueeze(2)  # [batch_size, n_points_x, 1]
    y_weights_expanded = y_weights.unsqueeze(1)  # [batch_size, 1, n_points_y]
    xy_weights = x_weights_expanded * y_weights_expanded  # [batch_size, n_points_x, n_points_y]
    
    # Create weight matrices for self-terms
    xx_weights = x_weights_expanded * x_weights.unsqueeze(1)  # [batch_size, n_points_x, n_points_x]
    yy_weights = y_weights_expanded * y_weights.unsqueeze(2)  # [batch_size, n_points_y, n_points_y]
    
    # E[||X - Y||] = Σ_i Σ_j w_i * v_j * ||x_i - y_j||
    cross_term = torch.sum(xy_weights * xy_dist, dim=[1, 2])  # [batch_size]
    
    # E[||X - X'||] = Σ_i Σ_j w_i * w_j * ||x_i - x_j||
    x_term = torch.sum(xx_weights * xx_dist, dim=[1, 2])  # [batch_size]
    
    # E[||Y - Y'||] = Σ_i Σ_j v_i * v_j * ||y_i - y_j||
    y_term = torch.sum(yy_weights * yy_dist, dim=[1, 2])  # [batch_size]
    
    # 5. Combine terms to get energy distance
    # E(P, Q) = 2*E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
    energy_distances = 2 * cross_term - x_term - y_term
    
    # If not squared, take the appropriate root
    if not squared:
        if p == 2:
            energy_distances = torch.sqrt(torch.clamp(energy_distances, min=0.0))
        else:
            # For other p values, take the p-th root
            energy_distances = torch.pow(torch.clamp(energy_distances, min=0.0), 1/p)
    
    # Apply reduction
    if reduction == 'none':
        return energy_distances
    elif reduction == 'mean':
        return torch.mean(energy_distances)
    elif reduction == 'sum':
        return torch.sum(energy_distances)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def energy_loss_direct(
    predicted_points: torch.Tensor,  # shape [batch_size, seq_len, dim]
    true_points: torch.Tensor,       # shape [batch_size, seq_len, dim]
    p: float = 2.0,
    squared: bool = True,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute energy distance directly between predicted points and true points.
    
    Args:
        predicted_points: Predicted points of shape [batch_size, seq_len, dim]
        true_points: True points of shape [batch_size, seq_len, dim]
        p: Power parameter for distance calculation (p=2 for Euclidean)
        squared: If True, return squared energy distance
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        Energy distance loss (reduced according to reduction parameter)
    """
    return energy_distance(
        predicted_points,
        true_points,
        p=p,
        squared=squared,
        reduction=reduction
    )


def energy_loss(
    predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
    labels: torch.Tensor,      # shape [batch_size, seq_len]
    positions: torch.Tensor,   # shape [batch_size, n_positions, dim]
    p: float = 2.0,
    squared: bool = True,
    reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """
    Compute energy distance between predicted points and target positions.
    
    Args:
        predictions: Predicted points of shape [batch_size, seq_len, dim]
        labels: Target labels of shape [batch_size, seq_len]
        positions: Target positions of shape [batch_size, n_positions, dim]
        p: Power parameter for distance calculation (p=2 for Euclidean)
        squared: If True, return squared energy distance
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        Energy distance loss (reduced according to reduction parameter)
    """
    batch_size, n_positions, dim = positions.shape
    
    # Validate inputs
    validate_cluster_inputs(predictions, positions, labels)
    
    # Compute class weights by counting labels
    positions_weights = compute_weights_from_labels(labels, n_positions)
    
    # Compute energy distance between predictions and positions
    return energy_distance(
        predictions,
        positions,
        None,              # No prediction weights (uniform)
        positions_weights, # Weighted by class frequency
        p=p,
        squared=squared,
        reduction=reduction
    )


class EnergyLoss(torch.nn.Module):
    """
    Energy distance loss for GMM models.
    
    This module computes the energy distance between predicted points
    and target positions based on the provided labels.
    """
    
    def __init__(
        self,
        p: float = 2.0,
        squared: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        """
        Initialize the energy distance loss.
        
        Args:
            p: Power parameter for distance calculation (p=2 for Euclidean)
            squared: If True, return squared energy distance
            reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        """
        super().__init__()
        self.p = p
        self.squared = squared
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
        labels: torch.Tensor,      # shape [batch_size, seq_len]
        positions: torch.Tensor    # shape [batch_size, n_positions, dim]
    ) -> torch.Tensor:
        """
        Compute energy distance between predicted points and target positions.
        
        Args:
            predictions: Predicted points of shape [batch_size, seq_len, dim]
            labels: Target labels of shape [batch_size, seq_len]
            positions: Target positions of shape [batch_size, n_positions, dim]
            
        Returns:
            Energy distance loss
        """
        return energy_loss(
            predictions,
            labels,
            positions,
            p=self.p,
            squared=self.squared,
            reduction=self.reduction
        )