"""
MSE loss implementations for GMM models.

This module provides Mean Squared Error loss functions for training and evaluating
models that predict points or cluster centers.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Literal

# Import utilities from the same package
from .utils import validate_point_tensors, validate_cluster_inputs, map_cluster_centers_to_points


def mse_loss_direct(
    predicted_points: torch.Tensor,  # shape [batch_size, seq_len, dim]
    true_points: torch.Tensor        # shape [batch_size, seq_len, dim]
) -> torch.Tensor:
    """
    Compute MSE loss directly between predicted points and true points.
    
    Args:
        predicted_points: Predicted points of shape [batch_size, seq_len, dim]
        true_points: True points of shape [batch_size, seq_len, dim]
        
    Returns:
        MSE loss between predictions and true points, multiplied by the feature dimension
    """
    # Validate inputs
    validate_point_tensors(predicted_points, true_points)
    
    # Extract feature dimension for scaling
    feature_dim = predicted_points.shape[-1]
    
    # Compute MSE scaled by feature dimension
    return F.mse_loss(predicted_points, true_points) * feature_dim


def mse_loss_with_labels(
    predicted_points: torch.Tensor,  # shape [batch_size, seq_len, dim]
    cluster_centers: torch.Tensor,   # shape [batch_size, n_clusters, dim]
    labels: torch.Tensor             # shape [batch_size, seq_len]
) -> torch.Tensor:
    """
    Compute MSE loss between predicted points and their true cluster centers.
    For each point, the true center is determined by its label.
    
    Args:
        predicted_points: Predicted points of shape [batch_size, seq_len, dim]
        cluster_centers: True cluster centers of shape [batch_size, n_clusters, dim]
        labels: Cluster labels of shape [batch_size, seq_len]
        
    Returns:
        MSE loss between predictions and their corresponding true centers,
        multiplied by the feature dimension
    """
    # Validate inputs
    validate_cluster_inputs(predicted_points, cluster_centers, labels)
    
    # Map cluster centers to each point based on labels
    true_centers = map_cluster_centers_to_points(predicted_points, cluster_centers, labels)
    
    # Extract feature dimension for scaling
    feature_dim = predicted_points.shape[-1]
    
    # Compute MSE between predicted points and their true centers, scaled by feature dimension
    return F.mse_loss(predicted_points, true_centers) * feature_dim


def mse_loss(
    predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
    labels: torch.Tensor,       # shape [batch_size, seq_len]
    positions: torch.Tensor,    # shape [batch_size, n_positions, dim]
    reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """
    Compute MSE loss between predicted points and their true positions.
    For each point, the true position is determined by its label.
    
    Args:
        predictions: Predicted points of shape [batch_size, seq_len, dim]
        labels: Target labels of shape [batch_size, seq_len]
        positions: Target positions of shape [batch_size, n_positions, dim]
        reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        
    Returns:
        MSE loss between predictions and their corresponding true positions,
        multiplied by the feature dimension
    """
    # For backward compatibility, always reuse mse_loss_with_labels for the base computation
    base_loss = mse_loss_with_labels(predictions, positions, labels)
    
    # Handle reduction for the unified interface
    if reduction == 'none':
        # Create per-batch losses
        batch_size = predictions.shape[0]
        batch_losses = torch.full((batch_size,), base_loss.item(), device=predictions.device)
        return batch_losses
    elif reduction == 'mean':
        return base_loss
    elif reduction == 'sum':
        return base_loss * predictions.shape[0]
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class MSELoss(torch.nn.Module):
    """
    MSE loss module for GMM models.
    
    This module computes the MSE loss between predicted points and their
    true positions based on the provided labels.
    """
    
    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        """
        Initialize the MSE loss.
        
        Args:
            reduction: How to reduce the batch dimension ('none', 'mean', 'sum')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,  # shape [batch_size, seq_len, dim]
        labels: torch.Tensor,      # shape [batch_size, seq_len]
        positions: torch.Tensor    # shape [batch_size, n_positions, dim]
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted points and their true positions.
        
        Args:
            predictions: Predicted points of shape [batch_size, seq_len, dim]
            labels: Target labels of shape [batch_size, seq_len]
            positions: Target positions of shape [batch_size, n_positions, dim]
            
        Returns:
            MSE loss between predictions and their corresponding true positions
        """
        return mse_loss(
            predictions,
            labels,
            positions,
            reduction=self.reduction
        )