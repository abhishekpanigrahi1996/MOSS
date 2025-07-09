import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional, Dict, Any


class MSELoss:
    """
    Mean Squared Error loss function with configurable reduction.
    
    A wrapper around PyTorch's MSE loss that can be used in training contexts
    where additional context information might be passed.
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initialize the MSE loss function.
        
        Args:
            reduction: Specifies the reduction to apply to the output.
                      Can be 'mean', 'sum', or 'none'.
        """
        self.reduction = reduction

    def __call__(
        self, 
        x: Tensor, 
        y: Tensor, 
        **ctx: Any
    ) -> Tensor:
        """
        Calculate the mean squared error loss between predictions and targets.
        
        Args:
            x: Predicted values (output from the model)
            y: Target values (ground truth)
            **ctx: Additional context information (unused in this implementation)
            
        Returns:
            The computed MSE loss as a tensor
        """
        return F.mse_loss(x, y, reduction=self.reduction)


class NormedMSELoss:
    """
    Mean Squared Error loss with L2 regularization on model weights.
    
    This loss function combines standard MSE loss with a regularization term
    that penalizes large model weights, helping to prevent overfitting.
    """
    
    def __init__(
        self, 
        beta: float = 0.5, 
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize the normed MSE loss function.
        
        Args:
            beta: Weight of the regularization term. Higher values lead to
                  stronger regularization.
            reduction: Specifies the reduction to apply to the MSE loss.
                      Can be 'mean', 'sum', or 'none'.
        """
        self.reduction = reduction
        self.beta = beta
    
    def __call__(
        self, 
        x: Tensor, 
        y: Tensor, 
        **ctx: Any
    ) -> Tensor:
        """
        Calculate the normed MSE loss (MSE + L2 regularization).
        
        Args:
            x: Predicted values (output from the model)
            y: Target values (ground truth)
            **ctx: Context dictionary that must contain a 'model' key with
                  a model that has a .norm() method
            
        Returns:
            The computed normed MSE loss as a tensor
        """
        # Standard MSE loss
        mse_loss = F.mse_loss(x, y, reduction=self.reduction)
        
        # Add L2 regularization term using model's norm method
        regularization_term = self.beta * ctx['model'].norm()
        
        return mse_loss + regularization_term
