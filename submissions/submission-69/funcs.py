import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from typing import Any, Tuple, Optional

from utils import filter_topk


class _StraightThrough(Function):
    """
    Base class for straight-through estimator functions.
    
    This implements the straight-through estimator pattern where the forward
    pass applies some non-differentiable operation, but the backward pass
    simply passes gradients through unchanged.
    """
    @staticmethod
    def setup_context(
        ctx: FunctionCtx, 
        inputs: Tuple[Tensor, float], 
        output: Tensor
    ) -> None:
        """
        Set up the context for the backward pass.
        
        Args:
            ctx: The context object to store information for backward pass
            inputs: The input tensors (weights, k)
            output: The output tensor from the forward pass
        """
        weights, k = inputs
        # Note: We don't save tensors for backward as we use straight-through
        # ctx.save_for_backward(weights, output)

    @staticmethod
    def backward(
        ctx: FunctionCtx, 
        grad_output: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Backward pass using straight-through estimation.
        
        Args:
            ctx: The context object containing saved information
            grad_output: Gradient of the loss with respect to the output
            
        Returns:
            Tuple of gradients with respect to inputs (weights, k).
            The gradient w.r.t. weights is passed through unchanged,
            and the gradient w.r.t. k is None (not differentiable).
        """
        # weights, output = ctx.saved_tensors
        # Straight-through estimation: pass gradient through unchanged
        return grad_output, None


class TopKMask(_StraightThrough):
    """
    Autograd function that applies top-k masking in forward pass.
    
    This function selects the top-k largest elements (by absolute value)
    and sets all other elements to zero in the forward pass, but uses
    straight-through estimation in the backward pass.
    """
    
    @staticmethod
    def forward(weights: Tensor, k: float) -> Tensor:
        """
        Forward pass: Apply top-k masking to weights.
        
        Args:
            weights: Input tensor to be masked
            k: Fraction of elements to keep (between 0 and 1)
            
        Returns:
            Binary mask tensor where 1 indicates kept elements and 0 indicates
            masked elements
        """
        return filter_topk(weights, k, return_mask=True)


def topk_mask(weights: Tensor, k: float) -> Tensor:
    """
    Convenience function to apply top-k masking with straight-through gradients.
    
    This function creates a binary mask that keeps only the top-k largest
    elements (by absolute value) and zeros out the rest. During backpropagation,
    gradients are passed through unchanged using straight-through estimation.
    
    Args:
        weights: Input tensor to be masked
        k: Fraction of elements to keep (between 0 and 1)
        
    Returns:
        Binary mask tensor where 1 indicates kept elements and 0 indicates
        masked elements
        
    Example:
        >>> weights = torch.tensor([[1.0, 0.1, 0.5], [0.2, 0.8, 0.3]])
        >>> mask = topk_mask(weights, 0.5)  # Keep top 50% of elements
        >>> masked_weights = weights * mask
    """
    return TopKMask.apply(weights, k)
