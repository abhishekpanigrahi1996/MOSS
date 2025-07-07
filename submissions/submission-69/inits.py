import torch
from torch import nn, Tensor
from typing import Any, Union, Tuple

from utils import ChainableFn


class RandomUniform(ChainableFn):
    """
    Initialize tensors with random values from a uniform distribution [0, 1).
    
    This initialization function generates random values uniformly distributed
    between 0 and 1, matching the shape of the input tensor.
    """

    name = 'RandomUniform'
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Generate random uniform values matching the input tensor shape.
        
        Args:
            arr: Input tensor whose shape will be matched
            **ctx: Additional context information (unused)
            
        Returns:
            New tensor with random uniform values in [0, 1)
        """
        return torch.rand_like(arr)


class RandomNormal(ChainableFn):
    """
    Initialize tensors with random values from a normal distribution.
    
    This initialization function generates random values from a normal
    (Gaussian) distribution with specified mean and standard deviation.
    """
    
    name = 'RandomNormal'
    
    def __init__(
        self, 
        prev: ChainableFn = None, 
        mean: float = 0., 
        std: float = 1.
    ) -> None:
        """
        Initialize the random normal generator.
        
        Args:
            prev: Previous function in the chain (if any)
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution
        """
        super().__init__(prev)
        self.mean = float(mean)
        self.std = float(std)
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Generate random normal values matching the input tensor shape.
        
        Args:
            arr: Input tensor whose shape will be matched
            **ctx: Additional context information (unused)
            
        Returns:
            New tensor with random normal values
        """
        return torch.normal(
            mean=self.mean,
            std=self.std,
            size=arr.shape,
            device=arr.device,
            dtype=arr.dtype
        )


class Zeros(ChainableFn):
    """
    Initialize tensors with all zeros.
    
    This initialization function creates a tensor filled with zeros,
    matching the shape and properties of the input tensor.
    """
    
    name = 'Zeros'
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Generate zero values matching the input tensor shape.
        
        Args:
            arr: Input tensor whose shape will be matched
            **ctx: Additional context information (unused)
            
        Returns:
            New tensor filled with zeros
        """
        return torch.zeros_like(
            arr,
            device=arr.device,
            dtype=arr.dtype
        )


class Ones(ChainableFn):
    """
    Initialize tensors with all ones.
    
    This initialization function creates a tensor filled with ones,
    matching the shape and properties of the input tensor.
    """
    
    name = 'Ones'
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Generate ones matching the input tensor shape.
        
        Args:
            arr: Input tensor whose shape will be matched
            **ctx: Additional context information (unused)
            
        Returns:
            New tensor filled with ones
        """
        return torch.ones_like(
            arr,
            device=arr.device,
            dtype=arr.dtype
        )


class Triu(ChainableFn):
    """
    Apply upper triangular masking to tensors.
    
    This function modifies a tensor by dampening the lower triangular part.
    It's used to encourage upper triangular structure in weight matrices,
    which may promote topologically-ordered connectivity patterns.
    """
    
    name = 'Triu'
    
    def __init__(
        self, 
        prev: ChainableFn = None, 
        diagonal: int = 0,
        damping: float = 0.9
    ) -> None:
        """
        Initialize the upper triangular masking function.
        
        Args:
            prev: Previous function in the chain (if any)
            diagonal: Which diagonal to use as the boundary. 0 means the main
                     diagonal, positive values shift upward, negative values
                     shift downward
            damping: Fraction of each lower triangular element to dampen.
        """
        super().__init__(prev)
        self.diag = diagonal
        self.damping = damping

    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply upper triangular masking by dampening lower triangular elements
        (for each element x in the lower triangular part, set it to x * (1-damping))
        
        Args:
            arr: Input tensor to be masked
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with dampened lower triangular elements
        """
        # Dampen lower triangular elements by `damping`
        return arr - arr.tril(self.diag - 1) * self.damping


class _Size(ChainableFn):
    """
    Create a tensor of zeros with specified dimensions.
    
    This is a special initialization function that creates tensors from
    size specifications rather than existing tensors.
    """
    
    name = 'Size'
    
    def fn(self, *size: int, **ctx: Any) -> Tensor:
        """
        Create a tensor of zeros with the specified size.
        
        Args:
            *size: Dimensions of the tensor to create
            **ctx: Additional context information (unused)
            
        Returns:
            New tensor of zeros with the specified dimensions
        """
        return torch.zeros(size)


# Global instances for convenient access
Size = _Size()  # For creating tensors from size specifications
Like = Zeros()  # For creating tensors matching existing tensor shapes
Like.name = 'Like'  # Rename for clarity
