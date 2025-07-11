"""
Base classes for Wasserstein distance implementations.

This module defines the base classes that all Wasserstein distance
implementations should inherit from. These provide a consistent
interface regardless of the backend being used.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Type, Any


class WassersteinBase(nn.Module):
    """Base class for all Wasserstein distance implementations."""
    
    def __init__(self):
        """Initialize the Wasserstein distance base class."""
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor] = None,
        y_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Wasserstein distance between two batches of point clouds.
        
        Args:
            x: First batch of points with shape [batch_size, n_points_x, dim]
            y: Second batch of points with shape [batch_size, n_points_y, dim]
            x_weights: Optional weights for x points [batch_size, n_points_x]
            y_weights: Optional weights for y points [batch_size, n_points_y]
            
        Returns:
            Wasserstein distances for each batch [batch_size]
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ExactWassersteinBase(WassersteinBase):
    """Base class for exact (unregularized) Wasserstein distance implementations."""
    
    def __init__(self):
        """Initialize the exact Wasserstein distance base class."""
        super().__init__()


class RegularizedWassersteinBase(WassersteinBase):
    """Base class for regularized Wasserstein distance implementations."""
    
    def __init__(self, epsilon: float = 0.1, max_iterations: int = 1000):
        """
        Initialize the regularized Wasserstein distance base class.
        
        Args:
            epsilon: Regularization parameter
            max_iterations: Maximum number of iterations for the algorithm
        """
        super().__init__()
        self.epsilon = epsilon
        self.max_iterations = max_iterations


# Registry of implementations
_exact_implementations: Dict[str, Type[ExactWassersteinBase]] = {}
_regularized_implementations: Dict[str, Type[RegularizedWassersteinBase]] = {}


def register_exact_implementation(name: str, implementation: Type[ExactWassersteinBase]):
    """
    Register an exact Wasserstein implementation.
    
    Args:
        name: Identifier for the implementation
        implementation: Implementation class to register
    """
    _exact_implementations[name] = implementation


def register_regularized_implementation(name: str, implementation: Type[RegularizedWassersteinBase]):
    """
    Register a regularized Wasserstein implementation.
    
    Args:
        name: Identifier for the implementation
        implementation: Implementation class to register
    """
    _regularized_implementations[name] = implementation


def get_exact_implementation(name: str, **kwargs: Any) -> ExactWassersteinBase:
    """
    Get an exact Wasserstein implementation by name.
    
    Args:
        name: Name of the implementation
        **kwargs: Arguments to pass to the implementation constructor
        
    Returns:
        Instantiated implementation
        
    Raises:
        ValueError: If implementation is not found
    """
    if name not in _exact_implementations:
        available = list(_exact_implementations.keys())
        raise ValueError(f"Implementation '{name}' not found. Available: {available}")
    
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Creating exact Wasserstein implementation: {name}")
    return _exact_implementations[name](**kwargs)


def get_regularized_implementation(
    name: str, 
    epsilon: float = 0.1, 
    max_iterations: int = 1000,
    **kwargs: Any
) -> RegularizedWassersteinBase:
    """
    Get a regularized Wasserstein implementation by name.
    
    Args:
        name: Name of the implementation
        epsilon: Regularization parameter
        max_iterations: Maximum number of iterations
        **kwargs: Additional arguments to pass to the implementation constructor
        
    Returns:
        Instantiated implementation
        
    Raises:
        ValueError: If implementation is not found
    """
    if name not in _regularized_implementations:
        available = list(_regularized_implementations.keys())
        raise ValueError(f"Implementation '{name}' not found. Available: {available}")
    
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Creating regularized Wasserstein implementation: {name}, epsilon={epsilon}, max_iterations={max_iterations}")
    return _regularized_implementations[name](
        epsilon=epsilon, 
        max_iterations=max_iterations,
        **kwargs
    )


def list_available_implementations():
    """
    List available implementations.
    
    Returns:
        Dictionary with available exact and regularized implementations
    """
    return {
        "exact": list(_exact_implementations.keys()),
        "regularized": list(_regularized_implementations.keys())
    }