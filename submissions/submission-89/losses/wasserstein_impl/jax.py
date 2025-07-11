"""
Core implementation of Wasserstein distance calculation using OTT-JAX with conversion to PyTorch.

JAX-based Wasserstein distance implementation with JIT compilation.
Offers two algorithms:
1. Sinkhorn: Entropy-regularized optimal transport (fast, differentiable)
2. Hungarian: Optimal assignment for equal-sized point clouds (exact)
"""

# Standard library imports
import os
import logging
from typing import Optional, Literal, Tuple, Callable, Dict, Any

# Third-party imports
import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack
import torch.nn as nn

# Disable XLA preallocating memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Local imports
from ..utils import validate_inputs

logger = logging.getLogger(__name__)

# Initialize JAX availability flags
_has_jax = False
_has_jax_gpu = False
_jax_devices = []

# Initialize function cache
_sinkhorn_function_cache = {}

# Conditionally import JAX and OTT
try:
    # Import JAX and OTT
    import jax
    import jax.numpy as jnp
    from jax.dlpack import from_dlpack, to_dlpack
    from ott.geometry import pointcloud
    from ott.solvers.linear import sinkhorn
    from ott.problems.linear import linear_problem
    
    # Hungarian is in a different location
    from ott.tools.unreg import hungarian
    
    # Log OTT info
    logger.info("Using OTT-JAX 0.5+ for optimal transport algorithms")

    # Check for available devices
    _jax_devices = jax.devices()
    _has_jax_gpu = any(d.platform == 'gpu' for d in _jax_devices)
    _has_jax = True

    # Log info about detected JAX setup
    if _has_jax_gpu:
        gpu_count = len([d for d in _jax_devices if d.platform == 'gpu'])
        logger.info(f"JAX found {gpu_count} GPU devices")
    else:
        logger.info("JAX GPU acceleration not available, using CPU")

except ImportError:
    _has_jax = False
    _has_jax_gpu = False
    logger.warning("JAX or OTT not available. To use JAX implementation, install with: pip install jax jaxlib ott-jax>=0.5.0")


# Define the core computational functions outside the try block
def _compute_hungarian_single(x, y):
    """Hungarian algorithm for a single example.
    
    This function computes the optimal matching between two point clouds using the
    Hungarian algorithm, and returns a differentiable cost (MSE between matched points).
    
    Args:
        x: First set of points [n_points, dim]
        y: Second set of points [n_points, dim]
        
    Returns:
        Differentiable MSE between matched points (scalar)
    """
    if not _has_jax:
        raise ImportError("JAX is not available")
    
    # Create PointCloud geometry
    geom = pointcloud.PointCloud(x=x, y=y)
    
    # Use Hungarian algorithm - returns (cost, paired_indices)
    # The cost is differentiable w.r.t. x and y and represents the mean squared error
    _, output = hungarian(geom)
    
    # Extract source and target indices
    src_indices, tgt_indices = output.paired_indices
    
    # Compute squared Euclidean distances between matched points
    # This is differentiable w.r.t. x and y
    squared_dists = jnp.sum((x[src_indices] - y[tgt_indices]) ** 2, axis=1)
    
    # Return mean squared error (differentiable)
    return jnp.mean(squared_dists)


# JIT-compiled single function for Hungarian algorithm
_compute_hungarian_single = jax.jit(_compute_hungarian_single)

# Batch version of the Hungarian algorithm
_compute_hungarian_batch = jax.jit(jax.vmap(_compute_hungarian_single))

def create_sinkhorn_functions(epsilon, max_iterations):
    """Create and return batched Sinkhorn functions with specific parameters.
    
    Args:
        epsilon: Regularization parameter
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (batched_forward_function, batched_gradient_function)
    """
    if not _has_jax:
        raise ImportError("JAX is not available")
    
    # Sinkhorn algorithm for a single example
    def _compute_sinkhorn_single(x, y, x_w, y_w):
        # Create PointCloud geometry with epsilon parameter
        geom = pointcloud.PointCloud(x=x, y=y, epsilon=epsilon)
        
        # Create linear problem with weights
        problem = linear_problem.LinearProblem(geom, a=x_w, b=y_w)
        
        # Use Sinkhorn algorithm
        solver = sinkhorn.Sinkhorn(
            threshold=1e-3,
            max_iterations=max_iterations,
            lse_mode=True  # More stable log-sum-exp implementation
        )
        
        solution = solver(problem)
        
        # Return regularized OT cost
        return solution.reg_ot_cost
    
    # Batch and JIT-compile Sinkhorn function
    sinkhorn_batch_jit = jax.jit(jax.vmap(_compute_sinkhorn_single))
    
    # Create gradient function for the whole batch with gradients for all inputs
    def grad_batch_fn(x_batch, y_batch, x_w_batch, y_w_batch):
        # Compute gradients with respect to all inputs (points and weights)
        grad_fn = jax.grad(lambda x, y, x_w, y_w: jnp.sum(sinkhorn_batch_jit(
            x, y, x_w, y_w
        )), argnums=(0, 1, 2, 3))
        
        return grad_fn(x_batch, y_batch, x_w_batch, y_w_batch)
    
    # Return both functions
    return sinkhorn_batch_jit, grad_batch_fn

def get_or_create_sinkhorn_functions(epsilon, max_iterations):
    """Get or create Sinkhorn functions for the given parameters from the cache.
    
    Args:
        epsilon: Regularization parameter
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (batched_forward_function, batched_gradient_function)
    """
    if not _has_jax:
        raise ImportError("JAX is not available")
        
    # Use parameter tuple as cache key
    param_key = (float(epsilon), int(max_iterations))
    
    if param_key not in _sinkhorn_function_cache:
        # Create and cache the functions
        sinkhorn_batch_jit, grad_batch_fn = create_sinkhorn_functions(epsilon, max_iterations)
        _sinkhorn_function_cache[param_key] = (sinkhorn_batch_jit, grad_batch_fn)
    
    # Get the cached functions
    return _sinkhorn_function_cache[param_key]


# Define Function classes for autograd
class _RegularizedWassersteinJAXFunction(torch.autograd.Function):
    """Internal PyTorch autograd function for JAX-based regularized Wasserstein distance."""
    
    @staticmethod
    def forward(ctx, x, y, x_weights, y_weights, epsilon, max_iterations):
        """Forward pass for regularized Wasserstein distance."""
        # Save epsilon and max_iterations for creating Sinkhorn function
        ctx.epsilon = epsilon
        ctx.max_iterations = max_iterations
        
        # Ensure tensors are contiguous and float32 (JAX has limited bfloat16 support via DLPack)
        # Then convert to JAX arrays using DLPack protocol
        x_contiguous = x.detach().to(torch.float32).contiguous()
        y_contiguous = y.detach().to(torch.float32).contiguous()
        x_weights_contiguous = x_weights.detach().to(torch.float32).contiguous()
        y_weights_contiguous = y_weights.detach().to(torch.float32).contiguous()
        
        x_jax = from_dlpack(x_contiguous)
        y_jax = from_dlpack(y_contiguous)
        x_weights_jax = from_dlpack(x_weights_contiguous)
        y_weights_jax = from_dlpack(y_weights_contiguous)
        
        # Get or create the Sinkhorn function with these parameters
        sinkhorn_batch_jit, _ = get_or_create_sinkhorn_functions(epsilon, max_iterations)
        
        # Compute the distances and VJP function directly
        distances_jax, vjp_fn = jax.vjp(
            sinkhorn_batch_jit, x_jax, y_jax, x_weights_jax, y_weights_jax
        )
        
        # Save VJP function for backward pass
        ctx.vjp_fn = vjp_fn
        
        # Direct conversion from JAX array to PyTorch tensor
        distances = torch_dlpack.from_dlpack(distances_jax)
        
        return distances
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients using JAX VJP."""
        # Get saved VJP function
        vjp_fn = ctx.vjp_fn
        
        # Make gradient contiguous and float32 to avoid issues with DLPack
        grad_output_contiguous = grad_output.detach().to(torch.float32).contiguous()
        
        grad_output_jax = from_dlpack(grad_output_contiguous)
        
        # Compute gradients using the VJP function
        # vjp_fn takes the output gradient and returns gradients for all inputs
        grad_x_jax, grad_y_jax, grad_x_weights_jax, grad_y_weights_jax = vjp_fn(grad_output_jax)
        
        # Direct conversion from JAX arrays to PyTorch tensors
        grad_x = torch_dlpack.from_dlpack(grad_x_jax)
        grad_y = torch_dlpack.from_dlpack(grad_y_jax)
        grad_x_weights = torch_dlpack.from_dlpack(grad_x_weights_jax)
        grad_y_weights = torch_dlpack.from_dlpack(grad_y_weights_jax)
        
        # Return gradients (None for parameters we don't backpropagate through)
        return grad_x, grad_y, grad_x_weights, grad_y_weights, None, None


class _ExactWassersteinJAXFunction(torch.autograd.Function):
    """Internal PyTorch autograd function for JAX-based exact Wasserstein distance."""
    
    @staticmethod
    def forward(ctx, x, y, x_weights, y_weights):
        """
        Forward pass for differentiable Wasserstein distance based on Hungarian matching.
        
        Args:
            x: First batch of points [batch_size, n_points, dim]
            y: Second batch of points [batch_size, n_points, dim]
            x_weights: Weights for x points [batch_size, n_points] (validated but not used)
            y_weights: Weights for y points [batch_size, n_points] (validated but not used)
            
        Returns:
            Differentiable MSE distances based on the Hungarian matching [batch_size]
        """
        # Ensure tensors are contiguous and float32 (for JAX compatibility)
        x_contiguous = x.detach().to(torch.float32).contiguous()
        y_contiguous = y.detach().to(torch.float32).contiguous()
        
        x_jax = from_dlpack(x_contiguous)
        y_jax = from_dlpack(y_contiguous)
        
        # Compute the distances and VJP function directly with _compute_hungarian_batch
        distances_jax, vjp_fn = jax.vjp(_compute_hungarian_batch, x_jax, y_jax)
        
        # Save VJP function for backward pass
        ctx.vjp_fn = vjp_fn
        
        # Direct conversion from JAX array to PyTorch tensor
        distances = torch_dlpack.from_dlpack(distances_jax)
        
        return distances
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computing gradients for x and y using JAX VJP.
        """
        # Get saved VJP function
        vjp_fn = ctx.vjp_fn
        
        # Make gradient contiguous and float32 to avoid issues with DLPack
        grad_output_contiguous = grad_output.detach().to(torch.float32).contiguous()
        
        # Try direct conversion to JAX array using DLPack protocol
        grad_output_jax = from_dlpack(grad_output_contiguous)
        
        # Compute gradients using the VJP function
        # vjp_fn takes the output gradient and returns gradients for all inputs
        grad_x_jax, grad_y_jax = vjp_fn(grad_output_jax)
        
        # Direct conversion from JAX arrays to PyTorch tensors
        grad_x = torch_dlpack.from_dlpack(grad_x_jax)
        grad_y = torch_dlpack.from_dlpack(grad_y_jax)
        
        # Return gradients (None for weights parameters)
        return grad_x, grad_y, None, None


# Import base classes
from .base import RegularizedWassersteinBase, ExactWassersteinBase
from .base import register_exact_implementation, register_regularized_implementation

class JaxRegularizedWasserstein(RegularizedWassersteinBase):
    """
    Regularized Wasserstein distance using the Sinkhorn algorithm with JAX backend.
    
    This class computes the regularized Wasserstein distance between point clouds
    using the Sinkhorn algorithm implemented in JAX for efficiency. It uses JAX's 
    vector-Jacobian product (VJP) mechanism for efficient and accurate gradient 
    computation during backpropagation.
    
    Args:
        epsilon: Regularization parameter (default: 0.01)
        max_iterations: Maximum number of Sinkhorn iterations (default: 100)
    """
    
    def __init__(self, epsilon: float = 0.01, max_iterations: int = 100):
        """Initialize with algorithm parameters."""
        super().__init__(epsilon=epsilon, max_iterations=max_iterations)
        
        if not _has_jax:
            raise ImportError("JAX is not available. Install with: pip install jax jaxlib ott-jax>=0.5.0")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        x_weights: Optional[torch.Tensor] = None,
        y_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regularized Wasserstein distance.
        
        Args:
            x: First batch of points [batch_size, n_points_x, dim]
            y: Second batch of points [batch_size, n_points_y, dim]
            x_weights: Weights for x points [batch_size, n_points_x] (optional)
            y_weights: Weights for y points [batch_size, n_points_y] (optional)
        
        Returns:
            Regularized Wasserstein distances [batch_size]
        """
        # Validate inputs
        x, y, x_weights, y_weights = validate_inputs(
            x, y, x_weights, y_weights,
            require_uniform_weights_and_equal_points=False,
            implementation_name="JAX Sinkhorn implementation"
        )
        
        # Use the Function class for forward/backward
        return _RegularizedWassersteinJAXFunction.apply(
            x, y, x_weights, y_weights, self.epsilon, self.max_iterations
        )


class JaxExactWasserstein(ExactWassersteinBase):
    """
    Differentiable Wasserstein distance using the Hungarian algorithm with JAX backend.
    
    This class computes a differentiable version of the Wasserstein distance between point clouds
    using the Hungarian algorithm implemented in JAX for efficiency. It uses JAX's vector-Jacobian
    product (VJP) mechanism to enable gradient flow through the point coordinates while still
    benefiting from the exact optimal transport solution.
    
    The implementation:
    1. Computes the optimal matching using the Hungarian algorithm
    2. Returns a differentiable mean squared error between matched points
    3. Uses JAX's VJP to efficiently compute gradients during backpropagation
    
    Requires uniform weights and equal point counts.
    """
    
    def __init__(self):
        """Initialize the differentiable Wasserstein distance module."""
        super().__init__()
        
        if not _has_jax:
            raise ImportError("JAX is not available. Install with: pip install jax jaxlib ott-jax>=0.5.0")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        x_weights: Optional[torch.Tensor] = None,
        y_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute differentiable Wasserstein distance.
        
        Args:
            x: First batch of points [batch_size, n_points_x, dim]
            y: Second batch of points [batch_size, n_points_y, dim]
            x_weights: Weights for x points [batch_size, n_points_x] (optional)
            y_weights: Weights for y points [batch_size, n_points_y] (optional)
        
        Returns:
            Differentiable MSE distances based on the Hungarian matching [batch_size]
        """
        # Validate inputs - Hungarian requires uniform weights and equal point counts
        x, y, x_weights, y_weights = validate_inputs(
            x, y, x_weights, y_weights,
            require_uniform_weights_and_equal_points=True,
            implementation_name="JAX Hungarian algorithm"
        )
        
        # Use the Function class for forward
        return _ExactWassersteinJAXFunction.apply(
            x, y, x_weights, y_weights
        )


def compute_wasserstein_jax(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: Optional[torch.Tensor] = None,
    y_weights: Optional[torch.Tensor] = None,
    algorithm: Literal["sinkhorn", "hungarian", "hungarian_diff"] = "sinkhorn",
    epsilon: float = 0.01,
    max_iterations: int = 100,
    **kwargs
) -> torch.Tensor:
    """
    Compute Wasserstein distance between point clouds using JAX.
    
    This function provides backward compatibility with the old API.
    
    Args:
        x: First set of points [batch_size, n_points_x, dim]
        y: Second set of points [batch_size, n_points_y, dim]
        x_weights: Weights for x points [batch_size, n_points_x]
        y_weights: Weights for y points [batch_size, n_points_y]
        algorithm: Algorithm to use:
            - "sinkhorn": Entropy-regularized OT (differentiable)
            - "hungarian", "hungarian_diff": Exact OT with differentiable MSE loss
        epsilon: Regularization parameter for Sinkhorn algorithm
        max_iterations: Maximum number of iterations for Sinkhorn algorithm
        
    Returns:
        Wasserstein distances for the batch [batch_size,]
    """
    if not _has_jax:
        raise ImportError("JAX or OTT not available. Install with: pip install jax jaxlib ott-jax>=0.5.0")
    
    # Use the appropriate algorithm
    if algorithm == "sinkhorn":
        model = JaxRegularizedWasserstein(epsilon=epsilon, max_iterations=max_iterations)
        return model(x, y, x_weights, y_weights)
    else:  # algorithm in ["hungarian", "hungarian_diff"]
        model = JaxExactWasserstein()
        return model(x, y, x_weights, y_weights)


# Register the implementations if JAX is available
if _has_jax:
    register_exact_implementation("jax", JaxExactWasserstein)
    register_regularized_implementation("jax", JaxRegularizedWasserstein) 