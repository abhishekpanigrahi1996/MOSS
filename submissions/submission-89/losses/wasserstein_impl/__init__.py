"""
Wasserstein distance implementations for PyTorch.

This package provides various implementations of Wasserstein distance
calculations that can be used within PyTorch models.
"""

# Import base classes and registry functions
from .base import (
    WassersteinBase,
    ExactWassersteinBase,
    RegularizedWassersteinBase,
    get_exact_implementation,
    get_regularized_implementation,
    list_available_implementations
)

# Import backend availability flags
from .backends import _has_jax, _has_pot, _has_scipy, get_available_backends

# Import implementations (these will register themselves)
from . import pot
from . import scipy

# Import JAX implementation if available
if _has_jax:
    from . import jax

# Expose the main classes and functions
__all__ = [
    "WassersteinBase",
    "ExactWassersteinBase",
    "RegularizedWassersteinBase",
    "get_exact_implementation",
    "get_regularized_implementation",
    "list_available_implementations",
    "get_available_backends",
    "_has_jax",
    "_has_pot",
    "_has_scipy"
]