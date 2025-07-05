"""
Backend availability detection for Wasserstein distance implementations.

This module checks which backend libraries are available in the current
environment and exposes flags that can be used to conditionally import
backend-specific code.
"""

import logging

logger = logging.getLogger(__name__)

# JAX availability
_has_jax = False
try:
    import jax
    import jax.numpy as jnp
    try:
        from ott.geometry import pointcloud
        from ott.solvers.linear import sinkhorn
        _has_jax = True
        logger.info("JAX and OTT are available")
    except ImportError:
        logger.warning("JAX is available but OTT is not")
except ImportError:
    logger.warning("JAX is not available")

# POT availability
_has_pot = False
try:
    import ot
    _has_pot = True
    logger.info("POT (Python Optimal Transport) is available")
except ImportError:
    logger.warning("POT is not available")

# SciPy availability
_has_scipy = False
try:
    import scipy
    import scipy.spatial.distance
    import scipy.optimize
    _has_scipy = True
    logger.info("SciPy is available")
except ImportError:
    logger.warning("SciPy is not available")


def get_available_backends():
    """
    Get a dictionary of available backend implementations.
    
    Returns:
        Dictionary with backend names as keys and availability as values
    """
    return {
        "jax": _has_jax,
        "pot": _has_pot,
        "scipy": _has_scipy
    }