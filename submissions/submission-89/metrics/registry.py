"""
Registry for loss functions and metrics.

This module provides a simple registry for loss functions and metrics,
allowing for easy registration and retrieval of factory functions.
"""

import logging
from typing import Dict, Any, Callable, Optional

# Import loss factory functions from losses module
from losses import create_mse_loss, create_energy_loss, create_wasserstein_loss

logger = logging.getLogger(__name__)

# Registry of loss factory functions
_LOSS_FACTORIES: Dict[str, Callable] = {}

def register_loss(loss_type: str, factory_fn: Callable) -> None:
    """
    Register a factory function for a loss type.
    
    Args:
        loss_type: String identifier for the loss type
        factory_fn: Factory function that creates the loss function
        
    Example:
        >>> from losses import create_js_divergence_loss
        >>> register_loss("js_divergence", create_js_divergence_loss)
    """
    global _LOSS_FACTORIES
    loss_type = loss_type.lower()
    
    if loss_type in _LOSS_FACTORIES:
        logger.warning(f"Overriding existing factory for loss type: {loss_type}")
        
    _LOSS_FACTORIES[loss_type] = factory_fn
    logger.debug(f"Registered factory for loss type: {loss_type}")


def get_loss_factory(loss_type: str) -> Optional[Callable]:
    """
    Get the factory function for a loss type.
    
    Args:
        loss_type: String identifier for the loss type
        
    Returns:
        Factory function or None if not found
    """
    global _LOSS_FACTORIES
    loss_type = loss_type.lower()
    
    # Try to load factory functions if registry is empty
    if not _LOSS_FACTORIES:
        _load_default_factories()
        
    return _LOSS_FACTORIES.get(loss_type)


def _load_default_factories() -> None:
    """
    Load the default factory functions.
    """
    global _LOSS_FACTORIES
    
    # Use the real loss factory functions from the losses module
    _LOSS_FACTORIES.update({
        "mse": create_mse_loss,
        "energy": create_energy_loss,
        "wasserstein": create_wasserstein_loss
    })
    
    logger.debug("Loaded loss factory functions from losses module")


def get_available_loss_types() -> list:
    """
    Get a list of all available loss types.
    
    Returns:
        List of registered loss type strings
    """
    global _LOSS_FACTORIES
    
    # Load factories if not already loaded
    if not _LOSS_FACTORIES:
        _load_default_factories()
        
    return list(_LOSS_FACTORIES.keys())