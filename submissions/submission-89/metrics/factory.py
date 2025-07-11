"""
Factory functions for creating loss functions and metrics.

This module provides factory functions for creating and configuring
loss and metric functions from string identifiers or configuration dictionaries.
It uses the configuration parser to standardize different configuration formats.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable, List

from .configuration import parse_loss_config
from .registry import get_loss_factory

logger = logging.getLogger(__name__)

def create_loss_from_config(
    loss_config: Union[str, Dict[str, Any]]
) -> Optional[Callable]:
    """
    Create a loss function from a string or dictionary configuration.
    
    This function provides a unified interface to create loss functions
    from either simple string identifiers ('mse', 'energy', 'wasserstein'),
    extended string formats ('wasserstein_exact_jax'), or detailed 
    configuration dictionaries.
    
    Args:
        loss_config: Either a string identifying the loss type or a dictionary with
                     detailed configuration including:
                     - 'type': Loss type ('mse', 'energy', 'wasserstein')
                     - Algorithm-specific parameters
    
    Returns:
        A callable loss function that takes (predictions, targets) parameters,
        or None if the loss cannot be created
        
    Example:
        >>> # Simple string configuration
        >>> mse_loss = create_loss_from_config('mse')
        >>> 
        >>> # Extended string configuration
        >>> wasserstein_loss = create_loss_from_config('wasserstein_exact_jax')
        >>> 
        >>> # Dictionary configuration for Wasserstein loss
        >>> wasserstein_loss = create_loss_from_config({
        ...     'type': 'wasserstein',
        ...     'algorithm': 'sinkhorn',
        ...     'backend': 'jax',
        ...     'epsilon': 0.01,
        ...     'max_iterations': 100
        ... })
    """
    # Parse and standardize the configuration
    try:
        parsed_config = parse_loss_config(loss_config)
    except ValueError as e:
        logger.error(f"Failed to parse loss configuration: {e}")
        return None
    
    # Get the base loss type
    loss_type = parsed_config.get('type')
    if not loss_type:
        logger.error("Loss configuration missing 'type' field after parsing")
        return None
    
    # Get the factory function for this loss type
    factory_fn = get_loss_factory(loss_type)
    if factory_fn is None:
        logger.error(f"No factory function registered for loss type: {loss_type}")
        return None
    
    # Remove the type key before passing to factory
    kwargs = {k: v for k, v in parsed_config.items() if k != 'type'}
    
    # Create and return the loss function
    try:
        return factory_fn(**kwargs)
    except Exception as e:
        logger.error(f"Error creating loss function of type {loss_type}: {e}")
        return None


def create_metric_functions(
    metric_configs: Optional[List[Union[str, Dict[str, Any]]]] = None
) -> Dict[str, Callable]:
    """
    Create a dictionary of metric functions from a list of metric configurations.
    
    This function generates callable metric functions for each requested metric
    using the same factory system as the loss functions.
    
    Args:
        metric_configs: List of metric configurations, where each configuration
                       can be either a string or a dictionary. If None, uses defaults.
                      
    Returns:
        Dictionary mapping metric names to their corresponding callable functions.
        
    Example:
        >>> metrics = create_metric_functions([
        ...     'mse',
        ...     'wasserstein_exact_jax',
        ...     {
        ...         'type': 'wasserstein',
        ...         'algorithm': 'sinkhorn',
        ...         'backend': 'jax'
        ...     }
        ... ])
        >>> result = metrics['wasserstein_jax'](predictions, targets)
    """
    # Default metrics if none provided
    if metric_configs is None:
        metric_configs = ['mse']
    
    metric_functions = {}
    
    # Process each metric configuration
    for config in metric_configs:
        # Parse the configuration
        try:
            parsed_config = parse_loss_config(config)
            base_type = parsed_config.get('type')
            
            # Create the metric function
            metric_fn = create_loss_from_config(parsed_config)
            
            if metric_fn is not None:
                # Generate a descriptive metric name
                metric_name = _generate_metric_name(parsed_config)
                
                # Add the metric function to the dictionary
                metric_functions[metric_name] = metric_fn
            
        except Exception as e:
            logger.warning(f"Failed to create metric function for config {config}: {e}")
    
    return metric_functions


def _generate_metric_name(config: Dict[str, Any]) -> str:
    """
    Generate a descriptive name for a metric based on its configuration.
    
    Args:
        config: Parsed configuration dictionary
        
    Returns:
        A string representing the metric name
    """
    base_type = config.get('type')
    if not base_type:
        return "unknown"
    
    # Start with the base type
    name_parts = [base_type]
    
    # Add algorithm if present for wasserstein
    if base_type == 'wasserstein' and 'algorithm' in config:
        name_parts.append(config['algorithm'])
        
    # Add backend if present for wasserstein
    if base_type == 'wasserstein' and 'backend' in config:
        name_parts.append(config['backend'])
    
    # Join with underscores
    return "_".join(name_parts)