"""
Configuration parsing module for loss and metric functions.

This module handles different formats of loss/metric configurations,
providing a unified interface that standardizes all formats into a
consistent dictionary structure.
"""

import logging
import re
from typing import Dict, Any, Union, Tuple, Optional, cast

# Define a type alias for loss configuration dictionaries
ConfigValue = Union[str, float, int]
LossConfigDict = Dict[str, ConfigValue]

logger = logging.getLogger(__name__)

def parse_loss_config(config: Union[str, Dict[str, Any]]) -> LossConfigDict:
    """
    Parse any loss configuration format into a standardized dictionary.
    
    This function handles multiple configuration formats:
    1. Simple strings: "mse", "wasserstein"
    2. Extended strings: "wasserstein_exact_jax"
    3. Dictionary formats (both nested and flat)
    4. Legacy parameter formats
    
    Args:
        config: Loss/metric configuration in any supported format
        
    Returns:
        Standardized configuration dictionary with consistent keys
        
    Example:
        >>> parse_loss_config("mse")
        {'type': 'mse'}
        
        >>> parse_loss_config("wasserstein_exact_jax")
        {'type': 'wasserstein', 'algorithm': 'exact', 'backend': 'jax'}
        
        >>> parse_loss_config({
        ...     'type': 'wasserstein',
        ...     'algorithm': 'sinkhorn',
        ...     'backend': 'jax'
        ... })
        {'type': 'wasserstein', 'algorithm': 'sinkhorn', 'backend': 'jax'}
    """
    # Handle string format
    if isinstance(config, str):
        return parse_string_loss_config(config)
    
    # Handle dictionary format
    elif isinstance(config, dict):
        return parse_dict_loss_config(config)
    
    # Handle unsupported formats
    else:
        raise ValueError(f"Unsupported configuration type: {type(config)}")


def parse_string_loss_config(config_str: str) -> LossConfigDict:
    """
    Parse string format loss configuration into a standardized dictionary.
    
    Handles both simple formats ("mse") and extended formats
    ("wasserstein_exact_jax").
    
    Args:
        config_str: String configuration to parse
        
    Returns:
        Standardized configuration dictionary
    """
    config_str = config_str.lower().strip()
    
    # Handle simple string format (e.g., "mse")
    if config_str in ["mse", "energy"]:
        return {"type": config_str}
    
    # Handle extended string format (e.g., "wasserstein_exact_jax")
    parts = config_str.split("_")
    base_type = parts[0]
    
    # Create initial config with base type
    result: LossConfigDict = {"type": base_type}
    
    # Extract additional parameters
    if base_type == "wasserstein" and len(parts) > 1:
        # Process each part after the base type
        for i, part in enumerate(parts[1:], 1):
            # Try to categorize the part
            if part in ["exact", "sinkhorn"]:
                result["algorithm"] = part
            elif part in ["jax", "pot", "scipy"]:
                result["backend"] = part
            # Detect numeric parameter (e.g., "wasserstein_0.01")
            elif re.match(r"^(\d+(\.\d*)?|\.\d+)$", part):
                if "epsilon" not in result:
                    result["epsilon"] = float(part)
            # Detect max iterations (e.g., "wasserstein_iter100")
            elif part.startswith("iter") and part[4:].isdigit():
                result["max_iterations"] = int(part[4:])
            else:
                logger.warning(f"Unknown parameter in extended string format: {part}")
    
    return result


def parse_dict_loss_config(config_dict: Dict[str, Any]) -> LossConfigDict:
    """
    Parse dictionary format loss configuration into a standardized dictionary.
    
    Handles both flat and nested dictionary formats, as well as legacy formats.
    
    Args:
        config_dict: Dictionary configuration to parse
        
    Returns:
        Standardized configuration dictionary
    """
    # Deep copy to avoid modifying the original
    result: LossConfigDict = {}
    
    # Extract the base type
    if "type" in config_dict:
        result["type"] = config_dict["type"].lower()
    elif "loss_type" in config_dict and isinstance(config_dict["loss_type"], str):
        result["type"] = config_dict["loss_type"].lower()
    else:
        raise ValueError("Dictionary configuration must contain a 'type' or 'loss_type' key")
    
    # Get the base type
    base_type = result["type"]
    
    # Handle wasserstein-specific parameters
    if base_type == "wasserstein":
        # Handle flat format parameters
        for param, dest in [
            ("algorithm", "algorithm"),
            ("backend", "backend"),
            ("epsilon", "epsilon"),
            ("max_iterations", "max_iterations")
        ]:
            if param in config_dict:
                result[dest] = config_dict[param]
        
        # Handle legacy format parameters (wasserstein_*)
        for legacy_param, dest in [
            ("wasserstein_algorithm", "algorithm"),
            ("wasserstein_backend", "backend"),
            ("wasserstein_epsilon", "epsilon"),
            ("wasserstein_max_iter", "max_iterations")
        ]:
            if legacy_param in config_dict and dest not in result:
                result[dest] = config_dict[legacy_param]
    
    return result


def extract_wasserstein_params(config: Union[str, Dict[str, Any]]) -> LossConfigDict:
    """
    Extract Wasserstein-specific parameters from any config format.
    
    This utility function focuses on extracting parameters specifically
    relevant to Wasserstein distance calculation.
    
    Args:
        config: Any loss/metric configuration
        
    Returns:
        Dictionary with Wasserstein-specific parameters
    """
    # Parse the configuration first
    parsed = parse_loss_config(config)
    
    # Check if it's a Wasserstein configuration
    if parsed.get("type") != "wasserstein":
        return {}
    
    # Extract Wasserstein-specific parameters
    result: LossConfigDict = {}
    for param in ["algorithm", "backend", "epsilon", "max_iterations"]:
        if param in parsed:
            result[param] = parsed[param]
    
    return result