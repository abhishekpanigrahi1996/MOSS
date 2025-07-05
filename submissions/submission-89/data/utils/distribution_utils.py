"""
Utilities for handling distribution sampling in a standardized way.

This module provides classes and functions to work with various probability 
distributions in a consistent manner, supporting both fixed values and
configurable distributions.
"""

import os
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from scipy import special


class DistributionType(str, Enum):
    """
    Enumeration of supported distribution types.
    
    Attributes
    ----------
    FIXED : str
        Fixed value distribution (constant)
    UNIFORM : str
        Uniform distribution
    NORMAL : str
        Normal (Gaussian) distribution
    TRUNCATED_NORMAL : str
        Truncated normal distribution
    TRUNCATED_LOGNORMAL: str
        Truncated lognormal distribution
    POISSON : str
        Poisson distribution
    TRUNCATED_POISSON : str
        Truncated Poisson distribution
    CHOICE : str
        Discrete choice from a list of options
    RANGE : str
        Alias for uniform with integer values
    """
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    TRUNCATED_LOGNORMAL = "truncated_lognormal"
    POISSON = "poisson"
    TRUNCATED_POISSON = "truncated_poisson"
    CHOICE = "choice"
    RANGE = "range"  # Alias for uniform with integer values


# Cache for loaded presets
_parameter_presets = None


def load_parameter_presets() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load parameter presets from JSON configuration file.
    
    Returns:
        Dictionary containing parameter presets for cluster_params,
        snr_db_params, and sample_count_distribution.
        
    Raises:
        FileNotFoundError: If the parameter_presets.json file cannot be found
        json.JSONDecodeError: If the file has invalid JSON
    """
    global _parameter_presets
    
    if _parameter_presets is not None:
        return _parameter_presets
    
    # Path to parameter presets file - fixed relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    presets_path = os.path.join(project_root, 'config', 'json_defaults', 'data', 'parameter_presets.json')
    
    if not os.path.exists(presets_path):
        raise FileNotFoundError(f"Parameter presets file not found at: {presets_path}")
    
    with open(presets_path, 'r') as f:
        _parameter_presets = json.load(f)
        print(f"Loaded data presets from {presets_path}")
    
    return _parameter_presets


def resolve_preset(
    parameter: Union[str, int, float, Dict[str, Any]], 
    preset_type: str
) -> Union[Dict[str, Any], int, float]:
    """
    Resolve a parameter value that might be a preset string.
    
    Args:
        parameter: The parameter value, which might be a preset string,
                  numerical value, or dictionary configuration.
        preset_type: The type of preset ('cluster_params', 'snr_db_params',
                     or 'sample_count_distribution').
                     
    Returns:
        Resolved parameter value. If input was a recognized preset string,
        returns the corresponding parameter dictionary. Otherwise, returns
        the input unchanged.
    """
    # If not a string, return as is
    if not isinstance(parameter, str):
        return parameter
    
    # Load presets
    presets = load_parameter_presets()
    
    # If string is a known preset, return the corresponding config
    if preset_type in presets and parameter in presets[preset_type]:
        return presets[preset_type][parameter]
    
    # If string is not a recognized preset, return as is
    return parameter


class DistributionSampler:
    """
    Generic utility for sampling from different distributions.
    
    This class provides a unified interface for sampling from different
    probability distributions based on configuration dictionaries.
    """
    
    @staticmethod
    def validate_config(config: Dict[str, Any], dist_type: DistributionType) -> None:
        """
        Validate that a configuration has the required parameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Distribution configuration dictionary
        dist_type : DistributionType
            Type of distribution
            
        Raises
        ------
        ValueError
            If the configuration is missing required parameters or has invalid values
        """
        if dist_type == DistributionType.FIXED:
            if 'value' not in config and 'n' not in config:
                raise ValueError(f"Fixed distribution requires 'value' or 'n' parameter")
        elif dist_type in [DistributionType.UNIFORM, DistributionType.RANGE]:
            if 'min' not in config or 'max' not in config:
                raise ValueError(f"{dist_type} distribution requires 'min' and 'max' parameters")
        elif dist_type == DistributionType.NORMAL:
            if 'mean' not in config or 'std' not in config:
                raise ValueError(f"Normal distribution requires 'mean' and 'std' parameters")
            if config['std'] <= 0:
                raise ValueError(f"Standard deviation must be positive")
        elif dist_type == DistributionType.TRUNCATED_NORMAL:
            if 'mean' not in config or 'std' not in config or 'min' not in config or 'max' not in config:
                raise ValueError(f"Truncated normal requires 'mean', 'std', 'min', and 'max' parameters")
            if config['std'] <= 0:
                raise ValueError(f"Standard deviation must be positive")
            if config['min'] >= config['max']:
                raise ValueError(f"Min value must be less than max value")
        elif dist_type == DistributionType.TRUNCATED_LOGNORMAL:
            if 'mean' not in config or 'sigma' not in config or 'min' not in config or 'max' not in config:
                raise ValueError(f"Truncated lognormal requires 'mean', 'sigma', 'min', and 'max' parameters")
            if config['sigma'] <= 0:
                raise ValueError(f"Sigma must be positive")
            if config['min'] >= config['max']:
                raise ValueError(f"Min value must be less than max value")
        elif dist_type == DistributionType.POISSON:
            if 'lam' not in config:
                raise ValueError(f"Poisson distribution requires 'lam' parameter")
            if config['lam'] <= 0:
                raise ValueError(f"Lambda must be positive")
        elif dist_type == DistributionType.TRUNCATED_POISSON:
            if 'lam' not in config or 'min' not in config or 'max' not in config:
                raise ValueError(f"Truncated Poisson requires 'lam', 'min', and 'max' parameters")
            if config['lam'] <= 0:
                raise ValueError(f"Lambda must be positive")
            if config['min'] > config['max']:
                raise ValueError(f"Min value must be less than or equal to max value")
        elif dist_type == DistributionType.CHOICE:
            if 'options' not in config:
                raise ValueError(f"Choice distribution requires 'options' parameter")
            if not isinstance(config['options'], (list, tuple)) or len(config['options']) == 0:
                raise ValueError(f"Options must be a non-empty list or tuple")
            if 'probs' in config:
                if not isinstance(config['probs'], (list, tuple)) or len(config['probs']) != len(config['options']):
                    raise ValueError(f"Probabilities must match number of options")
                if not np.isclose(sum(config['probs']), 1.0):
                    raise ValueError(f"Probabilities must sum to 1.0")
    
    @staticmethod
    def sample(config: Dict[str, Any], rng: np.random.Generator, max_attempts: int = 100) -> Any:
        """
        Sample a value from the distribution described by the config.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Distribution configuration
        rng : np.random.Generator
            Random number generator
        max_attempts : int, optional
            Maximum attempts for truncated distributions, by default 100
            
        Returns
        -------
        Any
            Sampled value
            
        Raises
        ------
        ValueError
            If the distribution type is not supported
        """
        dist_type = DistributionType(config['type'])
        DistributionSampler.validate_config(config, dist_type)
        
        if dist_type == DistributionType.FIXED:
            return config.get('value', config.get('n'))
        
        elif dist_type == DistributionType.UNIFORM:
            min_val = config['min']
            max_val = config['max']
            return rng.uniform(min_val, max_val)
        
        elif dist_type == DistributionType.RANGE:
            min_val = config['min']
            max_val = config['max']
            return rng.integers(min_val, max_val + 1)
        
        elif dist_type == DistributionType.NORMAL:
            mean = config['mean']
            std = config['std']
            return rng.normal(mean, std)
        
        elif dist_type == DistributionType.TRUNCATED_NORMAL:
            mean = config['mean']
            std = config['std']
            min_val = config['min']
            max_val = config['max']
            
            for _ in range(max_attempts):
                sample = rng.normal(mean, std)
                if min_val <= sample <= max_val:
                    return sample
                    
            # If we reach here, we've failed to get a value in range
            # Fall back to clamping a normal sample to the range
            return np.clip(rng.normal(mean, std), min_val, max_val)
        
        elif dist_type == DistributionType.TRUNCATED_LOGNORMAL:
            # Extract parameters
            mean = config['mean']  # Mean of the underlying normal distribution
            sigma = config['sigma']  # Standard deviation of the underlying normal distribution
            min_val = config['min']
            max_val = config['max']
            
            # Calculate the normal distribution bounds that correspond to our lognormal bounds
            # ln(min_val) and ln(max_val) are the bounds in the normal space
            min_normal = np.log(min_val)
            max_normal = np.log(max_val)
            
            # For efficiency, directly sample from truncated normal distribution
            # Generate uniform sample between 0 and 1
            u = rng.uniform(0, 1)
            
            # Transform to normal CDF between min_normal and max_normal
            norm_cdf_min = 0.5 * (1 + special.erf((min_normal - mean) / (sigma * np.sqrt(2))))
            norm_cdf_max = 0.5 * (1 + special.erf((max_normal - mean) / (sigma * np.sqrt(2))))
            
            # Rescale u to be between norm_cdf_min and norm_cdf_max
            u_scaled = norm_cdf_min + u * (norm_cdf_max - norm_cdf_min)
            
            # Inverse CDF to get normal sample
            normal_sample = mean + sigma * np.sqrt(2) * special.erfinv(2 * u_scaled - 1)
            
            # Transform to lognormal
            lognormal_sample = np.exp(normal_sample)
            
            # Safety check and clipping (should rarely be needed due to the above approach)
            if not (min_val <= lognormal_sample <= max_val):
                lognormal_sample = np.clip(lognormal_sample, min_val, max_val)
                
            return lognormal_sample
        
        elif dist_type == DistributionType.POISSON:
            lam = config['lam']
            min_val = config.get('min', 0)
            max_val = config.get('max', np.inf)
            
            # Add offset if specified
            offset = config.get('offset', 0)
            sample = offset + rng.poisson(lam)
            
            # Apply limits if specified
            if min_val is not None or max_val is not None:
                sample = np.clip(sample, min_val, max_val)
            
            return sample
            
        elif dist_type == DistributionType.TRUNCATED_POISSON:
            lam = config['lam']
            min_val = config['min']
            max_val = config['max']
            offset = config.get('offset', 0)
            
            # Keep sampling until we get a value in the required range
            max_attempts = 100
            for _ in range(max_attempts):
                sample = offset + rng.poisson(lam)
                if min_val <= sample <= max_val:
                    return sample
            
            # If we couldn't get a valid sample after max attempts, return closest valid value
            return np.clip(offset + rng.poisson(lam), min_val, max_val)
        
        elif dist_type == DistributionType.CHOICE:
            options = config['options']
            probs = config.get('probs', None)
            return rng.choice(options, p=probs)
        
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    @staticmethod
    def process_config(config: Union[Dict[str, Any], int, float, str, None], default: Dict[str, Any], 
                      presets: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process a configuration to standard format.
        
        Parameters
        ----------
        config : Union[Dict[str, Any], int, float, str, None]
            Configuration as dict, number, string, or None
        default : Dict[str, Any]
            Default configuration to use if config is None
        presets : Optional[Dict[str, Dict[str, Any]]], optional
            Dictionary of preset configurations, by default None
            
        Returns
        -------
        Dict[str, Any]
            Standardized configuration dictionary
            
        Raises
        ------
        ValueError
            If the configuration format is not supported or preset name is unknown
        """
        if config is None:
            return default
            
        # Handle number as fixed value
        if isinstance(config, (int, float)):
            return {'type': 'fixed', 'value': config}
            
        # Handle string as preset name
        if isinstance(config, str) and presets is not None:
            if config not in presets:
                raise ValueError(f"Unknown preset: {config}")
            return presets[config]
            
        # Use dictionary as is
        if isinstance(config, dict):
            if 'type' not in config:
                raise ValueError(f"Configuration must include 'type' key")
            return config
            
        raise ValueError(f"Configuration must be dict, int, float, string, or None")