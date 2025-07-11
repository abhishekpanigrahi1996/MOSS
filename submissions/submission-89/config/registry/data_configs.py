"""
Registry for data configurations.

This module provides registry access to predefined data configurations for various scenarios.
"""

import os
import json
from typing import Dict, Any, List

from ..experiment import DataConfig

# Path to data config file
CONFIG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_CONFIG_PATH = os.path.join(CONFIG_ROOT, 'config', 'json_defaults', 'data', 'defaults.json')

# Allow overriding the data config path via environment variable
DATA_CONFIG_PATH = os.environ.get('GMM_DATA_CONFIG_PATH', DEFAULT_DATA_CONFIG_PATH)

# No hardcoded fallbacks - all configurations are in JSON files

# Cache for loaded configs
_data_configs = None
_last_config_path = None

def _load_data_configs() -> Dict[str, Any]:
    """Load data configurations from JSON file."""
    global _data_configs, _last_config_path
    
    # If path changed, invalidate cache
    if _last_config_path != DATA_CONFIG_PATH:
        _data_configs = None
        _last_config_path = DATA_CONFIG_PATH
    
    if _data_configs is not None:
        return _data_configs
    
    if not os.path.exists(DATA_CONFIG_PATH):
        raise FileNotFoundError(
            f"Data configuration file not found at {DATA_CONFIG_PATH}. "
            f"Please make sure the file exists."
        )
    
    try:
        with open(DATA_CONFIG_PATH, 'r') as f:
            _data_configs = json.load(f)
            print(f"Loaded data presets from {DATA_CONFIG_PATH}")
            return _data_configs
    except Exception as e:
        raise RuntimeError(
            f"Error loading data configurations from {DATA_CONFIG_PATH}: {e}. "
            f"Please check the file format and permissions."
        )

def get_data_config(preset_name: str = "standard") -> DataConfig:
    """
    Get a data configuration by preset name.
    
    Args:
        preset_name: Name of the preset to use. Available presets:
            - 'simple': Basic data generation with fewer clusters
            - 'standard': Default data generation for most use cases
            - 'complex': Advanced data generation with more clusters and variation
            - '3d': 3D data points instead of 2D
            - 'challenging': Most difficult scenario with many clusters and low SNR
              
    Returns:
        DataConfig instance with preset settings
        
    Raises:
        ValueError: If preset name is not found
    """
    configs = _load_data_configs()
    
    if preset_name not in configs:
        valid_presets = list(configs.keys())
        raise ValueError(f"Unknown data preset '{preset_name}'. Valid options: {valid_presets}")
    
    return DataConfig.model_validate(configs[preset_name])

def list_data_presets() -> List[str]:
    """
    Get all available data configuration presets.
    
    Returns:
        List of available data preset names
    """
    return list(_load_data_configs().keys())