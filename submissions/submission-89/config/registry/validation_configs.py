"""
Registry for validation configurations.

This module provides registry access to predefined validation configurations for various scenarios.
"""

import os
import json
from typing import Dict, Any, List

from ..experiment import ValidationConfig

# Path to validation config file
CONFIG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALIDATION_CONFIG_PATH = os.path.join(CONFIG_ROOT, 'config', 'json_defaults', 'validation', 'defaults.json')

# No hardcoded fallbacks - all configurations are in JSON files

# Cache for loaded configs
_validation_configs = None

def _load_validation_configs() -> Dict[str, Any]:
    """Load validation configurations from JSON file."""
    global _validation_configs
    
    if _validation_configs is not None:
        return _validation_configs
    
    if not os.path.exists(VALIDATION_CONFIG_PATH):
        raise FileNotFoundError(
            f"Validation configuration file not found at {VALIDATION_CONFIG_PATH}. "
            f"Please make sure the file exists."
        )
    
    try:
        with open(VALIDATION_CONFIG_PATH, 'r') as f:
            _validation_configs = json.load(f)
            return _validation_configs
    except Exception as e:
        raise RuntimeError(
            f"Error loading validation configurations from {VALIDATION_CONFIG_PATH}: {e}. "
            f"Please check the file format and permissions."
        )

def get_validation_config(preset_name: str = "standard") -> ValidationConfig:
    """
    Get a validation configuration by preset name.
    
    Args:
        preset_name: Name of the preset to use. Available presets:
            - 'minimal': Basic validation with minimum metrics
            - 'standard': Default validation for most use cases
            - 'comprehensive': Complete validation with all metrics and visualizations
              
    Returns:
        ValidationConfig instance with preset settings
        
    Raises:
        ValueError: If preset name is not found
    """
    configs = _load_validation_configs()
    
    if preset_name not in configs:
        valid_presets = list(configs.keys())
        raise ValueError(f"Unknown validation preset '{preset_name}'. Valid options: {valid_presets}")
    
    return ValidationConfig.model_validate(configs[preset_name])

def list_validation_presets() -> List[str]:
    """
    Get all available validation configuration presets.
    
    Returns:
        List of available validation preset names
    """
    return list(_load_validation_configs().keys())