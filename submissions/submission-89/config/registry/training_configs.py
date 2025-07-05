"""
Registry for training configurations.

This module provides registry access to predefined training configurations for various scenarios.
"""

import os
import json
from typing import Dict, Any, List

from ..training import TrainingConfig

# Path to training config file
CONFIG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_CONFIG_PATH = os.path.join(CONFIG_ROOT, 'config', 'json_defaults', 'training', 'defaults.json')

# No hardcoded fallbacks - all configurations are in JSON files

# Cache for loaded configs
_training_configs = None

def _load_training_configs() -> Dict[str, Any]:
    """Load training configurations from JSON file."""
    global _training_configs
    
    if _training_configs is not None:
        return _training_configs
    
    if not os.path.exists(TRAINING_CONFIG_PATH):
        raise FileNotFoundError(
            f"Training configuration file not found at {TRAINING_CONFIG_PATH}. "
            f"Please make sure the file exists."
        )
    
    try:
        with open(TRAINING_CONFIG_PATH, 'r') as f:
            _training_configs = json.load(f)
            return _training_configs
    except Exception as e:
        raise RuntimeError(
            f"Error loading training configurations from {TRAINING_CONFIG_PATH}: {e}. "
            f"Please check the file format and permissions."
        )

def get_training_config(preset_name: str) -> TrainingConfig:
    """
    Get a training configuration by preset name.
    
    Args:
        preset_name: Name of the preset to use. Available presets:
            - 'quick': Fast training with aggressive learning rate for rapid experimentation
            - 'standard': Balanced training with moderate settings and good convergence
            - 'optimized': High learning rate for faster convergence with regularization
            - 'high_performance': Extended training for maximum accuracy
              
    Returns:
        TrainingConfig instance with preset settings
        
    Raises:
        ValueError: If preset name is not found
    """
    configs = _load_training_configs()
    
    if preset_name not in configs:
        valid_presets = list(configs.keys())
        raise ValueError(f"Unknown training preset '{preset_name}'. Valid options: {valid_presets}")
    
    return TrainingConfig.model_validate(configs[preset_name])

def list_training_presets() -> List[str]:
    """
    Get all available training configuration presets.
    
    Returns:
        List of available training preset names
    """
    return list(_load_training_configs().keys())