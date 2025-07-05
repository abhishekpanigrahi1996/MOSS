"""
Registry for model configurations.

This module provides registry access to predefined model configurations for various model sizes.
"""

import os
import json
from typing import Dict, Any, List

from ..model import ClusterPredictionConfig

# Path to model config file
CONFIG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_CONFIG_PATH = os.path.join(CONFIG_ROOT, 'config', 'json_defaults', 'model', 'defaults.json')

# No hardcoded fallbacks - all configurations are in JSON files

# Cache for loaded configs
_model_configs = None

def _load_model_configs() -> Dict[str, Any]:
    """Load model configurations from JSON file."""
    global _model_configs
    
    if _model_configs is not None:
        return _model_configs
    
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(
            f"Model configuration file not found at {MODEL_CONFIG_PATH}. "
            f"Please make sure the file exists."
        )
    
    try:
        with open(MODEL_CONFIG_PATH, 'r') as f:
            _model_configs = json.load(f)
            return _model_configs
    except Exception as e:
        raise RuntimeError(
            f"Error loading model configurations from {MODEL_CONFIG_PATH}: {e}. "
            f"Please check the file format and permissions."
        )

def get_model_config(model_name: str = "medium") -> ClusterPredictionConfig:
    """
    Get a model configuration by name.
    
    Args:
        model_name: Name of the model preset to use. Available presets:
            - 'tiny': Small model for quick experimentation (32 dim, 4 layers, 1 head)
            - 'small': Compact model for faster training (64 dim, 8 layers, 2 heads)
            - 'medium': Default model with balanced performance (128 dim, 16 layers, 4 heads)
            - 'large': Large model for maximum performance (256 dim, 48 layers, 8 heads)
              
    Returns:
        ClusterPredictionConfig instance with preset settings
        
    Raises:
        ValueError: If model name is not found
    """
    configs = _load_model_configs()
    
    if model_name not in configs:
        valid_presets = list(configs.keys())
        raise ValueError(f"Unknown model preset '{model_name}'. Valid options: {valid_presets}")
    
    return ClusterPredictionConfig.model_validate(configs[model_name])

def list_model_presets() -> List[str]:
    """
    Get all available model configuration presets.
    
    Returns:
        List of available model preset names
    """
    return list(_load_model_configs().keys())