"""
Preset configurations for GMM transformer experiments.

This module provides easy access to predefined experiment configurations
for common use cases and experiments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

from .experiment import ExperimentConfig
from .registry import ExperimentRegistry

logger = logging.getLogger(__name__)

# Path to experiment presets
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json_defaults')
EXPERIMENT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'experiment_simple.json')

# Cache for loaded presets
_experiment_presets = None

def _load_experiment_presets() -> Dict[str, Dict[str, str]]:
    """Load experiment presets from JSON file."""
    global _experiment_presets
    
    if _experiment_presets is not None:
        return _experiment_presets
    
    try:
        if os.path.exists(EXPERIMENT_CONFIG_PATH):
            with open(EXPERIMENT_CONFIG_PATH, 'r') as f:
                _experiment_presets = json.load(f)
                return _experiment_presets
    except Exception as e:
        logger.warning(f"Could not load experiment presets from {EXPERIMENT_CONFIG_PATH}: {e}")
        logger.warning("Using built-in default presets.")
    
    # Fallback to default presets
    _experiment_presets = {
        "quick_test": {
            "model": "tiny",
            "training": "quick",
            "data": "simple",
            "validation": "minimal"
        },
        "standard": {
            "model": "medium",
            "training": "standard",
            "data": "standard",
            "validation": "standard"
        },
        "high_performance": {
            "model": "large",
            "training": "optimized",
            "data": "complex",
            "validation": "comprehensive"
        }
    }
    
    return _experiment_presets

def get_preset_config(
    preset_name: str = "standard",
    device: str = "cuda",
    exp_name: Optional[str] = None,
    exp_id: Optional[str] = None,
    **override_kwargs
) -> ExperimentConfig:
    """
    Get a complete experiment configuration using a named preset.
    
    Args:
        preset_name: Name of the preset configuration to use
        device: Device to use for training
        exp_name: Optional experiment name override
        exp_id: Optional experiment ID override
        **override_kwargs: Additional kwargs to override specific settings
        
    Returns:
        Fully configured ExperimentConfig instance
        
    Raises:
        ValueError: If preset_name is not recognized
    """
    presets = _load_experiment_presets()
    
    if preset_name not in presets:
        valid_presets = list(presets.keys())
        raise ValueError(f"Unknown experiment preset '{preset_name}'. Valid options: {valid_presets}")
    
    preset = presets[preset_name]
    
    # Default name to preset if not provided
    if exp_name is None:
        exp_name = f"{preset_name}_experiment"
    
    # Create experiment config from preset
    config = ExperimentRegistry.get_experiment_config(
        model_name=preset.get("model", "medium"),
        training_preset=preset.get("training", "standard"),
        data_preset=preset.get("data", "standard"),
        validation_preset=preset.get("validation", "standard"),
        device=device,
        exp_name=exp_name,
        exp_id=exp_id,
        **override_kwargs
    )
    
    return config

def list_experiment_presets() -> List[str]:
    """
    Get list of all available experiment presets.
    
    Returns:
        List of preset names
    """
    return list(_load_experiment_presets().keys())

def get_preset_description(preset_name: str) -> Dict[str, str]:
    """
    Get the component configuration presets for a named experiment preset.
    
    Args:
        preset_name: Name of the preset to describe
        
    Returns:
        Dictionary with model, training, data, and validation preset names
        
    Raises:
        ValueError: If preset_name is not recognized
    """
    presets = _load_experiment_presets()
    
    if preset_name not in presets:
        valid_presets = list(presets.keys())
        raise ValueError(f"Unknown experiment preset '{preset_name}'. Valid options: {valid_presets}")
        
    return presets[preset_name]