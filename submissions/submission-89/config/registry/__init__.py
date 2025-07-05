"""
Registry module for experiment configurations.

This module provides centralized access to configuration presets
for models, training, data generation, and validation.
"""

from .registry import ConfigRegistry, ExperimentRegistry
from .model_configs import get_model_config, list_model_presets
from .training_configs import get_training_config, list_training_presets
from .data_configs import get_data_config, list_data_presets
from .validation_configs import get_validation_config, list_validation_presets

# Import description functions directly from ExperimentRegistry
from .registry import ExperimentRegistry as _ER
get_model_description = _ER.get_model_description
get_training_description = _ER.get_training_description
get_data_description = _ER.get_data_description
get_validation_description = _ER.get_validation_description

__all__ = [
    # General registries
    'ConfigRegistry',
    'ExperimentRegistry',
    
    # Model config utilities
    'get_model_config',
    'list_model_presets',
    'get_model_description',
    
    # Training config utilities
    'get_training_config',
    'list_training_presets',
    'get_training_description',
    
    # Data config utilities
    'get_data_config',
    'list_data_presets',
    'get_data_description',
    
    # Validation config utilities
    'get_validation_config',
    'list_validation_presets',
    'get_validation_description',
]