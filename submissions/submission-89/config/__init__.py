"""
Configuration components for GMM transformer framework.

This package provides a comprehensive configuration system using Pydantic
for type checking and validation.
"""

from .base import (
    ConfigBase,
    DeviceConfig,
    ExperimentMetadata,
    PathConfig
)

from .model import (
    TransformerConfig,
    ClusterPredictionConfig
)

from .training import (
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    TrainingConfig
)

from .logging import (
    LoggingConfig,
    VisualizationConfig
)

from .metrics import (
    MetricsConfig
)

from .validation import (
    ValidationConfig
)

from .data import (
    DataConfig
)

from .experiment import (
    ExperimentConfig
)

# Import registry
from .registry import (
    ConfigRegistry,
    ExperimentRegistry,
    get_model_config,
    get_training_config,
    get_data_config,
    get_validation_config,
    list_model_presets,
    list_training_presets,
    list_data_presets,
    list_validation_presets
)

# Import presets
from .presets import (
    get_preset_config,
    list_experiment_presets,
    get_preset_description
)

__all__ = [
    # Base configs
    'ConfigBase',
    'DeviceConfig',
    'ExperimentMetadata',
    'PathConfig',
    
    # Model configs
    'TransformerConfig',
    'ClusterPredictionConfig',
    
    # Training configs
    'OptimizerConfig',
    'SchedulerConfig',
    'LossConfig',
    'TrainingConfig',
    
    # Logging configs
    'LoggingConfig',
    'VisualizationConfig',
    
    # Metrics configs
    'MetricsConfig',
    
    # Validation configs
    'ValidationConfig',
    
    # Data configs
    'DataConfig',
    
    # Experiment configs
    'ExperimentConfig',
    
    # Registry
    'ConfigRegistry',
    'ExperimentRegistry',
    'get_model_config',
    'get_training_config',
    'get_data_config',
    'get_validation_config',
    'list_model_presets',
    'list_training_presets',
    'list_data_presets',
    'list_validation_presets',
    
    # Presets
    'get_preset_config',
    'list_experiment_presets',
    'get_preset_description'
]