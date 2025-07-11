"""
Registry of predefined model configurations.

This module provides a collection of pre-defined model architectures
with various parameter counts and configurations for different use cases.
It also includes training and data generation presets to create complete
experiment configurations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic

from ..model import ClusterPredictionConfig
from ..training import TrainingConfig
from ..experiment import DataConfig, ValidationConfig, ExperimentConfig

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar('T')
ConfigDict = Dict[str, Any]


class ConfigRegistry(Generic[T]):
    """
    Generic configuration registry for managing presets.
    
    This class handles loading JSON configurations and provides
    type-safe access to configuration presets.
    
    Args:
        name: Name of the configuration registry, for logging
        config_path: Path to JSON configuration file
        create_fn: Function to convert dictionary to typed configuration
    """
    
    def __init__(
        self,
        name: str,
        config_path: str,
        create_fn: Callable[[ConfigDict], T]
    ):
        self.name = name
        self.config_path = config_path
        self.create_fn = create_fn
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, ConfigDict]:
        """Load configurations from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"{self.name} configuration file not found at {self.config_path}. "
                f"Please make sure the file exists."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                configs = json.load(f)
            logger.debug(f"Loaded {self.name} presets from {self.config_path}")
            return configs
        except Exception as e:
            raise RuntimeError(
                f"Error loading {self.name} configurations from {self.config_path}: {e}. "
                f"Please check the file format and permissions."
            )
    
    def get(self, preset_name: str) -> T:
        """
        Get a configuration by preset name.
        
        Args:
            preset_name: Name of the preset to use
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in self.configs:
            preset_names = list(self.configs.keys())
            raise ValueError(f"Unknown {self.name} preset: '{preset_name}'. Available options: {preset_names}")
        
        config_dict = self.configs[preset_name]
        return self.create_fn(config_dict)
        
    def get_description(self, preset_name: str) -> Optional[str]:
        """
        Get description for a preset if available.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Description string or None if not available
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in self.configs:
            preset_names = list(self.configs.keys())
            raise ValueError(f"Unknown {self.name} preset: '{preset_name}'. Available options: {preset_names}")
        
        config_dict = self.configs[preset_name]
        
        # Check if description is included in the preset configuration
        if isinstance(config_dict, dict) and "description" in config_dict:
            return config_dict["description"]
        
        # No description found
        return None
    
    def get_all_presets(self) -> List[str]:
        """Get list of all available preset names."""
        return list(self.configs.keys())
    
    def register(self, preset_name: str, config_dict: ConfigDict) -> None:
        """
        Register a new configuration preset.
        
        Args:
            preset_name: Name for the configuration preset
            config_dict: Configuration dictionary
        """
        self.configs[preset_name] = config_dict


class ExperimentRegistry:
    """
    Registry of predefined configurations for experiments.
    
    This registry provides access to model, training, data, and validation presets,
    allowing for easy creation of complete experiment configurations.
    It also provides access to documentation for each preset type.
    """
    
    # Default config paths
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'json_defaults')
    MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, 'model', 'defaults.json')
    TRAINING_CONFIG_PATH = os.path.join(CONFIG_DIR, 'training', 'defaults.json')
    DATA_CONFIG_PATH = os.path.join(CONFIG_DIR, 'data', 'defaults.json')
    VALIDATION_CONFIG_PATH = os.path.join(CONFIG_DIR, 'validation', 'defaults.json')
    
    # Initialize registries
    _model_registry = ConfigRegistry[ClusterPredictionConfig](
        name="Model",
        config_path=MODEL_CONFIG_PATH,
        create_fn=lambda d: ClusterPredictionConfig.model_validate(d)
    )
    
    _training_registry = ConfigRegistry[TrainingConfig](
        name="Training",
        config_path=TRAINING_CONFIG_PATH,
        create_fn=lambda d: TrainingConfig.model_validate(d)
    )
    
    _data_registry = ConfigRegistry[DataConfig](
        name="Data",
        config_path=DATA_CONFIG_PATH,
        create_fn=lambda d: DataConfig.model_validate(d)
    )
    
    _validation_registry = ConfigRegistry[ValidationConfig](
        name="Validation",
        config_path=VALIDATION_CONFIG_PATH,
        create_fn=lambda d: ValidationConfig.model_validate(d)
    )
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ClusterPredictionConfig:
        """
        Get model configuration by name.
        
        Args:
            model_name: Model configuration name
            
        Returns:
            ClusterPredictionConfig instance
            
        Raises:
            ValueError: If model_name is not recognized
        """
        return cls._model_registry.get(model_name)
    
    @classmethod
    def get_training_config(cls, preset_name: str) -> TrainingConfig:
        """
        Get training configuration by preset name.
        
        Args:
            preset_name: Training preset name
            
        Returns:
            TrainingConfig instance
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        return cls._training_registry.get(preset_name)
    
    @classmethod
    def get_data_config(cls, preset_name: str) -> DataConfig:
        """
        Get data configuration by preset name.
        
        Args:
            preset_name: Data preset name
            
        Returns:
            DataConfig instance
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        return cls._data_registry.get(preset_name)
    
    @classmethod
    def get_validation_config(cls, preset_name: str) -> ValidationConfig:
        """
        Get validation configuration by preset name.
        
        Args:
            preset_name: Validation preset name
            
        Returns:
            ValidationConfig instance
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        return cls._validation_registry.get(preset_name)
    
    @classmethod
    def list_model_presets(cls) -> List[str]:
        """Get list of all available model presets."""
        return cls._model_registry.get_all_presets()
    
    @classmethod
    def list_training_presets(cls) -> List[str]:
        """Get list of all available training presets."""
        return cls._training_registry.get_all_presets()
    
    @classmethod
    def list_data_presets(cls) -> List[str]:
        """Get list of all available data presets."""
        return cls._data_registry.get_all_presets()
    
    @classmethod
    def list_validation_presets(cls) -> List[str]:
        """Get list of all available validation presets."""
        return cls._validation_registry.get_all_presets()
        
    @classmethod
    def get_model_description(cls, model_name: str) -> Optional[str]:
        """
        Get description for a model preset.
        
        Args:
            model_name: Name of the model preset
            
        Returns:
            Description string or None if not available
        """
        return cls._model_registry.get_description(model_name)
        
    @classmethod
    def get_training_description(cls, preset_name: str) -> Optional[str]:
        """
        Get description for a training preset.
        
        Args:
            preset_name: Name of the training preset
            
        Returns:
            Description string or None if not available
        """
        return cls._training_registry.get_description(preset_name)
        
    @classmethod
    def get_data_description(cls, preset_name: str) -> Optional[str]:
        """
        Get description for a data preset.
        
        Args:
            preset_name: Name of the data preset
            
        Returns:
            Description string or None if not available
        """
        return cls._data_registry.get_description(preset_name)
        
    @classmethod
    def get_validation_description(cls, preset_name: str) -> Optional[str]:
        """
        Get description for a validation preset.
        
        Args:
            preset_name: Name of the validation preset
            
        Returns:
            Description string or None if not available
        """
        return cls._validation_registry.get_description(preset_name)
    
    @classmethod
    def get_experiment_config(
        cls,
        model_name: str = "medium",
        training_preset: str = "standard",
        data_preset: str = "standard",
        validation_preset: str = "standard",
        device: str = "cuda",
        exp_name: Optional[str] = None,
        exp_id: Optional[str] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        use_mixed_precision: Optional[bool] = None,
        **kwargs
    ) -> ExperimentConfig:
        """
        Create a complete experiment configuration from presets.
        
        Args:
            model_name: Model preset name 
            training_preset: Training configuration preset
            data_preset: Data generation preset
            validation_preset: Validation preset
            device: Device to use for training ('cpu', 'cuda', etc.)
            exp_name: Experiment name (defaults to model_name)
            exp_id: Experiment ID (defaults to auto-generated)
            learning_rate: Override learning rate
            batch_size: Override batch size
            num_epochs: Override number of epochs
            use_mixed_precision: Override mixed precision setting
            **kwargs: Additional parameters to override
            
        Returns:
            Complete ExperimentConfig instance
        """
        # Set default experiment name if not provided
        if exp_name is None:
            exp_name = f"{model_name}_{training_preset}_{data_preset}"
        
        # Get configurations from registries
        model_config = cls.get_model_config(model_name)
        training_config = cls.get_training_config(training_preset)
        data_config = cls.get_data_config(data_preset)
        validation_config = cls.get_validation_config(validation_preset)
        
        # Override specific training parameters if provided
        if learning_rate is not None:
            training_config.optimizer.learning_rate = learning_rate
        if batch_size is not None:
            training_config.batch_size = batch_size
        if num_epochs is not None:
            training_config.num_epochs = num_epochs
        if use_mixed_precision is not None:
            # Set this directly in device config
            pass
        
        # Create metadata
        from ..base import ExperimentMetadata
        metadata = ExperimentMetadata(experiment_name=exp_name, id=exp_id)
        
        # Create device config
        from ..base import DeviceConfig
        device_config = DeviceConfig(
            device=device,
            use_mixed_precision=use_mixed_precision if use_mixed_precision is not None else False
        )
        
        # Create path config
        from ..base import PathConfig
        path_config = PathConfig()
        
        # Create experiment config
        config = ExperimentConfig(
            metadata=metadata,
            device=device_config,
            paths=path_config,
            model=model_config,
            training=training_config,
            data=data_config,
            validation=validation_config
        )
        
        # Apply any additional overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config