"""
Experiment configuration for GMM transformer framework.

This module combines all configuration components into a unified experiment config.
"""

from typing import Any, Optional
from pathlib import Path
from pydantic import Field, model_validator

from .base import ConfigBase, DeviceConfig, ExperimentMetadata, PathConfig
from .model import ClusterPredictionConfig, TransformerConfig
from .training import TrainingConfig, OptimizerConfig, SchedulerConfig, LossConfig
from .logging import LoggingConfig, VisualizationConfig
from .validation import ValidationConfig
from .metrics import MetricsConfig
from .data import DataConfig


class ExperimentConfig(ConfigBase):
    """Complete experiment configuration."""

    metadata: ExperimentMetadata = Field(
        default_factory=ExperimentMetadata,
        description="Experiment metadata"
    )

    paths: PathConfig = Field(
        default_factory=PathConfig,
        description="Path configuration"
    )

    device: DeviceConfig = Field(
        default_factory=DeviceConfig,
        description="Device configuration"
    )

    model: ClusterPredictionConfig = Field(
        default_factory=ClusterPredictionConfig,
        description="Model configuration"
    )

    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration"
    )

    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data configuration"
    )

    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and TensorBoard configuration"
    )

    @model_validator(mode='after')
    def validate_experiment_config(self) -> "ExperimentConfig":
        """Validate the complete experiment configuration."""
        # Check data dimension matches model input dimension
        if self.data.dim != self.model.input_dim:
            raise ValueError(
                f"Data dimension ({self.data.dim}) must match model input "
                f"dimension ({self.model.input_dim})"
            )

        # No assignment prediction - only center prediction is supported now

        return self

    # Helper properties to directly access nested attributes
    @property
    def transformer(self) -> TransformerConfig:
        """Direct access to transformer configuration."""
        return self.model.transformer

    @property
    def optimizer(self) -> OptimizerConfig:
        """Direct access to optimizer configuration."""
        return self.training.optimizer

    @property
    def scheduler(self) -> SchedulerConfig:
        """Direct access to scheduler configuration."""
        return self.training.scheduler

    @property
    def loss(self) -> LossConfig:
        """Direct access to loss configuration."""
        return self.training.loss

    @property
    def metrics(self) -> MetricsConfig:
        """Direct access to metrics configuration."""
        return self.validation.metrics

    @property
    def visualize(self) -> VisualizationConfig:
        """Direct access to visualization configuration."""
        return self.validation.visualize

    def get_experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        return self.paths.get_experiment_dir(self.metadata.id)

    def get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory path."""
        return self.paths.get_checkpoint_dir(self.metadata.id)

    def get_log_dir(self) -> Path:
        """Get the log directory path."""
        return self.paths.get_log_dir(self.metadata.id)

    def save_to_experiment_dir(self) -> Path:
        """
        Save configuration to experiment directory.

        Returns:
            Path to saved configuration file
        """
        # Create experiment directory
        exp_dir = self.get_experiment_dir()
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create config file path
        config_path = exp_dir / "config.json"

        # Save config
        self.save(config_path)

        return config_path

    @classmethod
    def load_from_experiment_dir(cls, experiment_id: str, base_dir: str = "./runs") -> "ExperimentConfig":
        """
        Load configuration from experiment directory.

        Args:
            experiment_id: Experiment ID
            base_dir: Base directory for experiments

        Returns:
            Loaded experiment configuration
        """
        # Create path to config file
        config_path = Path(base_dir) / experiment_id / "config.json"

        # Load config and ensure the correct return type
        config = cls.load(config_path)
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}")
        return config

    def create_data_loaders(
        self,
        batch_size: Optional[int] = None,
        num_train_samples: Optional[int] = None,
        num_val_samples: Optional[int] = None,
        device: Optional[Any] = None
    ):
        """
        Create data loaders for training and validation.

        Args:
            batch_size: Batch size for training (uses training.batch_size if None)
            num_train_samples: Number of samples per training epoch (uses training.num_train_samples if None)
            num_val_samples: Number of validation samples (uses validation.num_val_samples if None)
            device: Device to place tensors on

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Use config-specified values if not provided
        if batch_size is None:
            batch_size = self.training.batch_size
            
        if num_train_samples is None:
            num_train_samples = self.training.num_train_samples
            
        if num_val_samples is None:
            num_val_samples = self.validation.num_val_samples

        if device is None and hasattr(self.device, 'get_device'):
            device = self.device.get_device()

        # Use GMMDataLoader directly
        from data.loaders.data_loader import GMMDataLoader

        # Convert DataConfig to a dictionary
        data_config_dict = self.data.model_dump()

        # Create train and validation loaders with the static method
        train_loader, val_loader = GMMDataLoader.create_train_val_pair(
            config_dict=data_config_dict,
            train_batch_size=batch_size,
            val_batch_size=self.validation.validation_batch_size,
            train_samples=num_train_samples,
            val_samples=num_val_samples,
            device=device,
            base_seed=self.data.random_seed,
            fixed_data=False,  # Training data is always dynamic
            fixed_validation_data=self.validation.fixed_validation_data,  # Use the validation config parameter
            state_dir="output/states"
        )

        return train_loader, val_loader

