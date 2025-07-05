"""
Training configuration classes for GMM transformer framework.

This module defines the configuration options for training, 
optimization, and learning rate schedules.
"""

from typing import Literal, Dict, Any, Optional, List, Union
from pydantic import Field, field_validator, model_validator

from .base import ConfigBase


OptimType = Literal["adam", "adamw", "sgd", "adagrad", "rmsprop"]
SchedType = Literal["constant", "linear", "cosine", "exponential", "step"]
LossType = Literal["mse", "wasserstein", "energy"]
LossNormalizationType = Literal["none", "snr_power", "log"]


class OptimizerConfig(ConfigBase):
    """Configuration for optimizer settings."""
    
    optimizer: OptimType = Field(
        default="adamw",
        description="Optimizer type"
    )
    
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate"
    )
    
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay (L2 regularization)"
    )
    
    beta1: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Beta1 for Adam-based optimizers"
    )
    
    beta2: float = Field(
        default=0.999,
        gt=0.0,
        lt=1.0,
        description="Beta2 for Adam-based optimizers"
    )
    
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="Momentum for SGD"
    )
    
    exclude_bias_and_norm: bool = Field(
        default=True,
        description="Whether to exclude bias and normalization parameters from weight decay"
    )


class SchedulerConfig(ConfigBase):
    """Configuration for learning rate scheduler."""
    
    scheduler_type: SchedType = Field(
        default="cosine",
        description="Learning rate scheduler type"
    )
    
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup steps"
    )
    
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Ratio of total steps to use for warmup"
    )
    
    min_lr_ratio: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Minimum learning rate as a fraction of initial lr"
    )
    
    decay_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of steps for decay (for step and exponential schedulers)"
    )
    
    decay_rate: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Decay rate for exponential and step schedulers"
    )
    
    @model_validator(mode='after')
    def validate_scheduler_config(self) -> "SchedulerConfig":
        """Validate scheduler configuration."""
        if self.scheduler_type in ["step", "exponential"] and self.decay_steps is None:
            raise ValueError(
                f"decay_steps must be specified for {self.scheduler_type} scheduler"
            )
        return self


class LossConfig(ConfigBase):
    """Configuration for loss function."""
    
    loss_type: Union[str, Dict[str, Any]] = Field(
        default="wasserstein",
        description="Type of loss function or dict with detailed configuration"
    )
    
    normalization: LossNormalizationType = Field(
        default="none",
        description="How to normalize the loss: none, snr_power, or log"
    )
    
    snr_power: float = Field(
        default=0.0,
        description="Power to raise SNR to when using snr_power normalization (e.g., 1.0, 1.33, etc.)"
    )
    
    # Wasserstein loss parameters (compatible with old format)
    wasserstein_algorithm: Literal["exact", "sinkhorn"] = Field(
        default="exact",
        description="Algorithm for Wasserstein distance"
    )
    
    wasserstein_backend: Literal["auto", "jax", "pot", "scipy"] = Field(
        default="pot",
        description="Backend for Wasserstein distance computation"
    )
    
    wasserstein_epsilon: float = Field(
        default=0.01,
        gt=0.0,
        description="Regularization parameter for Sinkhorn algorithm"
    )
    
    wasserstein_max_iter: int = Field(
        default=100,
        ge=1,
        description="Maximum iterations for Sinkhorn algorithm"
    )
    
    use_true_weights: bool = Field(
        default=False,
        description="Whether to use true mixture weights from GMM parameters instead of counting labels"
    )
    
    @field_validator('loss_type')
    def validate_loss_type(cls, v):
        """Validate loss_type can be string or dict."""
        if isinstance(v, str):
            # For string format, remove previous restrictive validation
            return v
        elif isinstance(v, dict):
            # Ensure dict has a 'type' key
            if 'type' not in v:
                raise ValueError("Loss configuration dictionary must have a 'type' key")
            return v
        else:
            raise ValueError(f"loss_type must be a string or dictionary, got {type(v)}")
            
    @model_validator(mode='after')
    def update_wasserstein_params_from_string(self) -> "LossConfig":
        """
        Update wasserstein parameters based on extended string format.
        
        This ensures that fields like wasserstein_algorithm and wasserstein_backend
        are updated when using extended string formats like "wasserstein_sinkhorn_pot".
        """
        if isinstance(self.loss_type, str) and self.loss_type.startswith("wasserstein_"):
            # Import the parser
            try:
                from metrics.configuration import parse_loss_config
                
                # Parse the string to extract parameters
                parsed = parse_loss_config(self.loss_type)
                
                # Update the object's fields if parameters are specified in the string
                if "algorithm" in parsed:
                    self.wasserstein_algorithm = parsed["algorithm"]
                if "backend" in parsed:
                    self.wasserstein_backend = parsed["backend"]
            except ImportError:
                pass
                
        return self
    
    @field_validator('wasserstein_backend')
    def validate_backend(cls, v: str) -> str:
        """Validate Wasserstein backend."""
        valid_options = {'auto', 'jax', 'pot', 'scipy'}
        
        if v not in valid_options:
            raise ValueError(
                f"Invalid Wasserstein backend: {v}. Must be one of {valid_options}"
            )
            
        return v
        
    def create_loss_function(self):
        """
        Create a loss function based on the configuration.
        
        This method uses the loss factory utility to convert the configuration
        to a callable loss function.
        
        Returns:
            A callable loss function that takes (predictions, targets) parameters
        """
        import logging
        # Use metrics package
        try:
            from metrics import create_loss_from_config
            from metrics.configuration import parse_loss_config
            logger = logging.getLogger(__name__)
            
            # If loss_type is a dictionary, use it directly
            if isinstance(self.loss_type, dict):
                return create_loss_from_config(self.loss_type)
            
            # For wasserstein with extended string format (e.g., "wasserstein_exact_jax")
            # or plain "wasserstein", create a detailed configuration
            if isinstance(self.loss_type, str) and self.loss_type.startswith("wasserstein"):
                # Parse the configuration to extract parameters
                parsed_config = parse_loss_config(self.loss_type)
                
                # Start with the class parameters
                wasserstein_config = {
                    "type": "wasserstein",
                    "algorithm": self.wasserstein_algorithm,
                    "backend": self.wasserstein_backend,
                    "epsilon": self.wasserstein_epsilon,
                    "max_iterations": self.wasserstein_max_iter,
                    "use_true_weights": self.use_true_weights
                }
                
                # Override with any parameters extracted from the string
                if "algorithm" in parsed_config:
                    wasserstein_config["algorithm"] = parsed_config["algorithm"]
                if "backend" in parsed_config:
                    wasserstein_config["backend"] = parsed_config["backend"]
                    
                return create_loss_from_config(wasserstein_config)
            else:
                # Handle other loss types (mse, energy)
                return create_loss_from_config(self.loss_type)
        except ImportError:
            raise ImportError("metrics package not found. Please install it to use this functionality.")


class TrainingConfig(ConfigBase):
    """Configuration for training process."""
    
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Training batch size"
    )
    
    num_epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs"
    )
    
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer configuration"
    )
    
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler configuration"
    )
    
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description="Loss function configuration"
    )
    
    gradient_clip_val: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Gradient clipping value (None for no clipping)"
    )
    
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of steps to accumulate gradients"
    )
    
    early_stopping_patience: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of epochs to wait for improvement before stopping"
    )
    
    early_stopping_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum change to qualify as improvement"
    )
    
    save_best_model: bool = Field(
        default=True,
        description="Whether to save the best model based on validation loss"
    )
    
    num_train_samples: int = Field(
        default=32768,  # 2^15
        ge=1,
        description="Number of training samples per epoch (for dynamic data)"
    )
    
    val_every: int = Field(
        default=1,
        ge=1,
        description="Validate every N epochs"
    )
    
    checkpoint_every: int = Field(
        default=0,
        ge=0,
        description="Save checkpoint every N epochs (0 to disable regular checkpoints)"
    )
    
    show_progress_bar: bool = Field(
        default=True,
        description="Whether to show progress bar during training"
    )