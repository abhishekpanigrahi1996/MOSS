"""
Base configuration classes for GMM framework.

This module provides the foundation for configuration management using Pydantic.
"""

import os
import json
import uuid
import logging
from typing import Dict, Any, Optional, Union, List, ClassVar, Type
from datetime import datetime
from pathlib import Path

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ConfigBase(BaseModel):
    """Base class for all configuration objects."""
    
    @property
    def name(self) -> str:
        """Get configuration name based on class name."""
        return self.__class__.__name__
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConfigBase":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            Loaded configuration object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls.model_validate(data)
        logger.info(f"Configuration loaded from {path}")
        
        return config
    
    def merge_with(self, other: Union[Dict[str, Any], "ConfigBase"]) -> "ConfigBase":
        """
        Merge configuration with another configuration or dictionary.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            New configuration with merged values
        """
        if isinstance(other, ConfigBase):
            other_dict = other.model_dump()
        elif isinstance(other, dict):
            other_dict = other
        else:
            raise TypeError(f"Cannot merge with {type(other)}")
        
        # Create a new config with merged values
        my_dict = self.model_dump()
        merged = {**my_dict, **other_dict}
        
        return self.__class__.model_validate(merged)


class DeviceConfig(ConfigBase):
    """Configuration for device settings."""
    
    device: str = Field(
        default="auto",
        description="Device to use ('cpu', 'cuda', 'cuda:0', 'auto')"
    )
    
    use_mixed_precision: bool = Field(
        default=False, 
        description="Whether to use mixed precision training"
    )
    
    precision: str = Field(
        default="float32",
        description="Precision for mixed precision training",
    )
    
    compile_model: bool = Field(
        default=False,
        description="Whether to compile model using torch.compile() (requires PyTorch 2.0+)"
    )
    
    compile_mode: str = Field(
        default="default",
        description="Compilation mode for torch.compile()"
    )
    
    @field_validator('device')
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        valid_options = {'cpu', 'cuda', 'auto'}
        
        # For auto device selection
        if v == 'auto':
            return v
            
        # Basic device specifications
        if v in valid_options:
            return v
            
        # CUDA device specification with index
        if v.startswith('cuda:') and v[5:].isdigit():
            return v
            
        raise ValueError(
            f"Invalid device: {v}. Must be one of {valid_options} or 'cuda:N'"
        )
    
    @field_validator('precision')
    def validate_precision(cls, v: str) -> str:
        """Validate precision specification."""
        valid_options = {'float32', 'float16', 'bfloat16'}
        
        if v not in valid_options:
            raise ValueError(
                f"Invalid precision: {v}. Must be one of {valid_options}"
            )
            
        return v
    
    def get_device(self) -> torch.device:
        """
        Get torch device based on configuration.
        
        Returns:
            PyTorch device object
        """
        if self.device == "auto":
            # Automatically select the best available device
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        
        # Otherwise use the specified device
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """
        Get torch dtype based on configuration.
        
        Returns:
            PyTorch dtype for tensors
        """
        if not self.use_mixed_precision:
            return torch.float32
            
        if self.precision == "float16":
            return torch.float16
        elif self.precision == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32


class ExperimentMetadata(ConfigBase):
    """Metadata for experiment tracking."""
    
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Unique identifier for the experiment"
    )
    
    experiment_name: str = Field(
        default="default_experiment",
        description="Human-readable name for the experiment"
    )
    
    description: str = Field(
        default="",
        description="Description of the experiment"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the experiment"
    )
    
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when the experiment was created"
    )
    
    updated_at: Optional[str] = Field(
        default=None,
        description="Timestamp when the experiment was last updated"
    )
    
    user: Optional[str] = Field(
        default=None,
        description="User who created the experiment"
    )


class PathConfig(ConfigBase):
    """Configuration for file paths."""
    
    base_dir: str = Field(
        default="./runs",
        description="Base directory for experiment outputs"
    )
    
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for model checkpoints (relative to experiment dir)"
    )
    
    log_dir: str = Field(
        default="logs",
        description="Directory for TensorBoard logs (relative to experiment dir)"
    )
    
    data_dir: str = Field(
        default="data",
        description="Directory for dataset files (relative to base dir)"
    )
    
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """
        Get full path to experiment directory.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Path to experiment directory
        """
        return Path(self.base_dir) / experiment_id
    
    def get_checkpoint_dir(self, experiment_id: str) -> Path:
        """
        Get full path to checkpoint directory.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Path to checkpoint directory
        """
        return self.get_experiment_dir(experiment_id) / self.checkpoint_dir
    
    def get_log_dir(self, experiment_id: str) -> Path:
        """
        Get full path to log directory.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Path to log directory
        """
        return self.get_experiment_dir(experiment_id) / self.log_dir