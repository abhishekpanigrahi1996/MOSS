# Model Factory Refactoring Implementation Plan

## Overview

This document outlines the implementation plan for refactoring the model creation functionality in the GMM-v2 framework. The goal is to move the model creation from the `ExperimentManager._create_model` method to a dedicated factory function in the core module, ensuring consistent model creation across the codebase and better support for all configuration parameters.

## Current Issues

1. **Duplicate Model Creation Logic**: The same model creation logic exists in both `ExperimentManager._create_model` and `ModelManager.from_checkpoint`.
2. **Incomplete Parameter Support**: The current implementation doesn't use all available configuration fields.
3. **Tight Coupling**: Model creation is tightly coupled to the experiment manager, making it harder to use in other contexts.
4. **Inconsistency**: Model creation differs slightly between different parts of the codebase.

## Implementation Plan

### 1. Create Model Factory Function in Core

Create a new file `core/factory.py` with a factory function that centralizes model creation:

```python
"""
Factory functions for creating models from configuration.

This module provides functions for creating model instances from configuration
objects or dictionaries, ensuring consistent model creation across the codebase.
"""

import logging
from typing import Dict, Any, Optional, Union, Type

import torch
import torch.nn as nn

from config.base import ModelConfig
from core import GMMTransformer, ClusterPredictionModel

logger = logging.getLogger(__name__)

def create_model_from_config(
    config: Union[ModelConfig, Dict[str, Any]],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a model instance from configuration.
    
    Args:
        config: Model configuration object or dictionary
        device: Optional device to place the model on
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If the prediction type is not supported
    """
    # Convert dictionary to config object if needed
    if isinstance(config, dict):
        from config.base import ModelConfig
        config = ModelConfig.model_validate(config)
    
    # Extract configuration parameters
    prediction_type = config.prediction_type
    
    # Create model based on prediction type
    if prediction_type == "centers":
        model = ClusterPredictionModel(
            input_dim=config.transformer.input_dim,
            hidden_dim=config.transformer.hidden_dim,
            num_layers=config.transformer.num_layers,
            num_heads=config.transformer.num_heads,
            prediction_type="centers",
            dropout=config.transformer.dropout,
            dropout_attention=getattr(config.transformer, 'dropout_attention', config.transformer.dropout),
            dropout_residual=getattr(config.transformer, 'dropout_residual', config.transformer.dropout),
            dropout_embedding=getattr(config.transformer, 'dropout_embedding', config.transformer.dropout),
            activation=config.transformer.activation,
            ff_expansion=config.transformer.ff_expansion,
            bias=config.transformer.bias,
            norm_eps=config.transformer.norm_eps,
            use_flash_attn=config.transformer.use_flash_attn,
            layer_repetition=getattr(config.transformer, 'layer_repetition', 1),
            flow_speed=getattr(config.transformer, 'flow_speed', False)
        )
    elif prediction_type == "assignments":
        model = ClusterPredictionModel(
            input_dim=config.transformer.input_dim,
            hidden_dim=config.transformer.hidden_dim,
            num_layers=config.transformer.num_layers,
            num_heads=config.transformer.num_heads,
            prediction_type="assignments",
            num_clusters=config.num_clusters,
            dropout=config.transformer.dropout,
            dropout_attention=getattr(config.transformer, 'dropout_attention', config.transformer.dropout),
            dropout_residual=getattr(config.transformer, 'dropout_residual', config.transformer.dropout),
            dropout_embedding=getattr(config.transformer, 'dropout_embedding', config.transformer.dropout),
            activation=config.transformer.activation,
            ff_expansion=config.transformer.ff_expansion,
            bias=config.transformer.bias,
            norm_eps=config.transformer.norm_eps,
            use_flash_attn=config.transformer.use_flash_attn,
            layer_repetition=getattr(config.transformer, 'layer_repetition', 1),
            flow_speed=getattr(config.transformer, 'flow_speed', False)
        )
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    # Log model creation
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {prediction_type} prediction model with {num_params:,} parameters")
    
    return model
```

### 2. Update Core Package Exports

Update `core/__init__.py` to export the new factory function:

```python
from .factory import create_model_from_config

__all__ = [
    # ... existing exports ...
    
    # Factory functions
    'create_model_from_config'
]
```

### 3. Modify ExperimentManager

Update the `ExperimentManager._create_model` method to use the factory function:

```python
def _create_model(self) -> torch.nn.Module:
    """
    Create model based on configuration.
    
    Returns:
        Model instance
    """
    from core.factory import create_model_from_config
    
    # Create model using factory function
    device = self.config.device.get_device()
    model = create_model_from_config(
        config=self.config.model,
        device=device
    )
    
    return model
```

### 4. Update ModelManager

Modify the `ModelManager.from_checkpoint` method to use the factory function:

```python
@classmethod
def from_checkpoint(
    cls, 
    checkpoint_path: Union[str, Path], 
    device: Optional[Union[str, torch.device]] = None
) -> "ModelManager":
    """
    Load a model directly from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        A ModelManager instance with the loaded model
    """
    # ... existing code for loading checkpoint ...
    
    # Create model using factory function
    from core.factory import create_model_from_config
    
    model = create_model_from_config(
        config=config.model,
        device=device
    )
    
    # Load weights and move to device
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode by default
    
    # ... rest of the method ...
```

### 5. Add Model Registration System (Optional Extension)

For more flexibility, we can add a model registration system that allows for custom model types:

```python
# In core/factory.py, add:

_MODEL_REGISTRY = {}

def register_model(model_type: str, model_class: Type[nn.Module]) -> None:
    """
    Register a model class for a given model type.
    
    Args:
        model_type: The type identifier for the model
        model_class: The model class to register
    """
    _MODEL_REGISTRY[model_type] = model_class
    logger.info(f"Registered model type '{model_type}' with class {model_class.__name__}")

def create_model_from_config(
    config: Union[ModelConfig, Dict[str, Any]],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a model instance from configuration.
    
    Args:
        config: Model configuration object or dictionary
        device: Optional device to place the model on
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If the model type is not supported
    """
    # Convert dictionary to config object if needed
    if isinstance(config, dict):
        from config.base import ModelConfig
        config = ModelConfig.model_validate(config)
    
    # Check if we have a registered custom model class
    model_type = getattr(config, 'model_type', config.prediction_type)
    
    if model_type in _MODEL_REGISTRY:
        # Use registered custom model class
        model_class = _MODEL_REGISTRY[model_type]
        model = model_class(**config.model_dump())
    else:
        # Use default model creation logic
        # ... existing model creation code ...
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    # Log model creation
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_type} model with {num_params:,} parameters")
    
    return model

# Register default model types
register_model("centers", ClusterPredictionModel)
register_model("assignments", ClusterPredictionModel)
```

## Implementation Steps

1. **Create the Factory Module**:
   - Create `core/factory.py` with the core factory function
   - Update `core/__init__.py` to expose the new function

2. **Update ExperimentManager**:
   - Modify `ExperimentManager._create_model` to use the factory function
   - Remove duplicate code

3. **Update ModelManager**:
   - Modify `ModelManager.from_checkpoint` to use the factory function
   - Ensure consistent parameter handling

4. **Add Tests**:
   - Create tests for model factory in `tests/core/test_model_factory.py`
   - Test all supported model configurations
   - Test with edge cases and invalid configurations

5. **Update Documentation**:
   - Update model creation documentation
   - Add examples for custom model registration (if implemented)

## Testing Strategy

Tests should verify:

1. Models created with factory have the correct architecture
2. All configuration parameters are properly passed to models
3. Feature flags (layer_repetition, flow_speed) are correctly honored
4. Model registration works correctly (if implemented)
5. Error handling for invalid configurations

Example test cases:

```python
def test_create_centers_model():
    config = ModelConfig(
        prediction_type="centers",
        transformer=TransformerConfig(
            input_dim=2,
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            layer_repetition=2  # Test layer repetition feature
        )
    )
    
    model = create_model_from_config(config)
    
    assert isinstance(model, ClusterPredictionModel)
    assert model.prediction_type == "centers"
    assert model.transformer.hidden_dim == 64
    assert model.transformer.num_layers == 4
    assert model.transformer.layer_repetition == 2
```

## Benefits

The proposed refactoring will:

1. **Eliminate Code Duplication**: Single source of truth for model creation
2. **Improve Maintainability**: Easier to add new model parameters or types
3. **Enhance Consistency**: Same creation logic used throughout the codebase
4. **Support Future Extensions**: Registration system allows adding custom models
5. **Better Parameter Support**: All configuration options will be used correctly