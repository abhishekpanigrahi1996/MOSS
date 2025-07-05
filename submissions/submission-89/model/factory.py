"""
Factory functions for creating models from configuration.

This module provides functions for creating model instances from configuration
objects or dictionaries, ensuring consistent model creation across the codebase.
"""

import logging
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

from config.model import ClusterPredictionConfig
from model import GMMTransformer, ClusterPredictionModel

logger = logging.getLogger(__name__)

def create_model_from_config(
    config: Union[ClusterPredictionConfig, Dict[str, Any]],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a model instance from configuration.
    
    Args:
        config: Model configuration object or dictionary
        device: Optional device to place the model on
        
    Returns:
        Instantiated model
    """
    # Convert dictionary to config object if needed
    if isinstance(config, dict):
        from config.model import ClusterPredictionConfig
        config = ClusterPredictionConfig.model_validate(config)
    
    # Create a single set of kwargs for the model
    model_kwargs = {
        # Model-level parameters
        "input_dim": config.input_dim,
        "hidden_dim": config.transformer.hidden_dim,
        "dropout": config.transformer.dropout,
        "bias": config.bias,
        "norm_eps": config.norm_eps,
        "use_orthogonal_encdec": getattr(config, 'use_orthogonal_encdec', True),
        
        # Transformer parameters
        "num_layers": config.transformer.num_layers,
        "num_heads": config.transformer.num_heads,
        "activation": config.transformer.activation,
        "ff_expansion": config.transformer.ff_expansion,
        
        # Attention parameters
        "use_flash_attn": config.transformer.attention_config.use_flash_attn,
        "use_random_features": config.transformer.attention_config.type == "random_feature",
        "num_random_features": config.transformer.attention_config.num_features,
        "random_feature_eps": config.transformer.attention_config.feature_eps,
        "num_repeats": config.transformer.repeat_factor  # Use the same repeat_factor for random features
    }
    
    # Add layer repetition parameters if they exist
    if hasattr(config.transformer, 'layer_repeat_mode'):
        model_kwargs["layer_repeat_mode"] = config.transformer.layer_repeat_mode
    if hasattr(config.transformer, 'repeat_factor'):
        model_kwargs["repeat_factor"] = config.transformer.repeat_factor
    if hasattr(config.transformer, 'layer_groups'):
        model_kwargs["layer_groups"] = config.transformer.layer_groups
    if hasattr(config.transformer, 'group_repeat_factors'):
        model_kwargs["group_repeat_factors"] = config.transformer.group_repeat_factors
    
    # Add flow speed parameters from flow_config
    flow_config = config.transformer.flow_config
    model_kwargs["use_flow_predictor"] = flow_config.enabled
    model_kwargs["flow_predictor_type"] = flow_config.predictor_type
    model_kwargs["flow_predictor_per_layer"] = flow_config.per_layer
    model_kwargs["flow_distribution_mode"] = flow_config.distribution_mode
    
    # Add monotonic flow predictor parameters
    model_kwargs["flow_num_basis"] = flow_config.num_basis
    model_kwargs["flow_min_value"] = flow_config.min_value
    model_kwargs["flow_max_value"] = flow_config.max_value
    model_kwargs["flow_min_snr"] = flow_config.min_snr
    model_kwargs["flow_max_snr"] = flow_config.max_snr
    
    # Add pre-trained flow predictor parameters
    model_kwargs["load_pretrained_flow"] = flow_config.load_pretrained
    model_kwargs["pretrained_flow_path"] = flow_config.pretrained_path
    model_kwargs["freeze_flow_weights"] = flow_config.freeze_weights
    
    # Any remaining parameters from model creation will be passed through kwargs
    
    # Create model with a single set of parameters
    model = ClusterPredictionModel(**model_kwargs)
    
    # Move model to device if specified
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device is not None:
        model = model.to(device)
    
    # Log model creation
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created cluster prediction model with {num_params:,} parameters")
    
    return model