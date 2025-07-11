"""
Model configuration classes for GMM transformer framework.

This module defines the configuration options for model architecture.
"""

from typing import Literal, List, Dict, Any, Optional, Union
from pydantic import Field, model_validator

from .base import ConfigBase


ActivationType = Literal["gelu", "relu", "silu", "mish"]
AttentionType = Literal["standard", "random_feature"]


class AttentionConfig(ConfigBase):
    """Configuration for attention mechanism."""
    
    type: AttentionType = Field(
        default="standard",
        description="Type of attention mechanism to use"
    )
    
    # Standard attention options
    use_flash_attn: bool = Field(
        default=True,
        description="Whether to use flash attention when available (only for standard attention)"
    )
    
    # Random feature attention options
    num_features: int = Field(
        default=64,
        ge=1,
        description="Number of random features per head for random feature attention"
    )
    
    feature_eps: float = Field(
        default=1e-6,
        gt=0.0,
        description="Epsilon for numerical stability in random feature attention"
    )


class FlowConfig(ConfigBase):
    """Configuration for flow predictor."""
    
    enabled: bool = Field(
        default=False,
        description="Whether to use flow predictor"
    )
    
    predictor_type: Literal["dummy", "linear", "monotonic"] = Field(
        default="monotonic",
        description="Type of flow predictor"
    )
    
    per_layer: bool = Field(
        default=True,
        description="Whether to use per-layer flow speeds"
    )
    
    distribution_mode: Literal["direct", "fractional"] = Field(
        default="direct",
        description="Mode for distributing flow speed across repetitions"
    )
    
    min_value: float = Field(
        default=0.0,
        description="Minimum flow speed value"
    )
    
    max_value: float = Field(
        default=1.0,
        description="Maximum flow speed value"
    )
    
    num_basis: int = Field(
        default=8,
        ge=1,
        description="Number of basis functions for flow predictor"
    )
    
    min_snr: float = Field(
        default=0.0,
        description="Minimum SNR mapping value (corresponds to max_flow)"
    )
    
    max_snr: float = Field(
        default=20.0,
        description="Maximum SNR mapping value (corresponds to min_flow)"
    )
    
    load_pretrained: bool = Field(
        default=False,
        description="Whether to load a pre-trained flow predictor"
    )
    
    pretrained_path: Optional[str] = Field(
        default=None,
        description="Path to pre-trained model containing flow predictor"
    )
    
    freeze_weights: bool = Field(
        default=False,
        description="Whether to freeze flow predictor weights during training"
    )


class TransformerConfig(ConfigBase):
    """Configuration for transformer architecture."""
    
    hidden_dim: int = Field(
        default=256,
        ge=16,
        description="Hidden dimension of the model"
    )
    
    num_layers: int = Field(
        default=6,
        ge=1,
        description="Number of transformer layers"
    )
    
    num_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads"
    )
    
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Dropout probability for transformer layers"
    )
    
    ff_expansion: int = Field(
        default=4,
        ge=1,
        description="Feed-forward expansion factor"
    )
    
    activation: ActivationType = Field(
        default="gelu",
        description="Activation function for feed-forward network"
    )
    
    norm_eps: float = Field(
        default=1e-5,
        gt=0.0,
        description="Epsilon for layer normalization"
    )
    
    # Attention configuration
    attention_config: AttentionConfig = Field(
        default_factory=AttentionConfig,
        description="Attention mechanism configuration"
    )
    
    # Flow speed configuration
    flow_config: FlowConfig = Field(
        default_factory=FlowConfig,
        description="Flow speed prediction configuration"
    )
    
    # Layer repetition configuration
    layer_repeat_mode: str = Field(
        default="none",
        description="Mode for layer repetition ('none', 'cycle', 'layerwise', 'grouped')"
    )
    
    repeat_factor: int = Field(
        default=1,
        ge=1,
        description="Number of times to repeat layers in 'cycle' or 'layerwise' mode. For random feature attention, also determines the number of different random feature matrices."
    )
    
    layer_groups: Optional[List[int]] = Field(
        default=None,
        description="List of layer group sizes for 'grouped' repetition mode"
    )
    
    group_repeat_factors: Optional[List[int]] = Field(
        default=None,
        description="List of repeat factors for each group in 'grouped' mode"
    )
    
    @model_validator(mode='after')
    def validate_model_dimensions(self) -> "TransformerConfig":
        """Validate model dimensions are compatible."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        # Validate layer repetition settings
        if self.layer_repeat_mode == "grouped":
            if self.layer_groups is None or self.group_repeat_factors is None:
                raise ValueError("layer_groups and group_repeat_factors must be provided for grouped mode")
                
            if len(self.layer_groups) != len(self.group_repeat_factors):
                raise ValueError("layer_groups and group_repeat_factors must have the same length")
                
            if sum(self.layer_groups) != self.num_layers:
                raise ValueError(f"Sum of layer_groups {sum(self.layer_groups)} must match num_layers {self.num_layers}")
        
        return self


class ClusterPredictionConfig(ConfigBase):
    """Configuration for cluster prediction model."""
    
    # Model-level parameters
    input_dim: int = Field(
        default=2,
        ge=1,
        description="Dimension of input features (point coordinates)"
    )
    
    bias: bool = Field(
        default=False,
        description="Whether to use bias in input projection and output head"
    )
    
    norm_eps: float = Field(
        default=1e-5,
        gt=0.0,
        description="Epsilon for layer normalization"
    )
    
    use_orthogonal_encdec: bool = Field(
        default=True,
        description="Whether to use orthogonal encoder-decoder architecture instead of standard linear encoder-decoder"
    )
    
    # Transformer configuration
    transformer: TransformerConfig = Field(
        default_factory=TransformerConfig,
        description="Transformer model configuration"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        config = self.model_dump()
        
        # Flatten transformer config into top level
        transformer_config = config.pop("transformer")
        config.update(transformer_config)
        
        return config
        
    def calculate_effective_layers(self) -> int:
        """Calculate the effective number of layers after applying repetition."""
        transformer = self.transformer
        
        if transformer.layer_repeat_mode == "none":
            return transformer.num_layers
        elif transformer.layer_repeat_mode in ["cycle", "layerwise"]:
            return transformer.num_layers * transformer.repeat_factor
        elif transformer.layer_repeat_mode == "grouped":
            if transformer.layer_groups is None or transformer.group_repeat_factors is None:
                raise ValueError("layer_groups and group_repeat_factors must be provided for grouped mode")
            return sum(size * rep for size, rep in zip(transformer.layer_groups, transformer.group_repeat_factors))
        else:
            raise ValueError(f"Unknown layer_repeat_mode: {transformer.layer_repeat_mode}")