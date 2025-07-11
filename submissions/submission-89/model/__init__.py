"""
Model components of the GMM transformer framework.

This package contains the fundamental building blocks and model implementations
for the transformer-based GMM framework.
"""

from .blocks import (
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock
)

from .transformer import (
    GMMTransformer,
    ClusterPredictionModel
)

from .factory import create_model_from_config

__all__ = [
    # Blocks
    'LayerNorm',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    
    # Models
    'GMMTransformer',
    'ClusterPredictionModel',
    
    # Factory functions
    'create_model_from_config'
]