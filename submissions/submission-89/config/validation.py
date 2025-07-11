"""
Validation configuration for GMM transformer framework.

This module defines the configuration settings for validation and evaluation.
"""

from typing import Optional
from pydantic import Field

from .base import ConfigBase
from .metrics import MetricsConfig
from .logging import VisualizationConfig


class ValidationConfig(ConfigBase):
    """Configuration for validation."""

    validation_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for validation"
    )

    num_val_batches: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of batches to use for validation (None for all)"
    )

    num_val_samples: int = Field(
        default=1000,
        ge=1,
        description="Number of validation samples per epoch (for dynamic data)"
    )

    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration"
    )

    visualize: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration"
    )
    
    fixed_validation_data: bool = Field(
        default=False,
        description="If True, use a fixed validation dataset generated once for all validation runs"
    )

