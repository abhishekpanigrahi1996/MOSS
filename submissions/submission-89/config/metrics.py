"""
Metrics configuration module for GMM transformer framework.

This module defines the configuration options for metrics computation.
"""

from typing import Dict, Any, List, Union
from pydantic import Field, model_validator

from .base import ConfigBase


class MetricsConfig(ConfigBase):
    """Configuration for metrics computation."""

    enabled: bool = Field(
        default=True,
        description="Whether to compute metrics"
    )

    metrics: List[Union[str, Dict[str, Any]]] = Field(
        default=["mse"],
        min_length=1,
        description="List of metrics to compute (strings or detailed dict configs)"
    )

    compare_with_kmeans: bool = Field(
        default=True,
        description="Whether to compare with K-means baseline"
    )

    compute_detailed_metrics: bool = Field(
        default=False,
        description="Whether to compute detailed metrics (e.g., time, stats)"
    )

    metrics_cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of metrics cache (0 to disable)"
    )

    @model_validator(mode='after')
    def validate_metrics(self) -> "MetricsConfig":
        """Validate metrics list."""
        valid_base_metrics = {"mse", "wasserstein", "energy", "assignment_accuracy", "nmi", "ari"}

        for metric in self.metrics:
            if isinstance(metric, str):
                # Extract base name (for cases like 'wasserstein_jax')
                base_name = metric.split('_')[0] if '_' in metric else metric
                if base_name not in valid_base_metrics:
                    raise ValueError(
                        f"Invalid metric: {metric}. Base type must be one of {valid_base_metrics}"
                    )
            elif isinstance(metric, dict):
                if 'type' not in metric:
                    raise ValueError("Metric configuration dictionary must have a 'type' key")

                base_type = metric['type']
                if base_type not in valid_base_metrics:
                    raise ValueError(
                        f"Invalid metric type: {base_type}. Must be one of {valid_base_metrics}"
                    )
            else:
                raise ValueError(f"Metric must be a string or dictionary, got {type(metric)}")

        return self

