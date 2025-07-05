"""
Metrics package for GMM transformer models.

This package provides unified tools for creating, using, and tracking metrics
for model evaluation. It combines loss functions and metrics in a single coherent interface.
"""

# Public exports for backward compatibility
from .factory import create_loss_from_config, create_metric_functions
from .tracker import MetricsTracker, compute_metrics