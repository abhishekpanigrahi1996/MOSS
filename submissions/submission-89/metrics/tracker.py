"""
Metrics tracking and computation for model evaluation.

This module provides tools for computing, tracking, and accumulating metrics
during training and evaluation, with support for incremental updates.
"""

import logging
import torch
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set

from .utils import extract_batch_size, to_device
from .kmeans import compute_kmeans_baseline, SKLEARN_AVAILABLE
from .utils import extract_n_clusters

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Class for tracking and accumulating metrics across batches.
    
    Handles computation of metrics incrementally during validation,
    rather than saving all data and computing metrics at the end.
    """
    
    def __init__(
        self,
        metric_fns: Dict[str, Callable],
        compare_with_kmeans: bool = False,
        device: Optional[torch.device] = None,
        include_loss: bool = False
    ):
        """
        Initialize the metrics tracker.
        
        Args:
            metric_fns: Dictionary mapping metric names to metric functions
            compare_with_kmeans: Whether to compare with K-means baseline
            device: Device to use for computation (defaults to CPU)
            include_loss: Whether to track loss as a separate metric
        """
        self.metric_names = list(metric_fns.keys())
        self.metric_fns = metric_fns
        self.metrics = {name: 0.0 for name in self.metric_names}
        self.kmeans_metrics = {name: 0.0 for name in self.metric_names}
        self.sample_count = 0
        self.compare_with_kmeans = compare_with_kmeans
        self.device = device if device is not None else torch.device('cpu')
        self.include_loss = include_loss
        
        # For direct loss tracking
        if include_loss:
            self.loss_sum = 0.0
            
    def update_loss(self, loss_value: float, batch_size: int = 1):
        """
        Update loss metric for a batch.
        
        Args:
            loss_value: Loss value for this batch (assumed to be already a mean over the batch)
            batch_size: Batch size for weighted averaging
        """
        if self.include_loss:
            # loss_value is already a mean over the batch, so we just accumulate proportionally
            self.loss_sum += loss_value * (batch_size / 1.0)  # Weighting by batch proportion
            self.sample_count += batch_size
        
    def update(
        self,
        predictions: Any,
        targets: Any,
        points: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a new batch of data.
        
        Args:
            predictions: Model predictions (any format)
            targets: Ground truth targets (any format)
            points: Input points for K-means baseline
        """
        # Get batch size from predictions
        batch_size = extract_batch_size(predictions)
        if batch_size == 0:
            return
            
        # Update model metrics
        self._update_metrics(self.metrics, predictions, targets, batch_size)
        
        # Update K-means metrics if requested
        if self.compare_with_kmeans and SKLEARN_AVAILABLE and points is not None:
            n_clusters = extract_n_clusters(targets)
            kmeans_predictions = compute_kmeans_baseline(points, n_clusters or 5)
            
            if kmeans_predictions is not None:
                self._update_metrics(self.kmeans_metrics, kmeans_predictions, targets, batch_size)
        
        # Update sample count if not already updated through update_loss
        if not (self.include_loss and self.sample_count > 0):
            self.sample_count += batch_size
    
    def _update_metrics(
        self, 
        metrics_dict: Dict[str, float], 
        predictions: Any, 
        targets: Any, 
        batch_size: int
    ):
        """
        Update a specific metrics dictionary with computed values.
        
        This method handles metric functions that return mean values over the batch
        (as is standard for most loss functions). It properly accounts for this by
        accumulating values with appropriate batch size weighting.
        
        Args:
            metrics_dict: Dictionary to update
            predictions: Model predictions
            targets: Ground truth targets
            batch_size: Batch size for weighted averaging
        """
        for name, fn in self.metric_fns.items():
            try:
                # Compute metric - our loss functions return means over the batch
                value = fn(predictions, targets)
                
                # Convert to scalar if tensor
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                # Simply accumulate the mean value weighted by batch size proportion 
                # (since loss functions already return means)
                metrics_dict[name] += value * (batch_size / 1.0)  # Weighting by batch proportion
            except Exception as e:
                logger.warning(f"Failed to compute metric {name}: {e}")
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute final metrics and ratios as weighted averages.
        
        Since loss functions already return mean values per batch, this 
        method computes a weighted average across all batches, where each
        batch contributes proportionally to its size.
        
        Returns:
            Dictionary of computed metrics as weighted averages
        """
        if self.sample_count == 0:
            return {"error": "No samples processed"}
            
        # Compute average metrics
        result = {}
        
        # Add loss metric if tracked
        if self.include_loss and hasattr(self, 'loss_sum'):
            result["loss"] = self.loss_sum / self.sample_count
            
        # Add regular metrics
        for name in self.metric_names:
            if name in self.metrics:
                result[name] = self.metrics[name] / self.sample_count
                
        # Add K-means metrics
        if self.compare_with_kmeans:
            # Add flat format metrics
            for name in self.metric_names:
                if name in self.kmeans_metrics:
                    result[f"kmeans_{name}"] = self.kmeans_metrics[name] / self.sample_count
                    
            # Add nested format for backward compatibility
            kmeans_dict = {}
            for name in self.metric_names:
                if name in self.kmeans_metrics:
                    kmeans_dict[name] = self.kmeans_metrics[name] / self.sample_count
            result["kmeans"] = kmeans_dict
            
            # Compute improvement ratios
            for name in self.metric_names:
                if name in self.metrics and name in self.kmeans_metrics:
                    model_value = self.metrics[name] / self.sample_count
                    kmeans_value = self.kmeans_metrics[name] / self.sample_count
                    
                    # Determine if lower or higher values are better
                    lower_is_better = any(x in name.lower() for x in 
                                         ["mse", "wasserstein", "energy", "loss", "distance"])
                    
                    # Calculate ratio (avoid division by zero)
                    if abs(kmeans_value) > 1e-8 and lower_is_better:
                        ratio = model_value / kmeans_value
                        result[f"{name}_ratio"] = ratio
                    elif abs(model_value) > 1e-8 and not lower_is_better:
                        ratio = kmeans_value / model_value
                        result[f"{name}_ratio"] = ratio
                            
        return result
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {name: 0.0 for name in self.metric_names}
        self.kmeans_metrics = {name: 0.0 for name in self.metric_names}
        self.sample_count = 0
        
        if self.include_loss:
            self.loss_sum = 0.0


def compute_metrics(
    predictions: Any,
    targets: Any,
    points: Optional[Any] = None,
    metric_names: Optional[List[str]] = None,
    metric_fns: Optional[Dict[str, Callable]] = None,
    compare_with_kmeans: bool = False,
    with_details: bool = False
) -> Dict[str, Any]:
    """
    Compute metrics for model evaluation.
    
    Args:
        predictions: Model predictions in any format
        targets: Ground truth targets in any format
        points: Input points for K-means baseline
        metric_names: List of metric names to compute
        metric_fns: Dictionary mapping metric names to callable functions
        compare_with_kmeans: Whether to compare with K-means baseline
        with_details: Whether to include detailed metrics
        
    Returns:
        Dictionary of computed metrics
    """
    if metric_names is None or not metric_names:
        logger.warning("No metrics specified, using default 'mse'")
        metric_names = ["mse"]
        
    if metric_fns is None:
        logger.warning("No metric functions provided, metrics computation will be limited")
        return {"error": "No metric functions provided"}
    
    # Get computation device
    device = predictions.device if isinstance(predictions, torch.Tensor) else None
    
    # Create metrics tracker
    tracker = MetricsTracker(
        metric_fns=metric_fns,
        compare_with_kmeans=compare_with_kmeans,
        device=device
    )
    
    # Update with single batch
    tracker.update(predictions, targets, points)
    
    # Compute final metrics
    result = tracker.compute()
    
    # Add timing information if details requested
    if with_details:
        result["_details_available"] = True
        
    return result