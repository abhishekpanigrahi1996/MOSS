"""
K-means baseline utilities for GMM model evaluation.

This module provides utilities for comparing model performance against
K-means clustering baselines, which serve as a standard reference point.
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict, Any

from .utils import extract_n_clusters

# Import sklearn for K-means baseline if available
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scikit-learn not available, K-means baseline in metrics will be disabled"
    )

logger = logging.getLogger(__name__)

def compute_kmeans_baseline(
    points: torch.Tensor, 
    n_clusters: int
) -> Optional[torch.Tensor]:
    """
    Compute K-means clustering and map points to their cluster centers.
    
    Args:
        points: Input points tensor of shape [batch_size, n_points, dim]
        n_clusters: Number of clusters to compute
        
    Returns:
        Tensor with same shape as points, where each point is replaced by its cluster center.
        Returns None if scikit-learn is not available.
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    # Convert to numpy for scikit-learn
    points_np = points.detach().cpu().numpy()
    batch_size, n_points, dim = points_np.shape
    
    # Initialize output with same shape as input
    output = np.zeros_like(points_np)
    
    # Process each batch separately
    for b in range(batch_size):
        batch_points = points_np[b].reshape(n_points, dim)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(batch_points)
        centers = kmeans.cluster_centers_
        
        # Map each point to its cluster center
        for i in range(n_points):
            output[b, i] = centers[labels[i]]
    
    # Convert back to PyTorch and match device
    return torch.tensor(output, dtype=points.dtype, device=points.device)


def compute_kmeans_metrics(
    points: torch.Tensor,
    targets: Any,
    metric_fns: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute metrics for K-means baseline.
    
    Args:
        points: Input points tensor
        targets: Ground truth targets
        metric_fns: Dictionary of metric functions
        
    Returns:
        Dictionary of computed metrics for K-means baseline
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
    
    # Extract number of clusters
    n_clusters = extract_n_clusters(targets)
    
    # Compute K-means baseline
    kmeans_predictions = compute_kmeans_baseline(points, n_clusters)
    
    if kmeans_predictions is None:
        return {"error": "Failed to compute K-means baseline"}
    
    # Compute metrics for K-means predictions
    result = {}
    for name, fn in metric_fns.items():
        try:
            value = fn(kmeans_predictions, targets)
            if isinstance(value, torch.Tensor):
                value = value.item()
            result[name] = value
        except Exception as e:
            logger.warning(f"Failed to compute {name} for K-means baseline: {e}")
    
    return result


def compute_silhouette_score(
    points: torch.Tensor,
    predictions: torch.Tensor
) -> Optional[float]:
    """
    Compute silhouette score for clustering evaluation.
    
    Args:
        points: Input points tensor [batch_size, n_points, dim]
        predictions: Predicted cluster assignments [batch_size, n_points]
        
    Returns:
        Silhouette score or None if scikit-learn is not available
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    try:
        # Convert to numpy
        points_np = points.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        
        # Compute score for first batch only (for simplicity)
        score = silhouette_score(
            points_np[0], pred_np[0], 
            metric='euclidean', 
            random_state=42
        )
        
        return score
    except Exception as e:
        logger.warning(f"Failed to compute silhouette score: {e}")
        return None