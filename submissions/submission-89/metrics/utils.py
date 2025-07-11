"""
Shared utility functions for metrics and losses.

This module provides common utility functions used across the metrics system,
such as device management, tensor conversion, and batch size extraction.
"""

import logging
import torch
from typing import Any, Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def to_device(data: Any, device: torch.device) -> Any:
    """
    Move data to specified device, handling different data types appropriately.
    
    Args:
        data: Input data (tensor, dict of tensors, or other)
        device: Target device for tensor data
        
    Returns:
        Data with tensors moved to specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in data):
        return [x.to(device) for x in data]
    return data


def extract_batch_size(data: Any) -> int:
    """
    Extract batch size from various data formats.
    
    Args:
        data: Input data (tensor, dict, tuple, etc.)
        
    Returns:
        Batch size or 0 if not determinable
    """
    if isinstance(data, torch.Tensor):
        return data.size(0)
    elif isinstance(data, dict) and any(isinstance(v, torch.Tensor) for v in data.values()):
        # Return size from first tensor found
        for v in data.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if isinstance(data[0], torch.Tensor):
            return data[0].size(0)
    
    return 0


def extract_n_clusters(targets: Any) -> int:
    """
    Extract number of clusters from targets data in various formats.
    
    Args:
        targets: Target data in any format
        
    Returns:
        Number of clusters or 5 as a default
    """
    try:
        if isinstance(targets, dict) and "centers" in targets:
            return targets["centers"].shape[1]
        elif isinstance(targets, (list, tuple)) and len(targets) >= 1:
            if isinstance(targets[0], torch.Tensor):
                return targets[0].shape[1]
        elif isinstance(targets, torch.Tensor) and len(targets.shape) > 1:
            return targets.shape[1]
    except Exception as e:
        logger.warning(f"Failed to determine number of clusters: {e}")
    
    # Default fallback
    return 5


def tensor_to_numpy(tensor: Any) -> Any:
    """
    Convert PyTorch tensors to NumPy arrays, handling different data types.
    
    Args:
        tensor: Input tensor or structure containing tensors
        
    Returns:
        NumPy array or structure with tensors converted to NumPy
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, dict):
        return {k: tensor_to_numpy(v) for k, v in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [tensor_to_numpy(x) for x in tensor]
    return tensor