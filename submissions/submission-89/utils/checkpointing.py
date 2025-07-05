"""
Checkpoint utilities for GMM transformer models.

This module provides functions for saving and loading model checkpoints.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint: Dict[str, Any],
    path: Union[str, Path],
    save_optimizer: bool = True,
    save_scheduler: bool = True
) -> None:
    """
    Save model checkpoint to file.
    
    Args:
        checkpoint: Dictionary containing state to save
        path: Path to save checkpoint to
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    
    # Create filtered checkpoint if requested
    if not save_optimizer or not save_scheduler:
        filtered_checkpoint = checkpoint.copy()
        
        if not save_optimizer and "optimizer_state_dict" in filtered_checkpoint:
            del filtered_checkpoint["optimizer_state_dict"]
            
        if not save_scheduler and "scheduler_state_dict" in filtered_checkpoint:
            del filtered_checkpoint["scheduler_state_dict"]
            
        checkpoint = filtered_checkpoint
    
    try:
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")
        raise


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint from file.
    
    Args:
        path: Path to load checkpoint from
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        
    Returns:
        Loaded checkpoint dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        # Map to device if specified
        if device is None:
            checkpoint = torch.load(path)
        else:
            if isinstance(device, str):
                device = torch.device(device)
            checkpoint = torch.load(path, map_location=device)
        
        logger.debug(f"Checkpoint loaded from {path}")
        
        # Filter checkpoint if requested
        if not load_optimizer and "optimizer_state_dict" in checkpoint:
            del checkpoint["optimizer_state_dict"]
            
        if not load_scheduler and "scheduler_state_dict" in checkpoint:
            del checkpoint["scheduler_state_dict"]
        
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {e}")
        raise


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    prefix: str = "",
    suffix: str = ".pt"
) -> Optional[Path]:
    """
    Get the latest checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Optional filename prefix to filter by
        suffix: Optional filename suffix to filter by
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Get all checkpoint files
    checkpoints = [
        f for f in checkpoint_dir.iterdir()
        if f.is_file() and f.name.startswith(prefix) and f.name.endswith(suffix)
    ]
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return checkpoints[0]


def get_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric_name: str = "val_loss",
    lower_is_better: bool = True
) -> Optional[Path]:
    """
    Get the checkpoint with the best metric value.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Name of metric to compare
        lower_is_better: Whether lower metric values are better
        
    Returns:
        Path to best checkpoint, or None if no valid checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Get all checkpoint files
    checkpoints = [
        f for f in checkpoint_dir.iterdir()
        if f.is_file() and f.name.endswith(".pt")
    ]
    
    if not checkpoints:
        return None
    
    # Find checkpoint with best metric
    best_checkpoint = None
    best_metric = float('inf') if lower_is_better else float('-inf')
    
    for checkpoint_path in checkpoints:
        try:
            # Load checkpoint metadata only
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu"
            )
            
            # Check if metric exists in checkpoint
            if metric_name in checkpoint:
                metric_value = checkpoint[metric_name]
                
                # Update best checkpoint if metric is better
                if ((lower_is_better and metric_value < best_metric) or
                        (not lower_is_better and metric_value > best_metric)):
                    best_metric = metric_value
                    best_checkpoint = checkpoint_path
        except:
            # Skip invalid checkpoint files
            continue
    
    return best_checkpoint