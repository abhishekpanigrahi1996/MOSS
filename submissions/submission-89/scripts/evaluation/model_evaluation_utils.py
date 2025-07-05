"""
Utility functions for model evaluation.

This module provides functions for loading models from experiments and evaluating them.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader

from training.experiment import ExperimentManager
from utils.checkpointing import get_best_checkpoint, get_latest_checkpoint
from model.factory import create_model_from_config
from config.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


def load_model_from_experiment(
    experiment_dir: Union[str, Path], 
    load_best: bool = False, 
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, Any]:
    """
    Load model from the last checkpoint of an experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        load_best: Whether to load the best checkpoint instead of the latest one
        device: Device to load the model on
        
    Returns:
        The loaded model and its configuration
    """
    experiment_dir = Path(experiment_dir)
    
    try:
        # Determine device
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        # Find checkpoint directory
        checkpoint_dir = experiment_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Find checkpoint path
        if load_best:
            # Try to find best checkpoint first as "best_model.pt"
            best_path = checkpoint_dir / "best_model.pt"
            if best_path.exists():
                checkpoint_path = best_path
            else:
                # Try to find best checkpoint by metric
                checkpoint_path = get_best_checkpoint(checkpoint_dir)
                
            if checkpoint_path is None:
                raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
        else:
            # Try to find latest checkpoint first as "latest_model.pt"
            latest_path = checkpoint_dir / "latest_model.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
            else:
                # Try to find latest checkpoint by modification time
                checkpoint_path = get_latest_checkpoint(checkpoint_dir)
                
            if checkpoint_path is None:
                raise FileNotFoundError(f"No latest checkpoint found in {checkpoint_dir}")
                
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
        # Load config
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            config = ExperimentConfig.load(config_path)
        else:
            # Try to load config from checkpoint - don't need state_dict here
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'config' in checkpoint:
                config = ExperimentConfig.model_validate(checkpoint['config'])
            else:
                raise FileNotFoundError(f"No configuration found in experiment directory or checkpoint")
            
        # Create model
        model = create_model_from_config(
            config=config.model,
            device=device
        )
        
        # Load model weights - only need the state_dict here
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Ensure model is in evaluation mode
        model.eval()
        
        return model, config
        
    except FileNotFoundError as e:
        logger.error(f"Experiment directory or checkpoint not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from experiment: {e}")
        import traceback
        traceback.print_exc()
        raise