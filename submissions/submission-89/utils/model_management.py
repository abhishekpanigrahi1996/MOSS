"""
Model management utilities for GMM transformer models.

This module provides high-level interfaces for model loading, saving, 
evaluation, and visualization, with support for data state management
for reproducible and resumable experimentation.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.experiment import ExperimentConfig
from model import GMMTransformer, ClusterPredictionModel
from training.experiment import ExperimentManager
from metrics import MetricsTracker, create_metric_functions
from utils.checkpointing import load_checkpoint, get_latest_checkpoint, get_best_checkpoint
# Import visualization utilities
# Note: visualize_gmm_distributions doesn't exist, using other functions from visualization.py instead

logger = logging.getLogger(__name__)


class ModelManager:
    """
    High-level interface for model management operations.
    
    This class provides methods for saving, loading, evaluating, and
    visualizing models across different experiments.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Optional[ExperimentConfig] = None,
        experiment_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize a model manager.
        
        Args:
            model: The model to manage
            config: The model configuration
            experiment_dir: Optional directory for experiment outputs
        """
        self.model = model
        self.config = config
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        # Set device
        self.device = None
        if config and hasattr(config, 'device'):
            self.device = config.device.get_device()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        # Note: model.device is not available directly, we need to check parameters
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            self.model.to(self.device)
    
    @classmethod
    def from_experiment(cls, experiment_manager: ExperimentManager) -> "ModelManager":
        """
        Create a ModelManager from an ExperimentManager.
        
        Args:
            experiment_manager: The experiment manager to extract model from
            
        Returns:
            A ModelManager instance
        """
        if not hasattr(experiment_manager, 'trainer') or not experiment_manager.trainer:
            raise ValueError("Experiment does not have a trainer with model")
            
        model = experiment_manager.trainer.model
        config = experiment_manager.config
        experiment_dir = experiment_manager.experiment_dir
        
        return cls(model=model, config=config, experiment_dir=experiment_dir)
    
    @classmethod
    def from_checkpoint(
        cls, 
        checkpoint_path: Union[str, Path], 
        device: Optional[Union[str, torch.device]] = None
    ) -> "ModelManager":
        """
        Load a model directly from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model to
            
        Returns:
            A ModelManager instance with the loaded model
        """
        # Determine device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        # Load checkpoint
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
        
        # Extract configuration
        if 'config' in checkpoint:
            config = ExperimentConfig.model_validate(checkpoint['config'])
        else:
            logger.warning("No configuration found in checkpoint, using default")
            config = ExperimentConfig()
            
        # Create model using factory function
        from model.factory import create_model_from_config
        
        model = create_model_from_config(
            config=config.model,
            device=device
        )
            
        # Load weights and move to device
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Set to evaluation mode by default
        
        # Get experiment directory if can be determined
        checkpoint_path = Path(checkpoint_path)
        experiment_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else None
        
        return cls(model=model, config=config, experiment_dir=experiment_dir)
    
    @classmethod
    def from_experiment_dir(
        cls, 
        experiment_dir: Union[str, Path],
        load_best: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ) -> "ModelManager":
        """
        Load a model from an experiment directory.
        
        Args:
            experiment_dir: Path to the experiment directory
            load_best: Whether to load the best model (True) or latest (False)
            device: Device to load the model to
            
        Returns:
            A ModelManager instance
        """
        experiment_dir = Path(experiment_dir)
        checkpoint_dir = experiment_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Find the checkpoint to load
        if load_best:
            checkpoint_path = get_best_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                # Try looking for best_model.pt
                best_path = checkpoint_dir / "best_model.pt"
                if best_path.exists():
                    checkpoint_path = best_path
                else:
                    logger.warning("Best checkpoint not found, falling back to latest")
                    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        else:
            # Try looking for latest_model.pt first
            latest_path = checkpoint_dir / "latest_model.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
            else:
                checkpoint_path = get_latest_checkpoint(checkpoint_dir)
                
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
            
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        return cls.from_checkpoint(checkpoint_path, device)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file using PyTorch's save mechanism.
        
        Args:
            path: Path to save model to
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Create simple state dict with just the model weights
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def evaluate(
        self, 
        data_loader: Optional[DataLoader] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on data.
        
        Args:
            data_loader: DataLoader for evaluation data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if data_loader is None:
            raise ValueError("No data loader provided for evaluation")
        
        # Use validation metrics from config if available and no metrics specified
        if metrics is None and self.config and hasattr(self.config.validation, 'metrics'):
            metrics = self.config.validation.metrics.metrics
        
        # Default metrics if still None
        if metrics is None:
            metrics = ['mse', 'wasserstein']
        
        # Create metrics tracker
        metric_fns = create_metric_functions(metrics)
        compare_with_kmeans = self.config.validation.metrics.compare_with_kmeans if \
            (self.config and hasattr(self.config.validation, 'metrics')) else False
            
        tracker = MetricsTracker(
            metric_fns=metric_fns,
            compare_with_kmeans=compare_with_kmeans,
            device=self.device,
            include_loss=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create loss function if available in config
        loss_fn = None
        if self.config and hasattr(self.config.training, 'loss'):
            try:
                from metrics import create_loss_from_config
                loss_fn = create_loss_from_config(self.config.training.loss.loss_type)
            except:
                logger.warning("Could not create loss function from config")
                loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.MSELoss()
        
        # Evaluation loop
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = batch.get('points', batch.get('X', batch))
                    targets = batch
                else:
                    inputs = batch[0]
                    targets = batch[1]
                    
                # Move inputs to device
                inputs = inputs.to(self.device)
                
                # Handle targets based on type
                if isinstance(targets, dict):
                    for key, value in list(targets.items()):
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                elif isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = loss_fn(outputs, targets)
                batch_size = inputs.size(0)
                
                # Update metrics
                tracker.update_loss(loss.item(), batch_size)
                tracker.update(outputs, targets, inputs)
        
        # Compute metrics
        metrics = tracker.compute()
        
        return metrics
    
    def visualize(
        self, 
        data_loader: Optional[DataLoader] = None,
        num_samples: int = 5,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[plt.Figure]:
        """
        Generate visualizations of model predictions.
        
        Args:
            data_loader: DataLoader for visualization data
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations to
            
        Returns:
            List of matplotlib figures
        """
        if data_loader is None:
            raise ValueError("No data loader provided for visualization")
            
        output_dir = Path(output_dir) if output_dir else \
            (self.experiment_dir / "visualizations" if self.experiment_dir else Path("./visualizations"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get samples from data loader
        all_batches = []
        for batch in data_loader:
            all_batches.append(batch)
            if len(all_batches) >= num_samples:
                break
                
        if not all_batches:
            raise ValueError("No batches found in data loader")
            
        # If fewer batches than requested, use what we have
        actual_num_samples = min(num_samples, len(all_batches))
        
        # Generate visualizations
        figures = []
        with torch.no_grad():
            for i, batch in enumerate(all_batches[:actual_num_samples]):
                # Extract inputs and targets
                if isinstance(batch, dict):
                    inputs = batch.get('points', batch.get('X', batch))
                    targets = batch.get('Y', batch.get('centers', None))
                else:
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                
                # Move to device
                inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Convert to numpy for visualization
                input_np = inputs.cpu().numpy()
                output_np = outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else \
                           outputs['centers'].cpu().numpy() if isinstance(outputs, dict) and 'centers' in outputs else None
                target_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else \
                           targets['centers'].cpu().numpy() if isinstance(targets, dict) and 'centers' in targets else None
                
                # Create visualization
                fig = plt.figure(figsize=(15, 5))
                
                # Plot input distribution
                ax1 = fig.add_subplot(131)
                if input_np.shape[-1] == 1:  # 1D data
                    ax1.scatter(input_np[0, :, 0], np.zeros_like(input_np[0, :, 0]), alpha=0.5, label='Input')
                else:  # 2D data
                    ax1.scatter(input_np[0, :, 0], input_np[0, :, 1], alpha=0.5, label='Input')
                ax1.set_title('Input Distribution')
                ax1.legend()
                
                # Plot target distribution if available
                ax2 = fig.add_subplot(132)
                if target_np is not None:
                    if target_np.shape[-1] == 1:  # 1D data
                        ax2.scatter(target_np[0, :, 0], np.zeros_like(target_np[0, :, 0]), 
                                   alpha=0.5, color='green', label='Target')
                    else:  # 2D data
                        ax2.scatter(target_np[0, :, 0], target_np[0, :, 1], 
                                   alpha=0.5, color='green', label='Target')
                    ax2.set_title('Target Distribution')
                    ax2.legend()
                else:
                    ax2.set_title('Target Distribution (Not Available)')
                
                # Plot model output
                ax3 = fig.add_subplot(133)
                if output_np is not None:
                    if output_np.shape[-1] == 1:  # 1D data
                        ax3.scatter(output_np[0, :, 0], np.zeros_like(output_np[0, :, 0]), 
                                   alpha=0.5, color='red', label='Output')
                    else:  # 2D data
                        ax3.scatter(output_np[0, :, 0], output_np[0, :, 1], 
                                   alpha=0.5, color='red', label='Output')
                    ax3.set_title('Model Output')
                    ax3.legend()
                else:
                    ax3.set_title('Model Output (Error)')
                
                plt.tight_layout()
                
                # Save figure
                fig_path = output_dir / f"sample_{i+1}.png"
                plt.savefig(fig_path)
                logger.info(f"Saved visualization to {fig_path}")
                
                figures.append(fig)
        
        return figures