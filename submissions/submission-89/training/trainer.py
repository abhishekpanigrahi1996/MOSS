"""
Trainer module for GMM transformer models.

This module provides a comprehensive training and evaluation system.
"""

import os
import time
import logging
import warnings
import inspect
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW, Adam, SGD, RMSprop, Adagrad
from torch.amp import GradScaler, autocast
logger = logging.getLogger(__name__)

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

from config import (
    ExperimentConfig, DeviceConfig, TrainingConfig,
    OptimizerConfig, SchedulerConfig
)
from metrics import MetricsTracker, create_metric_functions

# Configure logger
logger = logging.getLogger(__name__)


class GMMTrainer:
    """
    Trainer for GMM transformer models.
    
    This class handles the entire training process, including optimization,
    validation, metric tracking, visualization, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[ExperimentConfig] = None,
        experiment_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Experiment configuration
            experiment_dir: Directory for experiment outputs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or ExperimentConfig()
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        # Set device
        self.device = self.config.device.get_device()
        self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        self.scaler = GradScaler() if self.config.device.use_mixed_precision else None
        
        # Initialize state
        self.current_epoch = 0
        self._expected_total_epochs = self.config.training.num_epochs
        self._start_epoch = 0
        self._additional_epochs = 0  # Will be set during resume_training
        self.best_val_loss = float('inf')
        self.history = {
            "train_loss": [],
            "train_normalized_loss": [],
            "val_loss": [],
            "val_normalized_loss": [],
            "learning_rate": [],
            "val_metrics": []
        }
        
        # Initialize metrics
        self.metric_fns = create_metric_functions(self.config.validation.metrics.metrics)
        
        # Set up TensorBoard if enabled
        self.writer = None
        if TENSORBOARD_AVAILABLE and self.config.logging.tensorboard_enabled and self.experiment_dir:
            log_dir = self.experiment_dir / "tensorboard"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging enabled at {log_dir}")
    
    def _create_optimizer(self) -> Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        optimizer_config = self.config.training.optimizer
        lr = optimizer_config.learning_rate
        wd = optimizer_config.weight_decay
        
        # Prepare optimizer parameters - exclude bias and normalization layers from weight decay
        if optimizer_config.exclude_bias_and_norm:
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if "bias" in name or "norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            params = [
                {'params': decay_params, 'weight_decay': wd},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
        else:
            params = self.model.parameters()
        
        # Create optimizer
        if optimizer_config.optimizer == "adam":
            optimizer = Adam(
                params,
                lr=lr,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                weight_decay=wd
            )
        elif optimizer_config.optimizer == "adamw":
            if optimizer_config.exclude_bias_and_norm:
                # Handle flow predictor parameters separately
                params_regular = []
                params_flow = []
                no_decay_params = []
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    if "flow_predictor" in name:
                        params_flow.append(param)
                    elif "bias" in name or "norm" in name:
                        no_decay_params.append(param)
                    else:
                        params_regular.append(param)
                
                params = [
                    {'params': params_regular, 'lr': lr, 'weight_decay': wd},
                    {'params': params_flow, 'lr': lr * 100, 'weight_decay': 0.0},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]
            else:
                # Handle flow predictor parameters separately
                params_regular = [p for n, p in self.model.named_parameters() if "flow_predictor" not in n]
                params_flow = [p for n, p in self.model.named_parameters() if "flow_predictor" in n]
                params = [
                    {'params': params_regular, 'lr': lr, 'weight_decay': wd},
                    {'params': params_flow, 'lr': lr * 100, 'weight_decay': 0.0}
                ]
                
            optimizer = AdamW(
                params,
                betas=(optimizer_config.beta1, optimizer_config.beta2)
            )
        elif optimizer_config.optimizer == "sgd":
            optimizer = SGD(
                params,
                lr=lr,
                momentum=optimizer_config.momentum,
                weight_decay=wd
            )
        elif optimizer_config.optimizer == "rmsprop":
            optimizer = RMSprop(
                params,
                lr=lr,
                weight_decay=wd
            )
        elif optimizer_config.optimizer == "adagrad":
            optimizer = Adagrad(
                params,
                lr=lr,
                weight_decay=wd
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[object]:
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            PyTorch scheduler or None
        """
        scheduler_config = self.config.training.scheduler
        training_config = self.config.training
        
        # Calculate total steps for warmup
        if self.train_loader is None:
            return None
            
        total_steps = len(self.train_loader) * training_config.num_epochs
        warmup_steps = int(total_steps * scheduler_config.warmup_ratio)
        
        # Get initial and final learning rates
        initial_lr = training_config.optimizer.learning_rate
        min_lr = initial_lr * scheduler_config.min_lr_ratio
        
        # Create scheduler based on type
        if scheduler_config.scheduler_type == "constant":
            # No scheduler needed for constant lr
            return None
            
        elif scheduler_config.scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            
            # For linear schedule with warmup
            if warmup_steps > 0:
                from torch.optim.lr_scheduler import SequentialLR, ConstantLR
                
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                decay = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=scheduler_config.min_lr_ratio,
                    total_iters=total_steps - warmup_steps
                )
                
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, decay],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=scheduler_config.min_lr_ratio,
                    total_iters=total_steps
                )
                
            return scheduler
            
        elif scheduler_config.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            # For cosine schedule with warmup
            if warmup_steps > 0:
                from torch.optim.lr_scheduler import SequentialLR, LinearLR
                
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=min_lr
                )
                
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                    eta_min=min_lr
                )
                
            return scheduler
            
        elif scheduler_config.scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            
            # Check for required decay_steps
            if scheduler_config.decay_steps is None:
                raise ValueError("decay_steps must be specified for step scheduler")
                
            scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.decay_steps,
                gamma=scheduler_config.decay_rate
            )
            
            return scheduler
            
        elif scheduler_config.scheduler_type == "exponential":
            from torch.optim.lr_scheduler import ExponentialLR
            
            # Calculate gamma to reach min_lr in total steps
            if scheduler_config.decay_steps is None:
                raise ValueError("decay_steps must be specified for exponential scheduler")
                
            gamma = (min_lr / initial_lr) ** (1.0 / scheduler_config.decay_steps)
            
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            
            return scheduler
            
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config.scheduler_type}")
    
    def _create_loss_function(self) -> Callable:
        """
        Create loss function based on configuration.
        
        Returns:
            Loss function
        """
        loss_config = self.config.training.loss
        
        if isinstance(loss_config.loss_type, str):
            if loss_config.loss_type == "mse":
                from losses import create_mse_loss
                return create_mse_loss()
            elif loss_config.loss_type == "wasserstein":
                from losses import create_wasserstein_loss
                return create_wasserstein_loss(
                    algorithm=loss_config.wasserstein_algorithm,
                    backend=loss_config.wasserstein_backend,
                    epsilon=loss_config.wasserstein_epsilon,
                    max_iterations=loss_config.wasserstein_max_iter,
                    use_true_weights=loss_config.use_true_weights
                )
            elif loss_config.loss_type == "energy":
                from losses import create_energy_loss
                return create_energy_loss()
            else:
                raise ValueError(f"Unknown loss type: {loss_config.loss_type}")
        elif isinstance(loss_config.loss_type, dict):
            # Handle dictionary-based configuration
            loss_type = loss_config.loss_type.get("type", "mse")
            # Create a copy of the config without the 'type' key
            loss_params = {k: v for k, v in loss_config.loss_type.items() if k != 'type'}
            
            if loss_type == "mse":
                from losses import create_mse_loss
                return create_mse_loss(**loss_params)
            elif loss_type == "wasserstein":
                try:
                    loss_params["use_true_weights"] = loss_config.use_true_weights
                except:
                    loss_params["use_true_weights"] = False
                from losses import create_wasserstein_loss
                return create_wasserstein_loss(**loss_params)
            elif loss_type == "energy":
                from losses import create_energy_loss
                return create_energy_loss(**loss_params)
            else:
                raise ValueError(f"Unknown loss type in dict config: {loss_type}")
        else:
            raise ValueError(f"Invalid loss configuration: {loss_config.loss_type}")
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, snr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss with optional normalization.
        
        Args:
            predictions: Model predictions
            targets: Target values
            snr: Optional SNR values for normalization
            
        Returns:
            Tuple of (raw_loss, normalized_loss)
        """
        # Get per-sample losses without reduction
        per_sample_loss = self.loss_fn(predictions, targets, reduction='none')
        
        # Compute raw loss as mean
        raw_loss = per_sample_loss.mean()
        
        # Apply normalization if specified
        if self.config.training.loss.normalization == "none":
            normalized_loss = raw_loss
        elif self.config.training.loss.normalization == "snr_power":
            if snr is None:
                warnings.warn("SNR values not provided for SNR power normalization")
                normalized_loss = raw_loss
            else:
                # Normalize by SNR raised to power
                snr_factor = torch.pow(snr, self.config.training.loss.snr_power)
                # Apply normalization to per-sample losses
                normalized_per_sample = per_sample_loss * (snr_factor + 1e-8)
                normalized_loss = normalized_per_sample.mean()
        elif self.config.training.loss.normalization == "log":
            # Apply log normalization to per-sample losses
            normalized_per_sample = torch.log(per_sample_loss + 1e-8)  # Add small epsilon to avoid log(0)
            normalized_loss = normalized_per_sample.mean()
        else:
            raise ValueError(f"Unknown loss normalization type: {self.config.training.loss.normalization}")
        
        return raw_loss, normalized_loss
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for (overrides config)
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        # Get number of epochs
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)
            
        # Calculate end epoch
        start_epoch = self.current_epoch
        end_epoch = start_epoch + num_epochs
        
        # Update expected total epochs for test compatibility
        self._expected_total_epochs = max(self._expected_total_epochs, end_epoch)
        
        # Track early stopping
        early_stopping_patience = self.config.training.early_stopping_patience
        early_stopping_delta = self.config.training.early_stopping_delta
        
        best_val_loss = self.best_val_loss
        patience_counter = 0
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch+1} to {end_epoch})")
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss, train_normalized_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_normalized_loss"].append(train_normalized_loss)
            
            # Learning rate is now updated and logged per batch
            
            # Still track final LR of the epoch for the history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)
            
            # Log to console
            logger.info(f"Epoch {epoch+1}/{end_epoch} - Train Loss: {train_loss:.6f}, Train Normalized Loss: {train_normalized_loss:.6f}")
            
            # Validate if needed
            if self.val_loader is not None and (epoch + 1) % self.config.training.val_every == 0:
                val_loss, val_normalized_loss, val_metrics = self._validate_epoch()
                self.history["val_loss"].append(val_loss)
                self.history["val_normalized_loss"].append(val_normalized_loss)
                self.history["val_metrics"].append(val_metrics)
                
                # Log to console
                logger.info(f"Validation Loss: {val_loss:.6f}, Validation Normalized Loss: {val_normalized_loss:.6f}")
                
                # Check for improvement
                improved = False
                if val_loss < best_val_loss - early_stopping_delta:
                    best_val_loss = val_loss
                    self.best_val_loss = val_loss
                    improved = True
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.training.save_best_model and self.experiment_dir:
                        self._save_checkpoint(is_best=True)
                        logger.info(f"Saved best model with validation loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    
                # Early stopping
                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
            
            # Always save the latest model after each epoch
            if self.experiment_dir:
                self._save_checkpoint(is_latest=True)
                
                # Additionally save epoch-specific checkpoint based on interval
                if self.config.training.checkpoint_every > 0:
                    if (epoch + 1) % self.config.training.checkpoint_every == 0:
                        self._save_checkpoint()
                        logger.info(f"Saved checkpoint at epoch {epoch+1}")
                    
            # Update TensorBoard
            if self.writer is not None:
                step = epoch + 1
                
                # Log training metrics
                self.writer.add_scalar("Loss/Train", train_loss, step)
                self.writer.add_scalar("Loss/TrainNormalized", train_normalized_loss, step)
                
                # Log validation metrics
                if self.val_loader is not None and (epoch + 1) % self.config.training.val_every == 0:
                    self.writer.add_scalar("Loss/Validation", val_loss, step)
                    self.writer.add_scalar("Loss/ValidationNormalized", val_normalized_loss, step)
                    
                    # Log detailed validation metrics
                    for metric_name, value in val_metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"Metrics/{metric_name}", value, step)
                            
                # Log histograms if enabled
                if self.config.logging.log_histograms:
                    if (epoch + 1) % self.config.logging.histogram_every_n_epochs == 0:
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                self.writer.add_histogram(f"Parameters/{name}", param.data, step)
                                if param.grad is not None:
                                    self.writer.add_histogram(f"Gradients/{name}", param.grad.data, step)
        
        # Save final checkpoint
        if self.experiment_dir:
            self._save_checkpoint(is_final=True)
            logger.info("Saved final checkpoint")
            
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            
        return self.history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Tuple of (average regular loss, average normalized loss)
        """
        self.model.train()
        total_loss = 0.0
        total_normalized_loss = 0.0
        num_batches = 0
        
        # Create progress bar if enabled
        if self.config.training.show_progress_bar:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = self.train_loader
        
        for batch in pbar:
            # Move batch to device
            if isinstance(batch, dict):
                inputs = batch.get('points', batch)
                targets = batch  # Keep the target as a dictionary
            else:
                inputs = batch[0]
                targets = batch[1]
                
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Handle targets differently based on type
            if isinstance(targets, dict):
                # Move all tensors in the dictionary to device
                for key, value in list(targets.items()):
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(self.device)
            else:
                # If targets is a tensor, move it to device directly
                targets = targets.to(self.device)
            
            # Forward pass with mixed precision if enabled
            with autocast(device_type=self.device.type, enabled=self.config.device.use_mixed_precision):
                # Check if model forward accepts targets parameter
                if 'targets' in inspect.signature(self.model.forward).parameters:
                    predictions = self.model(inputs, targets=targets)
                else:
                    predictions = self.model(inputs)
                    
                # Get SNR for normalization if available
                snr = None
                if isinstance(targets, dict) and 'snr_db' in targets:
                    snr = 10**(targets['snr_db'] / 10)  # Convert dB to linear scale
                    
                raw_loss, normalized_loss = self._compute_loss(predictions, targets, snr)
            
            # Backward pass
            if self.config.device.use_mixed_precision:
                self.scaler.scale(normalized_loss).backward()
            else:
                normalized_loss.backward()
            
            # Update weights
            if self.config.device.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update scheduler if it exists
            if self.scheduler is not None:
                current_step = self.current_epoch * len(self.train_loader) + num_batches
                if hasattr(self.scheduler, "step_update"):
                    self.scheduler.step_update(current_step)
                else:
                    self.scheduler.step()
            
            # Update statistics
            total_loss += raw_loss.item()
            total_normalized_loss += normalized_loss.item()
            num_batches += 1
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update progress bar
            if self.config.training.show_progress_bar:
                pbar.set_postfix({
                    "loss": f"{raw_loss.item():.6f}",
                    "norm_loss": f"{normalized_loss.item():.6f}",
                    "lr": f"{current_lr:.6e}"
                })
            
            # Log to TensorBoard
            if self.writer is not None and num_batches % self.config.logging.log_every_n_steps == 0:
                step = self.current_epoch * len(self.train_loader) + num_batches
                self.writer.add_scalar("Loss/Step", raw_loss.item(), step)
                self.writer.add_scalar("Loss/NormalizedStep", normalized_loss.item(), step)
                self.writer.add_scalar("LearningRate/Step", current_lr, step)
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_normalized_loss = total_normalized_loss / num_batches
        
        return avg_loss, avg_normalized_loss
    
    def _validate_epoch(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Validate the model on the validation set.
        
        Returns:
            Tuple of (average regular loss, average normalized loss, metrics dictionary)
        """
        if self.val_loader is None:
            return 0.0, 0.0, {}
            
        self.model.eval()
        total_loss = 0.0
        total_normalized_loss = 0.0
        num_batches = 0
        
        # Create metrics tracker
        compare_with_kmeans = self.config.validation.metrics.compare_with_kmeans
        tracker = MetricsTracker(
            metric_fns=self.metric_fns,
            compare_with_kmeans=compare_with_kmeans,
            device=self.device,
            include_loss=True
        )
        
        # Set up progress bar
        pbar = tqdm(
            total=len(self.val_loader),
            disable=not self.config.training.show_progress_bar,
            desc="Validation"
        )
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = batch.get('points', batch)
                    targets = batch  # Keep the target as a dictionary
                else:
                    inputs = batch[0]
                    targets = batch[1]
                    
                # Move inputs to device
                inputs = inputs.to(self.device)
                
                # Handle targets differently based on type
                if isinstance(targets, dict):
                    # Move all tensors in the dictionary to device
                    for key, value in list(targets.items()):
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                else:
                    # If targets is a tensor, move it to device directly
                    targets = targets.to(self.device)
                
                # Forward pass
                # Check if model forward accepts targets parameter
                if 'targets' in inspect.signature(self.model.forward).parameters:
                    predictions = self.model(inputs, targets=targets)
                else:
                    predictions = self.model(inputs)
                    
                # Get SNR for normalization if available
                snr = None
                if isinstance(targets, dict) and 'snr_db' in targets:
                    snr = 10**(targets['snr_db'] / 10)  # Convert dB to linear scale
                    
                raw_loss, normalized_loss = self._compute_loss(predictions, targets, snr)
                
                # Update statistics
                total_loss += raw_loss.item()
                total_normalized_loss += normalized_loss.item()
                num_batches += 1
                
                # Update metrics tracker
                tracker.update_loss(raw_loss.item(), inputs.size(0))
                tracker.update(predictions, targets, inputs)
                
                # Update progress bar
                pbar.update()
        
        # Close progress bar
        pbar.close()
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_normalized_loss = total_normalized_loss / num_batches
        
        # Compute metrics
        metrics = tracker.compute()
        
        return avg_loss, avg_normalized_loss, metrics
    
    def evaluate(
        self,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        loader = test_loader or self.val_loader
        
        if loader is None:
            return {"error": "No data loader provided for evaluation"}
            
        self.model.eval()
        
        # Create metrics tracker
        compare_with_kmeans = self.config.validation.metrics.compare_with_kmeans
        tracker = MetricsTracker(
            metric_fns=self.metric_fns,
            compare_with_kmeans=compare_with_kmeans,
            device=self.device,
            include_loss=True
        )
        
        # Set up progress bar
        pbar = tqdm(
            total=len(loader),
            disable=not self.config.training.show_progress_bar,
            desc="Evaluation"
        )
        
        # Evaluation loop
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = batch.get('points', batch)
                    targets = batch  # Keep the target as a dictionary
                else:
                    inputs = batch[0]
                    targets = batch[1]
                    
                # Move inputs to device
                inputs = inputs.to(self.device)
                
                # Handle targets differently based on type
                if isinstance(targets, dict):
                    # For dictionary targets, move individual tensors to the device
                    if 'centers' in targets:
                        targets['centers'] = targets['centers'].to(self.device)
                    if 'labels' in targets:
                        targets['labels'] = targets['labels'].to(self.device)
                else:
                    # If targets is a tensor, move it to device directly
                    targets = targets.to(self.device)
                
                # Forward pass
                # Check if model forward accepts targets parameter
                if 'targets' in inspect.signature(self.model.forward).parameters:
                    outputs = self.model(inputs, targets=targets)
                else:
                    outputs = self.model(inputs)
                
                # Get per-sample losses without reduction
                per_sample_loss = self.loss_fn(outputs, targets, reduction='none')
                
                # Compute regular loss as mean
                loss = per_sample_loss.mean()
                batch_size = inputs.size(0)
                
                # Update metrics
                tracker.update_loss(loss.item(), batch_size)
                tracker.update(outputs, targets, inputs)
                
                # Update progress bar
                pbar.update()
        
        # Close progress bar
        pbar.close()
        
        # Compute metrics
        metrics = tracker.compute()
        
        return metrics
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False, is_latest: bool = False, custom_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
            is_latest: Whether this is the latest model
            custom_path: Optional custom path to save checkpoint to
        """
        if self.experiment_dir is None and custom_path is None:
            return
            
        # Create checkpoint
        # IMPORTANT: self.current_epoch represents the last COMPLETED epoch (0-indexed)
        # This is needed for proper resumption and test compatibility
        checkpoint = {
            "epoch": self.current_epoch,  # epoch represents the last completed epoch
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.best_val_loss,
            "history": self.history,
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else None,
            "batch_size": self.config.training.batch_size,
            "train_samples": self.config.training.num_train_samples
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        # Determine save path
        if custom_path is not None:
            path = Path(custom_path)
        else:
            # Create checkpoints directory
            checkpoint_dir = self.experiment_dir / "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save checkpoint based on type
            if is_best:
                path = checkpoint_dir / "best_model.pt"
            elif is_final:
                path = checkpoint_dir / "final_model.pt"
            elif is_latest:
                path = checkpoint_dir / "latest_model.pt"
            else:
                path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch+1}.pt"
                
        # Create parent directory if it doesn't exist
        os.makedirs(path.parent, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        # Save data loader states if available
        if hasattr(self, 'train_loader') and hasattr(self.train_loader, 'save_state') and self.experiment_dir:
            try:
                # Save training data loader state
                data_state_path = self.experiment_dir / "data_state.json"
                self.train_loader.save_state(str(data_state_path))
                logger.info(f"Saved training data loader state to {data_state_path}")
                
                # Save validation data loader state if available
                if hasattr(self, 'val_loader') and hasattr(self.val_loader, 'save_state'):
                    val_state_path = self.experiment_dir / "val_data_state.json"
                    self.val_loader.save_state(str(val_state_path))
                    logger.info(f"Saved validation data loader state to {val_state_path}")
            except Exception as e:
                logger.warning(f"Error saving data loader state: {e}")
                
    def _reconfigure_scheduler(self, total_remaining_epochs: int) -> None:
        """
        Reconfigure learning rate scheduler for new parameters.
        
        Args:
            total_remaining_epochs: Total number of remaining epochs
        """
        if self.scheduler is None:
            logger.info("No scheduler to reconfigure")
            return
            
        scheduler_config = self.config.training.scheduler
        
        # Get current learning rate from optimizer
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Calculate updated total steps
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * total_remaining_epochs
        warmup_steps = int(total_steps * scheduler_config.warmup_ratio)
        
        logger.info(f"Reconfiguring scheduler: current_lr={current_lr:.6e}, "
                    f"steps_per_epoch={steps_per_epoch}, "
                    f"total_steps={total_steps}, "
                    f"warmup_steps={warmup_steps}")
        
        # Store original scheduler
        old_scheduler = self.scheduler
        
        # Create new scheduler based on the type
        if scheduler_config.scheduler_type == "constant":
            # No need to reconfigure for constant LR
            logger.info("Constant LR scheduler doesn't need reconfiguration")
            return
            
        elif scheduler_config.scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            # For linear schedule with warmup
            if warmup_steps > 0:
                # Create warmup scheduler if needed
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=current_lr / 10.0 / current_lr,  # Warmup from 10x smaller LR
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                # Create decay scheduler
                min_lr = current_lr * scheduler_config.min_lr_ratio
                decay = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=min_lr / current_lr,
                    total_iters=total_steps - warmup_steps
                )
                
                # Combine schedulers
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, decay],
                    milestones=[warmup_steps]
                )
                logger.info(f"Reconfigured linear scheduler with warmup: current_lr={current_lr:.6e}, min_lr={min_lr:.6e}")
            else:
                # Just linear decay
                min_lr = current_lr * scheduler_config.min_lr_ratio
                self.scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=min_lr / current_lr,
                    total_iters=total_steps
                )
                logger.info(f"Reconfigured linear scheduler: current_lr={current_lr:.6e}, min_lr={min_lr:.6e}")
                
        elif scheduler_config.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
            
            # Cosine schedule with optional warmup
            if warmup_steps > 0:
                # Create warmup scheduler
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=current_lr / 10.0 / current_lr,  # Warmup from 10x smaller LR
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                # Create cosine scheduler
                min_lr = current_lr * scheduler_config.min_lr_ratio
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=min_lr
                )
                
                # Combine schedulers
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps]
                )
                logger.info(f"Reconfigured cosine scheduler with warmup: current_lr={current_lr:.6e}, min_lr={min_lr:.6e}")
            else:
                # Just cosine annealing
                min_lr = current_lr * scheduler_config.min_lr_ratio
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                    eta_min=min_lr
                )
                logger.info(f"Reconfigured cosine scheduler: current_lr={current_lr:.6e}, min_lr={min_lr:.6e}")
        
        elif scheduler_config.scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            
            # Check for required decay_steps
            if scheduler_config.decay_steps is None:
                logger.warning("decay_steps not specified for step scheduler, using 1 epoch")
                decay_steps = steps_per_epoch
            else:
                decay_steps = scheduler_config.decay_steps
                
            self.scheduler = StepLR(
                self.optimizer,
                step_size=decay_steps,
                gamma=scheduler_config.decay_rate
            )
            
            logger.info(f"Reconfigured step scheduler: current_lr={current_lr:.6e}, "
                        f"step_size={decay_steps}, gamma={scheduler_config.decay_rate}")
            
        else:
            logger.warning(f"Unsupported scheduler type for reconfiguration: {scheduler_config.scheduler_type}")
            self.scheduler = old_scheduler  # Restore original scheduler
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load training state
        if "epoch" in checkpoint:
            # Resume from next epoch (checkpoint from epoch N, start at N+1)
            # self.current_epoch = LAST COMPLETED epoch
            self.current_epoch = checkpoint["epoch"] + 1
            
        if "loss" in checkpoint:
            self.best_val_loss = checkpoint["loss"]
            
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")
        
    def export_model(self, path: Union[str, Path], format: str = "pytorch") -> None:
        """
        Export model to file.
        
        Args:
            path: Path to save exported model
            format: Export format ('pytorch', 'onnx', or 'torchscript')
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        os.makedirs(path.parent, exist_ok=True)
        
        if format == "pytorch":
            # Save PyTorch model
            torch.save(self.model.state_dict(), path)
        elif format == "torchscript":
            # Export to TorchScript
            self.model.eval()
            
            # Create example inputs
            batch_size = 1
            seq_len = 100
            input_dim = self.model.transformer.input_dim
            dummy_input = torch.zeros(batch_size, seq_len, input_dim, device=self.device)
            
            # Trace model
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(path)
        elif format == "onnx":
            # Export to ONNX
            try:
                import onnx
                
                self.model.eval()
                
                # Create example inputs
                batch_size = 1
                seq_len = 100
                input_dim = self.model.transformer.input_dim
                dummy_input = torch.zeros(batch_size, seq_len, input_dim, device=self.device)
                
                # Export model
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size", 1: "sequence_length"},
                        "output": {0: "batch_size", 1: "sequence_length"}
                    }
                )
            except ImportError:
                logger.error("ONNX export requires the 'onnx' package. Install with: pip install onnx")
                raise
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Exported model to {path} in {format} format")