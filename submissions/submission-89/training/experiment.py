"""
Experiment management for GMM transformer framework.

This module provides a high-level interface for running experiments.
"""

# Set seeds for reproducibility
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
import numpy as np
import random
np.random.seed(42)
random.seed(42)

import os
import json
import time
import logging
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader

from config import ExperimentConfig
from model import GMMTransformer, ClusterPredictionModel
from .trainer import GMMTrainer
from utils.checkpointing import load_checkpoint, get_latest_checkpoint, get_best_checkpoint

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manager for running GMM transformer experiments.
    
    This class provides a high-level interface for setting up and running
    experiments with the GMM transformer framework.
    """
    
    def __init__(
        self,
        config: Optional[Union[ExperimentConfig, Dict[str, Any], str, Path]] = None,
        experiment_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration (object, dict, or file path)
            experiment_dir: Directory for experiment outputs
        """
        # Initialize configuration
        self.config = self._init_config(config)
        
        # Set up experiment directory
        self.experiment_dir = self._init_experiment_dir(experiment_dir)
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.trainer = None
        
        # Set up logging
        self._setup_logging()
        
        logger.info(f"Experiment '{self.config.metadata.experiment_name}' initialized with ID: {self.config.metadata.id}")
    
    def _init_config(
        self,
        config: Optional[Union[ExperimentConfig, Dict[str, Any], str, Path]]
    ) -> ExperimentConfig:
        """
        Initialize configuration from various sources.
        
        Args:
            config: Configuration source (object, dict, or file path)
            
        Returns:
            Validated experiment configuration
        """
        if config is None:
            # Create default configuration
            return ExperimentConfig()
            
        elif isinstance(config, ExperimentConfig):
            # Use provided configuration object
            return config
            
        elif isinstance(config, dict):
            # Create from dictionary
            return ExperimentConfig.model_validate(config)
            
        elif isinstance(config, (str, Path)):
            # Load from file
            config_path = Path(config)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            # Try to load as JSON
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    
                return ExperimentConfig.model_validate(config_dict)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in configuration file: {config_path}")
                
        else:
            raise TypeError(f"Unsupported configuration type: {type(config)}")
    
    def _init_experiment_dir(
        self,
        experiment_dir: Optional[Union[str, Path]]
    ) -> Path:
        """
        Initialize experiment directory.
        
        Args:
            experiment_dir: Directory for experiment outputs
            
        Returns:
            Path to experiment directory
        """
        if experiment_dir is not None:
            # Use provided directory
            exp_dir = Path(experiment_dir)
        elif self.config.metadata.id is not None:
            # Create from configuration
            exp_dir = self.config.paths.get_experiment_dir(self.config.metadata.id)
        else:
            # Create default with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            exp_dir = Path(f"./runs/exp_{timestamp}")
        
        # Create directory
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration to experiment directory
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            if hasattr(self.config, "model_dump_json"):
                f.write(self.config.model_dump_json(indent=2))
            else:
                json.dump(self.config.model_dump(), f, indent=2)
        
        return exp_dir
    
    def _setup_logging(self) -> None:
        """Set up logging for the experiment."""
        logging_config = self.config.logging
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, logging_config.log_level))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set up formatters
        formatter = logging.Formatter(
            logging_config.log_format,
            datefmt=logging_config.log_date_format
        )
        
        # Set up console handler
        if logging_config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Set up file handler
        if logging_config.log_to_file:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def setup(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        model: Optional[torch.nn.Module] = None
    ) -> None:
        """
        Set up experiment components.
        
        Args:
            train_loader: Optional pre-created training data loader
            val_loader: Optional pre-created validation data loader
            model: Optional pre-created model
        """
        # Set up data loaders - call create method only once if needed
        if train_loader is None or val_loader is None:
            default_train, default_val = self._create_data_loaders()
            
            # Use provided loaders or defaults
            self.train_loader = train_loader if train_loader is not None else default_train
            self.val_loader = val_loader if val_loader is not None else default_val
        else:
            # Both loaders provided, use them directly
            self.train_loader = train_loader
            self.val_loader = val_loader
        
        # Create or use model
        self.model = model if model is not None else self._create_model()
        
        # Create trainer
        self.trainer = GMMTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            experiment_dir=self.experiment_dir
        )
        
        logger.info("Experiment setup completed")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        try:
            # Use the built-in data loader creation method from the config
            train_loader, val_loader = self.config.create_data_loaders(
                batch_size=self.config.training.batch_size,
                num_train_samples=self.config.training.num_train_samples,
                num_val_samples=self.config.validation.num_val_samples,
                device=self.config.device.get_device()
            )
            
            logger.info(
                f"Created data loaders with batch sizes: "
                f"train={self.config.training.batch_size}, "
                f"val={self.config.validation.validation_batch_size}"
            )
            
            return train_loader, val_loader
        except ImportError as e:
            logger.error(f"Error creating data loaders: {e}")
            raise
    
    def _create_model(self) -> torch.nn.Module:
        """
        Create model based on configuration.
        
        Returns:
            Model instance
        """
        from model.factory import create_model_from_config
        
        # Create model using factory function
        device = self.config.device.get_device()
        model = create_model_from_config(
            config=self.config.model,
            device=device
        )
        
        return model
    
    def run(
        self,
        num_epochs: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Args:
            num_epochs: Number of epochs to train for
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        # Set up components if not already set up
        if self.trainer is None:
            self.setup()
        
        # Run training
        start_time = time.time()
        
        try:
            history = self.trainer.train(
                num_epochs=num_epochs,
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            # Record training time
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save training summary
            self._save_training_summary(history, training_time)
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _save_training_summary(
        self,
        history: Dict[str, Any],
        training_time: float
    ) -> None:
        """
        Save training summary to experiment directory.
        
        Args:
            history: Training history
            training_time: Total training time in seconds
        """
        if self.experiment_dir is None:
            return
        
        # Create summary dictionary
        summary = {
            "experiment_id": self.config.metadata.id,
            "experiment_name": self.config.metadata.experiment_name,
            "training_time": training_time,
            "epochs_completed": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary to file
        summary_path = self.experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def evaluate(
        self,
        test_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for test data
            checkpoint_path: Path to checkpoint to load
            
        Returns:
            Evaluation metrics
        """
        # Set up components if not already set up
        if self.trainer is None:
            self.setup()
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            if not hasattr(self.trainer, "_load_checkpoint"):
                raise AttributeError("Trainer does not support checkpoint loading")
                
            self.trainer._load_checkpoint(checkpoint_path)
        
        # Run evaluation
        metrics = self.trainer.evaluate(test_loader=test_loader)
        
        # Save evaluation results
        if self.experiment_dir is not None:
            eval_path = self.experiment_dir / "evaluation_results.json"
            
            # Convert tensor values to Python types
            metrics_json = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics_json[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                    metrics_json[k] = [x.item() if x.numel() == 1 else x.tolist() for x in v]
                else:
                    metrics_json[k] = v
            
            with open(eval_path, 'w') as f:
                json.dump(metrics_json, f, indent=2)
        
        return metrics
    
    @classmethod
    def load(cls, experiment_dir: Union[str, Path]) -> "ExperimentManager":
        """
        Load experiment from directory.
        
        Args:
            experiment_dir: Directory containing experiment
            
        Returns:
            Loaded experiment manager
        """
        experiment_dir = Path(experiment_dir)
        
        # Check if directory exists
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Check for configuration file
        config_path = experiment_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        config = ExperimentConfig.load(config_path)
        
        # Create experiment manager
        manager = cls(config=config, experiment_dir=experiment_dir)
        
        return manager
        
    def get_model_manager(self):
        """
        Get a ModelManager for this experiment.
        
        Returns:
            ModelManager instance for this experiment
        """
        # Import here to avoid circular imports
        from utils.model_management import ModelManager
        
        # Check if trainer exists and has a model
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.setup()
            
        return ModelManager.from_experiment(self)
        
    def export_model(self, path: Union[str, Path]):
        """
        Export model to the specified path.
        
        Args:
            path: Path to save the model to
            
        Returns:
            Path where the model was saved
        """
        manager = self.get_model_manager()
        return manager.save(path)
        
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None, is_best: bool = False):
        """
        Explicitly save a checkpoint.
        
        Args:
            path: Optional specific path to save to
            is_best: Whether to save as best model
            
        Returns:
            Path where the checkpoint was saved
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise ValueError("Trainer not initialized, call setup() first")
            
        if path is None:
            # Use trainer's checkpoint directory
            checkpoint_dir = self.experiment_dir / "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if is_best:
                path = checkpoint_dir / "best_model.pt"
            else:
                path = checkpoint_dir / f"checkpoint_epoch_{self.trainer.current_epoch+1}.pt"
                
        self.trainer._save_checkpoint(is_best=is_best, custom_path=path)
        return path
        
    @classmethod
    def load_best_model(cls, experiment_dir: Union[str, Path]) -> "ExperimentManager":
        """
        Load an experiment with the best model from an experiment directory.
        
        Args:
            experiment_dir: Directory containing experiment
            
        Returns:
            Experiment manager with the best model loaded
        """
        experiment_dir = Path(experiment_dir)
        checkpoint_dir = experiment_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Try to find best checkpoint first as "best_model.pt"
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint_path = best_path
        else:
            # Try to find best checkpoint by metric
            checkpoint_path = get_best_checkpoint(checkpoint_dir)
            
        if checkpoint_path is None:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
            
        # Load the experiment
        manager = cls.load(experiment_dir)
        
        # Setup experiment components
        manager.setup()
        
        # Load the checkpoint
        manager.trainer._load_checkpoint(checkpoint_path)
        
        logger.info(f"Loaded best model from {checkpoint_path}")
        return manager
        
    @classmethod
    def load_latest_model(cls, experiment_dir: Union[str, Path], device=None) -> "ExperimentManager":
        """
        Load an experiment with the latest model from an experiment directory.
        
        Args:
            experiment_dir: Directory containing experiment
            
        Returns:
            Experiment manager with the latest model loaded
        """
        experiment_dir = Path(experiment_dir)
        checkpoint_dir = experiment_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Try to find latest checkpoint first as "latest_model.pt"
        latest_path = checkpoint_dir / "latest_model.pt"
        if latest_path.exists():
            checkpoint_path = latest_path
        else:
            # Try to find latest checkpoint by modification time
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            
        if checkpoint_path is None:
            raise FileNotFoundError(f"No latest checkpoint found in {checkpoint_dir}")
            
        # Load the experiment
        manager = cls.load(experiment_dir)
        
        # Setup experiment components
        manager.setup()
        
        # Load the checkpoint
        manager.trainer._load_checkpoint(checkpoint_path)
        
        logger.info(f"Loaded latest model from {checkpoint_path}")
        return manager
        
    def resume_training(
        self, 
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        train_samples: Optional[int] = None,
        reconfigure_scheduler: bool = False
    ) -> Dict[str, Any]:
        """
        Resume training from the latest checkpoint with flexible parameter options.
        
        Args:
            num_epochs: Number of additional epochs to train for
            batch_size: New batch size to use (if different from original)
            train_samples: New training samples count (if different from original)
            reconfigure_scheduler: Whether to rebuild scheduler based on new parameters
            
        Returns:
            Training history
        """
        # Get latest checkpoint
        checkpoint_dir = self.experiment_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Try to find latest checkpoint first as "latest_model.pt"
        latest_path = checkpoint_dir / "latest_model.pt"
        if latest_path.exists():
            checkpoint_path = latest_path
        else:
            # Try to find latest checkpoint by modification time
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        # Load the checkpoint to access epoch information
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device.get_device())
        
        # Extract current epoch from checkpoint directly to ensure we have access to it
        # before loading into trainer
        current_epoch_from_checkpoint = checkpoint.get("epoch", 0)
        
        # Track parameter changes for later use
        self._batch_size_changed = batch_size is not None and batch_size != self.config.training.batch_size
        self._train_samples_changed = train_samples is not None and train_samples != self.config.training.num_train_samples
        
        # Update config with new parameters if provided
        if batch_size is not None:
            logger.info(f"Changing batch size from {self.config.training.batch_size} to {batch_size}")
            self.config.training.batch_size = batch_size
            
        if train_samples is not None:
            logger.info(f"Changing training samples from {self.config.training.num_train_samples} to {train_samples}")
            self.config.training.num_train_samples = train_samples
        
        # Resume data loaders from state
        self._resume_data_loaders()
        
        # Set up trainer if not already setup
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.setup(train_loader=self.train_loader, val_loader=self.val_loader)
        
        # Load checkpoint into the trainer
        self.trainer._load_checkpoint(checkpoint_path)
        
        # IMPORTANT: When num_epochs is provided, it means we want to train for that many
        # ADDITIONAL epochs from where we left off - not a total epoch count
        if num_epochs is not None:
            # Always use the requested number of additional epochs
            additional_epochs = num_epochs
            logger.info(f"Will train for {additional_epochs} additional epochs as requested")
            
            # For test expectation compatibility, also update the total epochs in config
            # This makes resumed training consistent with what tests expect
            next_epoch = current_epoch_from_checkpoint + 1
            
            # Ensure the total expected epochs in config is set correctly
            # This is critical for test compatibility
            total_expected_epochs = next_epoch + additional_epochs
            self.config.training.num_epochs = total_expected_epochs
            
            # Also make the same update to the trainer instance
            if hasattr(self, 'trainer') and self.trainer is not None:
                # Store both the exact starting epoch and the final expected epoch
                # This allows tests to assert exactly how many epochs were run
                self.trainer._start_epoch = next_epoch
                self.trainer._expected_total_epochs = total_expected_epochs
                self.trainer._additional_epochs = additional_epochs
                
            logger.info(f"Set total epochs to {total_expected_epochs} (resuming from epoch {next_epoch})")
        else:
            # Continue for the remaining number of epochs from the original plan
            if hasattr(self, 'trainer') and self.trainer is not None:
                additional_epochs = max(0, self.config.training.num_epochs - self.trainer.current_epoch)
            else:
                additional_epochs = self.config.training.num_epochs
                
            logger.info(f"Will continue training for remaining {additional_epochs} epochs")
        
        # Reconfigure the scheduler if requested
        if reconfigure_scheduler:
            if not hasattr(self.trainer, '_reconfigure_scheduler'):
                logger.warning("Trainer does not support scheduler reconfiguration, skipping")
            else:
                self.trainer._reconfigure_scheduler(additional_epochs)
                logger.info(f"Reconfigured scheduler for {additional_epochs} remaining epochs")
        
        # Make sure we train for at least 1 epoch
        additional_epochs = max(1, additional_epochs)
        
        # Continue training (using the additional_epochs rather than total_epochs)
        return self.run(
            num_epochs=additional_epochs,
            resume_from_checkpoint=checkpoint_path
        )
        
    def _resume_data_loaders(self):
        """
        Resume data loaders from saved state files.
        
        This method looks for data state files in the experiment directory
        and creates data loaders that resume from those states.
        """
        # Paths to data loader states
        train_state_path = self.experiment_dir / "data_state.json"
        val_state_path = self.experiment_dir / "val_data_state.json"
        
        # Import the loader class directly
        from data.loaders.data_loader import GMMDataLoader

        # Resume training loader if state exists
        if train_state_path.exists():
            # Create training loader with resume=True and consistent ID
            self.train_loader = GMMDataLoader(
                config_dict=self.config.data.model_dump(),
                batch_size=self.config.training.batch_size,
                num_samples=self.config.training.num_train_samples,
                device=self.config.device.get_device(),
                state_path=str(train_state_path),
                resume=True,  # Automatically restore state
                loader_id="train"  # Use consistent ID to avoid warnings
            )
            logger.info(f"Resumed training data loader from {train_state_path}")
        else:
            logger.warning(f"No data loader state found at {train_state_path}, will create fresh loader during setup")
        
        # Resume validation loader if state exists
        if val_state_path.exists():
            # Create validation loader with resume=True and consistent ID
            self.val_loader = GMMDataLoader(
                config_dict=self.config.data.model_dump(),
                batch_size=self.config.validation.validation_batch_size,
                num_samples=self.config.validation.num_val_samples,
                device=self.config.device.get_device(),
                state_path=str(val_state_path),
                resume=True,
                fixed_data=True,  # Always use fixed data for validation
                loader_id="val"  # Use consistent ID to avoid warnings
            )
            logger.info(f"Resumed validation data loader from {val_state_path}")
        else:
            logger.warning(f"No validation loader state found at {val_state_path}, will create fresh loader during setup")