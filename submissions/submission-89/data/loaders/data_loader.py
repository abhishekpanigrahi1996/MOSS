"""
Data loader for GMM data with support for dynamic and fixed data generation.

This module provides the GMMDataLoader class, which supports both dynamic 
data generation and fixed data generation with serializable state.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from ..core import DataGenerator
from .factory import create_generators, save_state, load_state


class GMMDataLoader:
    """
    Self-contained DataLoader with SNR and MI control support.
    
    This class provides an iterator interface for generating GMM data
    for training or evaluation, with support for both dynamic data
    generation and fixed data generation.
    
    Attributes
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary for GMM parameters
    batch_size : int
        Batch size for data generation
    num_samples : int
        Samples per epoch
    device : Optional[torch.device]
        Device for tensor placement
    seed : Optional[int]
        Seed for reproducibility
    loader_id : str
        Unique ID for this loader
    fixed_data : bool
        If True, generate all data upfront
    state_path : Optional[str]
        Path to state file for reproducibility
    
    Examples
    --------
    >>> # Create a data loader with dynamic data generation
    >>> config = {
    ...    "dim": 2,
    ...    "control_mode": "snr",
    ...    "cluster_params": {"type": "fixed", "value": 3},
    ...    "snr_db_params": {"type": "uniform", "min": 3.0, "max": 10.0},
    ...    "sample_count_distribution": {"type": "fixed", "value": 1000},
    ...    "vary_parameter_in_batch": True
    ... }
    >>> loader = GMMDataLoader(
    ...     config_dict=config,
    ...     batch_size=16,
    ...     num_samples=1000,
    ...     seed=42
    ... )
    >>> 
    >>> # Use in a training loop
    >>> for epoch in range(3):
    ...     for data, targets in loader:
    ...         # Use data and targets for training
    ...         pass
    """
    
    def __init__(self, config_dict: Dict[str, Any], 
                batch_size: int = 16, 
                num_samples: int = 1000, 
                device: Optional[torch.device] = None, 
                base_seed: Optional[int] = None,
                loader_id: Optional[str] = None, 
                fixed_data: bool = False,
                state_path: Optional[str] = None,
                resume: bool = False):
        """
        Initialize with configuration.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary for GMM parameters
        batch_size : int, optional
            Batch size for data generation, by default 16
        num_samples : int, optional
            Samples per epoch, by default 1000
        device : Optional[torch.device], optional
            Device for tensor placement, by default None
        base_seed : Optional[int], optional
            Base seed for reproducibility. Combined with loader_id to derive a unique seed
            for this specific loader, by default None
        loader_id : Optional[str], optional
            Unique ID for this loader, by default None
        fixed_data : bool, optional
            If True, generate all data upfront, by default False
        state_path : Optional[str], optional
            Path to state file for reproducibility, by default None
        resume : bool, optional
            If True and state_path is provided, resume from the saved state.
            If False (default), initialize with seed even if state_path exists.
        
        Notes
        -----
        The seeding approach uses `base_seed` with a specific `loader_id`, which gives
        consistent results when using multiple related loaders.
        
        When both `state_path` and seeding are provided, the behavior depends on the 
        `resume` parameter:
        
        - If `resume=True`: Load and restore state from the file, ignoring the seed.
          Continue generating data from the point where the state was saved.
        
        - If `resume=False` (default): Initialize using the seed. The state_path is
          only used for saving state, not loading initially.
        """
        self.config_dict = config_dict
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.base_seed = base_seed
        self.loader_id = loader_id or f"loader_{id(self)}"
        self.fixed_data = fixed_data
        self.state_path = state_path
        self.resume = resume
        
        # Derive operational seed
        if base_seed is not None:
            import hashlib
            hashed_id = int(hashlib.md5(str(self.loader_id).encode()).hexdigest(), 16) % 1000000
            self.seed = base_seed + hashed_id
        else:
            self.seed = None
        
        # Validate state_path when resume=True
        if resume and state_path is None:
            raise ValueError("state_path must be provided when resume=True")
        
        # Tracking variables
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = (num_samples + batch_size - 1) // batch_size
        
        # Flag to indicate if the loader state was restored and is resuming
        self._resumed = False
        
        # Create generators based on configuration or state
        self._initialize_generators()
    
    def _initialize_generators(self) -> None:
        """
        Initialize parameter and data generators.
        
        This method creates generators with optional state loading.
        """
        # Prepare configuration with seed
        config = self.config_dict.copy()
        if self.seed is not None:
            config["random_seed"] = self.seed
        
        # Load state if resuming
        loaded_state = None
        if self.resume and self.state_path and os.path.exists(self.state_path):
            # Load state from file
            loaded_state = load_state(self.state_path)
            # Extract and restore loader state
            if "loader" in loaded_state:
                self._restore_loader_state(loaded_state["loader"])
        
        # Create generators with or without state
        self.param_generator, self.data_generator = create_generators(
            config, 
            state=loaded_state
        )
    
    def _generate_next_batch(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next batch of data.
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Tuple of (data_tensor, targets_dict)
        """
        # Calculate correct batch size for the current batch
        samples_left = self.num_samples - (self.current_batch * self.batch_size)
        current_batch_size = min(self.batch_size, samples_left)
        
        # Generate batch using our data generator
        data, targets = self.data_generator.generate_training_batch(
            batch_size=current_batch_size,
            device=self.device
        )
        
        return data, targets
    
    def save_state(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current state for reproducibility.
        
        Parameters
        ----------
        path : Optional[str], optional
            Path to save state to, by default self.state_path
            
        Returns
        -------
        Dict[str, Any]
            State dictionary that was saved
        """
        save_path = path or self.state_path
        if save_path is None:
            save_path = f"gmm_loader_state_{self.loader_id}.json"
        
        # Create loader state dictionary
        # Important: we save current_batch as the current position, which means
        # when we resume, the next batch will be the one after this position
        loader_state = {
            "loader": {
                "current_epoch": self.current_epoch,
                "current_batch": self.current_batch,
                "loader_id": self.loader_id,
                "fixed_data": self.fixed_data,
                "base_seed": self.base_seed,
                "seed": self.seed
            }
        }        
        # Use the new save_state function
        return save_state(
            self.param_generator,
            self.data_generator,
            save_path,
            additional_state=loader_state
        )
    
    def _restore_loader_state(self, loader_state: Dict[str, Any]) -> None:
        """
        Restore loader-specific state.
        
        Parameters
        ----------
        loader_state : Dict[str, Any]
            Loader state dictionary
        """
        # Restore batch position
        self.current_epoch = loader_state.get("current_epoch", 0)
        self.current_batch = loader_state.get("current_batch", 0)
        # Restored position - epoch: {self.current_epoch}, batch: {self.current_batch}
        
        # Restore seed information
        if "seed" in loader_state:
            self.seed = loader_state["seed"]
        
        if "base_seed" in loader_state:
            self.base_seed = loader_state["base_seed"]
        
        # Validate fixed_data setting
        stored_fixed_data = loader_state.get("fixed_data")
        if stored_fixed_data is not None and stored_fixed_data != self.fixed_data:
            import warnings
            warnings.warn(
                f"Mismatch in fixed_data setting: stored={stored_fixed_data}, "
                f"current={self.fixed_data}. This may affect reproducibility."
            )
        
        # Validate loader_id
        stored_loader_id = loader_state.get("loader_id")
        if stored_loader_id is not None and stored_loader_id != self.loader_id:
            import warnings
            warnings.warn(
                f"Mismatch in loader_id: stored={stored_loader_id}, "
                f"current={self.loader_id}. This may affect reproducibility."
            )
        
        # Mark as resumed
        self._resumed = True
    
    def reset(self, reset_counters: bool = True) -> None:
        """
        Reset the loader to its initial state.
        
        Parameters
        ----------
        reset_counters : bool, optional
            Whether to reset batch/epoch counters, by default True
        """
        # Reset resume flag
        self._resumed = False
        
        # Reset counters if requested
        if reset_counters:
            self.current_epoch = 0
            self.current_batch = 0
        
        # Re-initialize generators with the seed
        if self.seed is not None:
            self._initialize_generators()
    
    def __iter__(self) -> 'GMMDataLoader':
        """
        Prepare for iteration and return self as iterator.
        
        For fixed_data=True:
        - Creates generators with the initial seed for each iteration
        - When resuming, preserves the batch counter position
        
        For fixed_data=False:
        - Maintains random progression across iterations
        - When resuming, continues from the saved state
        
        Returns
        -------
        GMMDataLoader
            Self for iteration
        """
        # Store whether we're resuming from a saved state
        is_resuming = self._resumed
        
        # Reset batch counter when needed:
        # 1. At the end of an epoch
        # 2. When not resuming (starting fresh)
        if self.current_batch >= self.batches_per_epoch:
            # End of epoch - always reset position
            self.current_batch = 0
        elif not is_resuming:
            # Not resuming - start from beginning of epoch
            self.current_batch = 0
        
        # For fixed data mode:
        # We always recreate generators with the seed UNLESS resuming
        if self.fixed_data and (self.current_batch == 0 or not is_resuming):
            # Reset the generators to the initial state for deterministic data
            self._initialize_generators()
        
        # Reset the resume flag for the next iteration
        self._resumed = False
        
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get the next batch.
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Tuple of (data_tensor, targets_dict)
            
        Raises
        ------
        StopIteration
            When the epoch is complete
        """
        # Check if we've completed an epoch
        if self.current_batch >= self.batches_per_epoch:
            # Reset batch counter and increment epoch counter
            self.current_batch = 0
            self.current_epoch += 1
            
            # Signal end of iteration
            raise StopIteration
        
        # Generate a batch
        batch = self._generate_next_batch()
        
        # Increment batch counter
        self.current_batch += 1
        
        return batch
    
    def __len__(self) -> int:
        """
        Get the number of batches per epoch.
        
        Returns
        -------
        int
            Number of batches per epoch
        """
        return self.batches_per_epoch
        
    @staticmethod
    def get_state_path(state_dir: Optional[str], loader_id: str) -> Optional[str]:
        """
        Generate a standard state path from directory and loader ID.
        
        Parameters
        ----------
        state_dir : Optional[str]
            Directory to store state files, or None if state saving is not needed
        loader_id : str
            Unique ID for the loader
            
        Returns
        -------
        Optional[str]
            Full path to the state file, or None if state_dir is None
        """
        if state_dir is None:
            return None
            
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, f"{loader_id}_state.json")
    
    @staticmethod
    def create_train_val_pair(
        config_dict: Dict[str, Any],
        train_batch_size: int = 16,
        val_batch_size: int = 32,
        train_samples: int = 1000,
        val_samples: int = 500,
        device: Optional[torch.device] = None,
        base_seed: Optional[int] = None,
        fixed_data: bool = False,
        fixed_validation_data: bool = False,
        state_dir: Optional[str] = None,
        resume: bool = False) -> Tuple['GMMDataLoader', 'GMMDataLoader']:
        """
        Create a pair of train and validation loaders with consistent seeding.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary for GMM parameters
        train_batch_size : int, optional
            Batch size for training data, by default 16
        val_batch_size : int, optional
            Batch size for validation data, by default 32
        train_samples : int, optional
            Number of samples per epoch for training, by default 1000
        val_samples : int, optional
            Number of samples per epoch for validation, by default 500
        device : Optional[torch.device], optional
            Device for tensor placement, by default None
        base_seed : Optional[int], optional
            Base seed for reproducibility, by default None
        fixed_data : bool, optional
            If True, use fixed data for both train and val, by default False
        fixed_validation_data : bool, optional
            If True, use fixed data for validation only, by default False
        state_dir : Optional[str], optional
            Directory to store state files, by default None
        resume : bool, optional
            If True, attempt to resume from saved state files, by default False
            
        Returns
        -------
        Tuple[GMMDataLoader, GMMDataLoader]
            Tuple of (train_loader, val_loader)
        """
        # Create train loader
        train_state_path = GMMDataLoader.get_state_path(state_dir, "train")
        # Use explicit resume parameter - no fallbacks
        resume_train = resume and train_state_path is not None
        train_loader = GMMDataLoader(
            config_dict=config_dict,
            batch_size=train_batch_size,
            num_samples=train_samples,
            device=device,
            base_seed=base_seed,
            loader_id="train",
            fixed_data=fixed_data,
            state_path=train_state_path,
            resume=resume_train
        )
        
        # Create validation loader
        val_state_path = GMMDataLoader.get_state_path(state_dir, "val")
        # Use explicit resume parameter - no fallbacks
        resume_val = resume and val_state_path is not None
        val_loader = GMMDataLoader(
            config_dict=config_dict,
            batch_size=val_batch_size,
            num_samples=val_samples,
            device=device,
            base_seed=base_seed,
            loader_id="val",
            fixed_data=fixed_data or fixed_validation_data,
            state_path=val_state_path,
            resume=resume_val
        )
        
        return train_loader, val_loader