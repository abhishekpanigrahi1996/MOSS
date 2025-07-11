"""
Generator for GMM data batches with both SNR and MI control.

This module provides the DataGenerator class for generating batches of
Gaussian Mixture Model data for training or evaluation.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List, Union

from .parameter_generator import ParameterGenerator
from ..utils.random_utils import RandomState


class DataGenerator:
    """
    Generates batches of GMM data for training or testing.
    
    This class takes a ParameterGenerator and produces batches of data
    with control over variation within batches.
    
    Attributes
    ----------
    param_generator : ParameterGenerator
        Generator for GMM parameters
    vary_clusters_in_batch : bool
        Whether to vary clusters within a batch
    vary_control_in_batch : bool
        Whether to vary control parameter (SNR/MI) within batch
    
    Examples
    --------
    >>> # Create a parameter generator with SNR control
    >>> from gmm_v2.data.core import ParameterGenerator
    >>> cluster_config = {'type': 'fixed', 'value': 3}
    >>> snr_config = {'type': 'uniform', 'min': 3.0, 'max': 10.0}
    >>> sample_config = {'type': 'fixed', 'value': 1000}
    >>> param_gen = ParameterGenerator(
    ...     dim=2, 
    ...     cluster_config=cluster_config,
    ...     snr_config=snr_config,
    ...     sample_count_config=sample_config,
    ...     seed=42
    ... )
    >>> 
    >>> # Create a data generator
    >>> data_gen = DataGenerator(
    ...     param_generator=param_gen,
    ...     vary_clusters_in_batch=False,
    ...     vary_control_in_batch=True
    ... )
    >>> 
    >>> # Generate a batch of data
    >>> data, labels, params = data_gen.generate_batch(batch_size=2)
    >>> print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    """
    
    def __init__(
        self,
        param_generator: ParameterGenerator,
        vary_clusters_in_batch: bool = False,  # Deprecated, always False
        vary_control_in_batch: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the data generator.
        
        Parameters
        ----------
        param_generator : ParameterGenerator
            Generator for GMM parameters
        vary_clusters_in_batch : bool, optional
            Whether to vary clusters within a batch (deprecated, always False), by default False
        vary_control_in_batch : bool, optional
            Whether to vary control parameter (SNR/MI), by default False
        seed : Optional[int], optional
            Random seed for reproducibility, by default None
        """
        self.param_generator = param_generator
        # Enforce vary_clusters_in_batch=False to prevent tensor dimension errors
        self.vary_clusters_in_batch = False
        self.vary_control_in_batch = vary_control_in_batch
        
        # Show warning if user tried to enable vary_clusters_in_batch
        if vary_clusters_in_batch:
            import warnings
            warnings.warn(
                "vary_clusters_in_batch=True is deprecated and not supported due to "
                "tensor dimension incompatibility. Setting to False."
            )
        
        # Set up random state
        self._setup_random_state(seed)
        
    def _setup_random_state(self, seed: Optional[int] = None) -> None:
        """
        Set up random number generation.
        
        Parameters
        ----------
        seed : Optional[int], optional
            Random seed for initialization, by default None
        """
        # Use the RandomState class for state management
        self.random_state = RandomState(seed)
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with current state
        """
        state = {
            "random_state": self.random_state.get_state(),
            "config": {
                "vary_clusters_in_batch": self.vary_clusters_in_batch,
                "vary_control_in_batch": self.vary_control_in_batch
            }
        }
        
        # Save cached data if available for reproducibility
        if hasattr(self, '_cached_data'):
            # Convert numpy arrays to lists for JSON serialization
            state["cached_data"] = {
                "data": self._cached_data.tolist() if self._cached_data is not None else None,
                "labels": self._cached_labels.tolist() if self._cached_labels is not None else None,
                "batch_size": self._cached_batch_size
            }
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a serialized dictionary.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from get_state
        """
        # Restore random state first for consistent behavior
        if "random_state" in state:
            self.random_state.set_state(state["random_state"])
        
        # Restore configuration if available
        if "config" in state:
            config = state["config"]
            self.vary_control_in_batch = config.get("vary_control_in_batch", self.vary_control_in_batch)
            # vary_clusters_in_batch is always forced to False
            
        # Apply param_generator state if provided from old format for compatibility
        if "param_generator_state" in state:
            self.param_generator.set_state(state["param_generator_state"])
            
        # Restore cached data if available
        if "cached_data" in state:
            cached = state["cached_data"]
            if cached["data"] is not None:
                self._cached_data = np.array(cached["data"])
                self._cached_labels = np.array(cached["labels"])
                self._cached_batch_size = cached["batch_size"]
                # Generate GMM parameters for this data
                self._cached_params = None  # Will be re-generated when needed
    
    def generate_batch(self, batch_size: int, separate_params: bool = True) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Generate a batch of GMM data.
        
        Parameters
        ----------
        batch_size : int
            Number of examples to generate
        separate_params : bool, optional
            Whether to use different params for each example, by default True
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List]
            Tuple of (data, labels, params) where data has shape (batch_size, sample_count, dim)
            and labels has shape (batch_size, sample_count)
        """
        # Get GMM parameters and sample count from parameter generator
        if separate_params:
            gmm_params, sample_count = self.param_generator.generate(
                batch_size=batch_size,
                vary_clusters=self.vary_clusters_in_batch,
                vary_control=self.vary_control_in_batch
            )
        else:
            # Use same parameters for all examples
            gmm_params, sample_count = self.param_generator.generate(
                batch_size=1,
                vary_clusters=False,
                vary_control=False
            )
            # Duplicate the parameters for the batch
            gmm_params = gmm_params * batch_size
        
        # Prepare arrays for data and labels
        dim = self.param_generator.dim
        data = np.zeros((batch_size, sample_count, dim))
        labels = np.full((batch_size, sample_count), -1)
        
        # Generate data for each GMM
        for i in range(batch_size):
            gmm_data, gmm_labels = gmm_params[i].generate_samples(sample_count, self.random_state)
            
            # Store data and labels
            data[i] = gmm_data
            labels[i] = gmm_labels
        
        return data, labels, gmm_params
        
    def generate_training_batch(self, batch_size: int, 
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate a batch of GMM data for training.
        
        Parameters
        ----------
        batch_size : int
            Number of examples to generate
        device : Optional[torch.device], optional
            PyTorch device to place tensors on, by default None
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Tuple of (data_tensor, targets_dict) where data_tensor has shape 
            (batch_size, sample_count, dim) and targets_dict contains 'centers',
            'labels', 'params', and optionally 'mi' for MI-controlled batches
        """
        # Generate data
        batch_data, batch_labels, batch_params = self.generate_batch(
            batch_size=batch_size,
            separate_params=True
        )
        
        # Extract centers
        batch_centers = np.array([param.means for param in batch_params])
        
        # Create tensors
        data_tensor = torch.tensor(batch_data, dtype=torch.float32)
        centers_tensor = torch.tensor(batch_centers, dtype=torch.float32)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.int64)
        
        # Move to device if specified
        if device:
            data_tensor = data_tensor.to(device)
            centers_tensor = centers_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
        
        # Create target dictionary
        targets = {
            'centers': centers_tensor,
            'labels': labels_tensor,
            'params': batch_params
        }
        
        # Add noise std values for metrics
        noise_stds = np.array([param.noise_std for param in batch_params])
        noise_tensor = torch.tensor(noise_stds, dtype=torch.float32)
        if device:
            noise_tensor = noise_tensor.to(device)
        targets['noise_std'] = noise_tensor
        
        # Add SNR values
        snr_values = np.array([param.snr_db for param in batch_params])
        snr_tensor = torch.tensor(snr_values, dtype=torch.float32)
        if device:
            snr_tensor = snr_tensor.to(device)
        targets['snr_db'] = snr_tensor
        
        # Add MI values for MI-controlled batches
        mi_values = []
        has_mi_values = False
        
        for param in batch_params:
            if param.estimated_mi is not None:
                mi_values.append(param.estimated_mi)
                has_mi_values = True
            else:
                # Add placeholder if some params don't have MI
                mi_values.append(0.0)
                
        if has_mi_values:
            mi_tensor = torch.tensor(mi_values, dtype=torch.float32)
            if device:
                mi_tensor = mi_tensor.to(device)
            targets['mi'] = mi_tensor
        
        # Add MI factor values for MI-factor-controlled batches
        mi_factor_values = []
        has_mi_factor_values = False
        
        for param in batch_params:
            if param.mi_factor is not None:
                mi_factor_values.append(param.mi_factor)
                has_mi_factor_values = True
            else:
                # Add placeholder if some params don't have MI factor
                mi_factor_values.append(0.0)
                
        if has_mi_factor_values:
            mi_factor_tensor = torch.tensor(mi_factor_values, dtype=torch.float32)
            if device:
                mi_factor_tensor = mi_factor_tensor.to(device)
            targets['mi_factor'] = mi_factor_tensor
            
        return data_tensor, targets