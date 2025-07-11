"""
Generator for GMM parameters with SNR, MI, and MI factor control capabilities.

This module provides the ParameterGenerator class for generating
parameterized Gaussian Mixture Models with control over the number of clusters,
SNR, MI, MI factor, and sample counts.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

from .gmm_params import GMMParams
from ..utils.random_utils import RandomState
from ..utils.distribution_utils import DistributionSampler


class ParameterGenerator:
    """
    Generator for GMM parameters with SNR, MI, and MI factor control capabilities.
    
    This class handles the generation of GMM parameters including clusters,
    mixture weights, component means, and noise levels.
    
    Attributes
    ----------
    dim : int
        Dimensionality of the data
    cluster_config : Dict[str, Any]
        Configuration for cluster sampling
    snr_config : Optional[Dict[str, Any]]
        Configuration for SNR sampling
    mi_config : Optional[Dict[str, Any]]
        Configuration for MI sampling
    mi_factor_config : Optional[Dict[str, Any]]
        Configuration for MI factor sampling
    sample_count_config : Dict[str, Any]
        Configuration for sample count sampling
    alpha_dirichlet : float
        Concentration parameter for Dirichlet distribution
    control_mode : str
        Control mode, either 'snr', 'mi', or 'mi_factor'
    
    Examples
    --------
    >>> # Create a parameter generator with SNR control
    >>> cluster_config = {'type': 'fixed', 'value': 3}
    >>> snr_config = {'type': 'uniform', 'min': 3.0, 'max': 10.0}
    >>> sample_config = {'type': 'fixed', 'value': 1000}
    >>> gen = ParameterGenerator(
    ...     dim=2, 
    ...     cluster_config=cluster_config,
    ...     snr_config=snr_config,
    ...     sample_count_config=sample_config,
    ...     seed=42
    ... )
    >>> params, counts = gen.generate(batch_size=2)
    >>> print(f"Batch size: {len(params)}, Sample count: {counts[0]}")
    
    >>> # Create a parameter generator with MI factor control
    >>> cluster_config = {'type': 'fixed', 'value': 3}
    >>> mi_factor_config = {'type': 'uniform', 'min': 0.0, 'max': 1.0}
    >>> sample_config = {'type': 'fixed', 'value': 1000}
    >>> gen = ParameterGenerator(
    ...     dim=2, 
    ...     cluster_config=cluster_config,
    ...     mi_factor_config=mi_factor_config,
    ...     sample_count_config=sample_config,
    ...     seed=42
    ... )
    >>> params, counts = gen.generate(batch_size=2)
    >>> print(f"Batch size: {len(params)}, Sample count: {counts[0]}")
    """
    
    def __init__(
        self,
        dim: int = 3,
        cluster_config: Dict[str, Any] = None,
        # Control configuration - specify one of these
        snr_config: Optional[Dict[str, Any]] = None,
        mi_config: Optional[Dict[str, Any]] = None,
        mi_factor_config: Optional[Dict[str, Any]] = None,
        # Sample count configuration
        sample_count_config: Dict[str, Any] = None,
        # Other parameters
        alpha_dirichlet: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the parameter generator.
        
        Parameters
        ----------
        dim : int, optional
            Dimensionality of the data, by default 3
        cluster_config : Dict[str, Any], optional
            Configuration for clusters, by default None
        snr_config : Optional[Dict[str, Any]], optional
            Configuration for SNR control, by default None
        mi_config : Optional[Dict[str, Any]], optional
            Configuration for MI control, by default None
        mi_factor_config : Optional[Dict[str, Any]], optional
            Configuration for MI factor control, by default None
        sample_count_config : Dict[str, Any], optional
            Configuration for sample counts, by default None
        alpha_dirichlet : float, optional
            Concentration parameter for Dirichlet distribution, by default 1.0
        seed : Optional[int], optional
            Random seed for reproducibility, by default None
            
        Raises
        ------
        ValueError
            If all of snr_config, mi_config, and mi_factor_config are None
        """
        self.dim = dim
        self.alpha_dirichlet = alpha_dirichlet
        
        # Determine control mode
        if snr_config is None and mi_config is None and mi_factor_config is None:
            raise ValueError("At least one of snr_config, mi_config, or mi_factor_config must be provided")
        
        # Default control mode priority: mi_factor > mi > snr
        if mi_factor_config is not None:
            self.control_mode = 'mi_factor'
        elif mi_config is not None:
            self.control_mode = 'mi'
        else:
            self.control_mode = 'snr'
            
        self.snr_config = snr_config
        self.mi_config = mi_config
        self.mi_factor_config = mi_factor_config
        
        # Set up random state
        self._setup_random_state(seed)
        
        # Set default configurations if not provided
        self._setup_default_configs(cluster_config, sample_count_config)
        
        # Cache for parameter sampling
        self._cached_n_clusters = None
        self._cached_snr_db = None
        self._cached_mi_target = None
        self._cached_mi_factor = None
        
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
        
    def _setup_default_configs(self, 
                             cluster_config: Optional[Dict[str, Any]] = None,
                             sample_count_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up default configurations for parameters.
        
        Parameters
        ----------
        cluster_config : Optional[Dict[str, Any]], optional
            Configuration for clusters, by default None
        sample_count_config : Optional[Dict[str, Any]], optional
            Configuration for sample counts, by default None
        """
        # Default cluster config: 3-5 clusters
        default_cluster_config = {
            'type': 'range',
            'min': 3,
            'max': 5
        }
        
        # Default sample count config: 1000 samples
        default_sample_count_config = {
            'type': 'fixed',
            'value': 1000
        }
        
        # Use defaults if not provided
        self.cluster_config = cluster_config if cluster_config is not None else default_cluster_config
        self.sample_count_config = sample_count_config if sample_count_config is not None else default_sample_count_config
        
        # Validate that the configurations have 'type' field
        for config_name, config in [
            ('cluster_config', self.cluster_config),
            ('sample_count_config', self.sample_count_config)
        ]:
            if 'type' not in config:
                raise ValueError(f"{config_name} must include 'type' field")
    
    def _sample_n_clusters(self, vary: bool = True) -> int:
        """
        Sample number of clusters based on configuration.
        
        Parameters
        ----------
        vary : bool, optional
            Whether to vary clusters or use cached value, by default True
            
        Returns
        -------
        int
            Number of clusters
        """
        if not vary and self._cached_n_clusters is not None:
            return self._cached_n_clusters
        
        n_clusters = DistributionSampler.sample(self.cluster_config, self.random_state.rng)
        
        # Ensure we have an integer value for n_clusters
        n_clusters = int(round(n_clusters))
        
        # Ensure at least one cluster
        if n_clusters < 1:
            n_clusters = 1
            
        # Cache if not varying
        if not vary:
            self._cached_n_clusters = n_clusters
            
        return n_clusters
    
    def _sample_snr_db(self, vary: bool = True) -> float:
        """
        Sample SNR in dB based on configuration.
        
        Parameters
        ----------
        vary : bool, optional
            Whether to vary SNR or use cached value, by default True
            
        Returns
        -------
        float
            SNR in dB
            
        Raises
        ------
        ValueError
            If SNR sampling is requested but no configuration is provided
        """
        if not vary and self._cached_snr_db is not None:
            return self._cached_snr_db
            
        if self.snr_config is None:
            raise ValueError("Cannot sample SNR without configuration")
        
        snr_db = DistributionSampler.sample(self.snr_config, self.random_state.rng)
            
        # Cache if not varying
        if not vary:
            self._cached_snr_db = snr_db
            
        return snr_db
        
    def _sample_mi_target(self, vary: bool = True) -> float:
        """
        Sample MI target based on configuration.
        
        Parameters
        ----------
        vary : bool, optional
            Whether to vary MI or use cached value, by default True
            
        Returns
        -------
        float
            MI target
            
        Raises
        ------
        ValueError
            If MI sampling is requested but no configuration is provided
        """
        if not vary and self._cached_mi_target is not None:
            return self._cached_mi_target
            
        if self.mi_config is None:
            raise ValueError("Cannot sample MI without configuration")
        
        mi_target = DistributionSampler.sample(self.mi_config, self.random_state.rng)
            
        # Cache if not varying
        if not vary:
            self._cached_mi_target = mi_target
            
        return mi_target
    
    def _sample_mi_factor(self, vary: bool = True) -> float:
        """
        Sample MI factor (0-1) based on configuration.
        
        Parameters
        ----------
        vary : bool, optional
            Whether to vary MI factor or use cached value, by default True
            
        Returns
        -------
        float
            MI factor in [0,1]
            
        Raises
        ------
        ValueError
            If MI factor sampling is requested but no configuration is provided
            If valid value can't be sampled after max_attempts
        """
        if not vary and self._cached_mi_factor is not None:
            return self._cached_mi_factor
            
        if self.mi_factor_config is None:
            raise ValueError("Cannot sample MI factor without configuration")
        
        # Try to sample a valid MI factor (in [0,1])
        max_attempts = 10
        for _ in range(max_attempts):
            mi_factor = DistributionSampler.sample(self.mi_factor_config, self.random_state.rng)
            
            # Check if value is valid
            if 0.0 <= mi_factor <= 1.0:
                # Cache if not varying
                if not vary:
                    self._cached_mi_factor = mi_factor
                return mi_factor
        
        # If we reach here, we couldn't sample a valid value
        # As a fallback, return a clipped value with a warning
        mi_factor = DistributionSampler.sample(self.mi_factor_config, self.random_state.rng)
        mi_factor = np.clip(mi_factor, 0.0, 1.0)
        
        import warnings
        warnings.warn(
            f"Failed to sample valid MI factor in [0,1] after {max_attempts} attempts. "
            f"Using clipped value: {mi_factor}. Check mi_factor_config."
        )
        
        # Cache if not varying
        if not vary:
            self._cached_mi_factor = mi_factor
            
        return mi_factor
    
    def _sample_count(self) -> int:
        """
        Sample the number of data points based on configuration.
        
        Returns
        -------
        int
            Number of samples
        """
        sample_count = DistributionSampler.sample(self.sample_count_config, self.random_state.rng)
        
        # Ensure positive sample count
        if sample_count <= 0:
            sample_count = 1
            
        return int(sample_count)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with the current state
        """
        state = {
            "random_state": self.random_state.get_state(),
            "cached_n_clusters": self._cached_n_clusters,
            "cached_snr_db": self._cached_snr_db,
            "cached_mi_target": self._cached_mi_target,
            "cached_mi_factor": self._cached_mi_factor
        }
        
        # Store the last batch of varying parameters if available
        if hasattr(self, '_last_batch_snr_values'):
            state["last_batch_snr_values"] = self._last_batch_snr_values
        if hasattr(self, '_last_batch_mi_values'):
            state["last_batch_mi_values"] = self._last_batch_mi_values
        if hasattr(self, '_last_batch_mi_factor_values'):
            state["last_batch_mi_factor_values"] = self._last_batch_mi_factor_values
            
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a serialized dictionary.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from get_state
        """
        if "random_state" in state:
            self.random_state.set_state(state["random_state"])
        
        self._cached_n_clusters = state.get("cached_n_clusters")
        self._cached_snr_db = state.get("cached_snr_db")
        self._cached_mi_target = state.get("cached_mi_target")
        self._cached_mi_factor = state.get("cached_mi_factor")
    
    def generate(self, batch_size: int = 1, 
                vary_clusters: bool = False,  # Changed default, deprecated parameter
                vary_control: bool = True) -> Tuple[List[GMMParams], int]:
        """
        Generate a batch of GMM parameters with sample count.
        
        Parameters
        ----------
        batch_size : int, optional
            Number of parameter sets to generate, by default 1
        vary_clusters : bool, optional
            Whether to vary clusters within the batch (deprecated, always False), by default False
        vary_control : bool, optional
            Whether to vary the control parameter (SNR, MI, or MI factor), by default True
            
        Returns
        -------
        Tuple[List[GMMParams], int]
            Tuple of (List[GMMParams], sample_count) - parameters and uniform sample count for the batch
        """
        # Force vary_clusters to be False to prevent tensor dimension errors
        vary_clusters = False
        
        # Always reset all cached values to ensure new random samples for each batch
        # This fixes the bug where random values would be sampled once and reused
        self._cached_n_clusters = None
        self._cached_snr_db = None
        self._cached_mi_target = None
        self._cached_mi_factor = None
            
        batch_params = []
        
        # Sample single count for the entire batch
        sample_count = self._sample_count()
        
        for _ in range(batch_size):
            # Sample clusters, weights, means
            n_clusters = self._sample_n_clusters(vary_clusters)
            weights = self.random_state.rng.dirichlet(np.ones(n_clusters) * self.alpha_dirichlet)
            means = self.random_state.rng.standard_normal(size=(n_clusters, self.dim))
            
            # Create params based on control mode preference
            if self.control_mode == 'mi_factor' and self.mi_factor_config is not None:
                # Sample MI factor and create params
                mi_factor = self._sample_mi_factor(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    mi_factor=mi_factor,
                    control_mode='mi_factor'
                )
            elif self.control_mode == 'mi' and self.mi_config is not None:
                # Sample MI target and create params
                mi_target = self._sample_mi_target(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    mi_target=mi_target,
                    control_mode='mi'
                )
            elif self.control_mode == 'snr' and self.snr_config is not None:
                # Sample SNR and create params
                snr_db = self._sample_snr_db(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    snr_db=snr_db,
                    control_mode='snr'
                )
            elif self.mi_factor_config is not None:
                # Fall back to MI factor if available
                mi_factor = self._sample_mi_factor(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    mi_factor=mi_factor,
                    control_mode='mi_factor'
                )
            elif self.mi_config is not None:
                # Fall back to MI if available
                mi_target = self._sample_mi_target(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    mi_target=mi_target,
                    control_mode='mi'
                )
            elif self.snr_config is not None:
                # Fall back to SNR if available
                snr_db = self._sample_snr_db(vary_control)
                
                params = GMMParams(
                    dim=self.dim,
                    n_clusters=n_clusters,
                    weights=weights,
                    means=means,
                    snr_db=snr_db,
                    control_mode='snr'
                )
            else:
                raise ValueError("No valid control configuration available")
                
            batch_params.append(params)
            
        return batch_params, sample_count