"""
Parameters for Gaussian Mixture Models with SNR, MI, and MI factor control.

This module provides a data class for representing GMM parameters,
supporting SNR-based, MI-based, and MI factor-based noise control.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from .noise_controller import NoiseController
from ..utils.random_utils import RandomState


@dataclass
class GMMParams:
    """
    Parameters for a Gaussian Mixture Model.
    
    Supports SNR-based, MI-based, and MI factor-based noise control.
    
    Attributes
    ----------
    dim : int
        Dimensionality of the data
    n_clusters : int
        Number of clusters in the mixture
    weights : np.ndarray
        Mixture weights, shape (K,)
    means : np.ndarray
        Mixture component means, shape (K, D)
    snr_db : Optional[float]
        Signal-to-noise ratio in dB (for SNR control mode)
    mi_target : Optional[float]
        Target mutual information value (for MI control mode)
    mi_factor : Optional[float]
        Normalized MI factor in [0,1] (for MI factor control mode)
    noise_std : Optional[float]
        Noise standard deviation (derived parameter)
    estimated_mi : Optional[float]
        Estimated mutual information (derived parameter)
    control_mode : str
        Control mode, either 'snr', 'mi', or 'mi_factor'
    
    Examples
    --------
    >>> # SNR control mode
    >>> weights = np.array([0.3, 0.7])
    >>> means = np.array([[0, 0], [1, 1]])
    >>> params = GMMParams(dim=2, n_clusters=2, weights=weights, means=means, 
    ...                   snr_db=5.0, control_mode='snr')
    >>> print(f"Noise std: {params.noise_std:.4f}")
    
    >>> # MI control mode
    >>> params = GMMParams(dim=2, n_clusters=2, weights=weights, means=means,
    ...                   mi_target=1.0, control_mode='mi')
    >>> print(f"Noise std: {params.noise_std:.4f}")
    
    >>> # MI factor control mode
    >>> params = GMMParams(dim=2, n_clusters=2, weights=weights, means=means,
    ...                   mi_factor=0.5, control_mode='mi_factor')
    >>> print(f"Noise std: {params.noise_std:.4f}")
    """
    dim: int
    n_clusters: int
    weights: np.ndarray
    means: np.ndarray
    
    # Control parameters
    snr_db: Optional[float] = None
    mi_target: Optional[float] = None
    mi_factor: Optional[float] = None
    
    # Derived parameters
    noise_std: Optional[float] = None
    estimated_mi: Optional[float] = None
    
    # Tracking info
    control_mode: str = 'snr'  # 'snr', 'mi', or 'mi_factor'
    
    def __post_init__(self) -> None:
        """
        Calculate dependent parameters based on control mode.
        
        This method runs automatically after initialization to
        calculate noise level and other derived parameters.
        """
        # Validate inputs
        if self.weights.shape[0] != self.n_clusters:
            raise ValueError(f"Weights length ({self.weights.shape[0]}) must match n_clusters ({self.n_clusters})")
        
        if self.means.shape[0] != self.n_clusters:
            raise ValueError(f"Means rows ({self.means.shape[0]}) must match n_clusters ({self.n_clusters})")
            
        if self.means.shape[1] != self.dim:
            raise ValueError(f"Means columns ({self.means.shape[1]}) must match dim ({self.dim})")
            
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        # Calculate derived parameters
        if self.control_mode not in ['snr', 'mi', 'mi_factor']:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
            
        if self.noise_std is None:
            if self.control_mode == 'snr' and self.snr_db is not None:
                # Calculate noise_std based on SNR
                result = NoiseController.calculate_noise_for_snr(
                    weights=self.weights, 
                    means=self.means, 
                    snr_db=self.snr_db
                )
                self.noise_std = result['sigma']
                
                # Also estimate MI for compatibility
                result_mi = NoiseController.estimate_mi_for_noise(
                    weights=self.weights,
                    means=self.means,
                    noise_std=self.noise_std
                )
                self.estimated_mi = result_mi['mi_estimate']
                
                # Calculate equivalent MI factor
                self.mi_factor = NoiseController.absolute_to_mi_factor(
                    self.estimated_mi, self.weights
                )
                
            elif self.control_mode == 'mi' and self.mi_target is not None:
                # Calculate noise_std based on MI target
                result = NoiseController.calculate_noise_for_mi_target(
                    weights=self.weights,
                    means=self.means,
                    mi_target=self.mi_target
                )
                self.noise_std = result['sigma']
                self.estimated_mi = result['mi_estimate']
                
                # Calculate equivalent SNR and MI factor for compatibility
                self.snr_db = NoiseController.calculate_snr_from_noise(
                    weights=self.weights,
                    means=self.means,
                    noise_std=self.noise_std
                )
                
                self.mi_factor = NoiseController.absolute_to_mi_factor(
                    self.estimated_mi, self.weights
                )
            
            elif self.control_mode == 'mi_factor' and self.mi_factor is not None:
                # Ensure mi_factor is in [0,1] by clipping
                if not 0 <= self.mi_factor <= 1:
                    original_mi_factor = self.mi_factor
                    self.mi_factor = np.clip(self.mi_factor, 0.0, 1.0)
                    import warnings
                    warnings.warn(f"MI factor {original_mi_factor} clipped to {self.mi_factor}")
                
                # Calculate noise_std based on MI factor
                result = NoiseController.calculate_noise_for_mi_factor(
                    weights=self.weights,
                    means=self.means,
                    mi_factor=self.mi_factor
                )
                self.noise_std = result['sigma']
                self.estimated_mi = result['mi_estimate']
                
                # Also calculate equivalent SNR for compatibility
                self.snr_db = NoiseController.calculate_snr_from_noise(
                    weights=self.weights,
                    means=self.means,
                    noise_std=self.noise_std
                )
                
                # Calculate absolute MI target from factor
                self.mi_target = NoiseController.mi_factor_to_absolute(
                    self.mi_factor, self.weights
                )
            else:
                raise ValueError("Either snr_db, mi_target, or mi_factor must be provided based on control_mode")
        else:
            # If noise_std is directly provided, calculate other parameters
            # Estimate MI
            result_mi = NoiseController.estimate_mi_for_noise(
                weights=self.weights,
                means=self.means,
                noise_std=self.noise_std
            )
            self.estimated_mi = result_mi['mi_estimate']
            
            # Calculate MI factor
            self.mi_factor = NoiseController.absolute_to_mi_factor(
                self.estimated_mi, self.weights
            )
            
            # Set mi_target for compatibility if in mi control mode
            if self.control_mode == 'mi':
                self.mi_target = self.estimated_mi
            
            # Calculate SNR
            self.snr_db = NoiseController.calculate_snr_from_noise(
                weights=self.weights,
                means=self.means,
                noise_std=self.noise_std
            )
    
    def generate_samples(self, n_samples: int = 1000, 
                        random_state: Optional[RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from this GMM.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 1000
        random_state : Optional[RandomState], optional
            Random state for reproducibility, by default None
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (data, labels) where data has shape (n_samples, dim)
            and labels has shape (n_samples,)
            
        Raises
        ------
        ValueError
            If noise_std is not set
        """
        if self.noise_std is None:
            raise ValueError("Cannot generate data: noise_std is not set")
        
        # Handle random state
        if random_state is None:
            random_state = RandomState()
        
        # Save initial state for reproducibility before any operations
        initial_state = random_state.get_state()
            
        # Generate samples per cluster
        samples_per_cluster = random_state.rng.multinomial(n_samples, self.weights)
        
        data = []
        labels = []
        
        for k in range(self.n_clusters):
            n_k = samples_per_cluster[k]
            if n_k > 0:
                # Generate samples from this cluster
                cluster_samples = random_state.rng.normal(
                    loc=self.means[k], 
                    scale=self.noise_std, 
                    size=(n_k, self.dim)
                )
                data.append(cluster_samples)
                labels.append(np.full(n_k, k))
        
        # Combine and shuffle
        if data:
            data = np.vstack(data)
            labels = np.concatenate(labels)
            
            # Use permutation for shuffling with reproducible index generation
            indices = random_state.rng.permutation(len(data))
            data = data[indices]
            labels = labels[indices]
        else:
            data = np.empty((0, self.dim))
            labels = np.empty(0, dtype=int)
        
        return data, labels
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the parameters
        """
        return {
            'dim': self.dim,
            'n_clusters': self.n_clusters,
            'weights': self.weights.tolist(),
            'means': self.means.tolist(),
            'snr_db': self.snr_db,
            'mi_target': self.mi_target,
            'mi_factor': self.mi_factor,
            'noise_std': self.noise_std,
            'estimated_mi': self.estimated_mi,
            'control_mode': self.control_mode
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GMMParams':
        """
        Create parameters from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with parameter values
            
        Returns
        -------
        GMMParams
            Initialized parameters
        """
        # Convert lists back to numpy arrays
        weights = np.array(data['weights'])
        means = np.array(data['means'])
        
        return cls(
            dim=data['dim'],
            n_clusters=data['n_clusters'],
            weights=weights,
            means=means,
            snr_db=data.get('snr_db'),
            mi_target=data.get('mi_target'),
            mi_factor=data.get('mi_factor'),
            noise_std=data.get('noise_std'),
            estimated_mi=data.get('estimated_mi'),
            control_mode=data.get('control_mode', 'snr')
        )