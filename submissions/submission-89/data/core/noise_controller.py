"""
Utilities for calculating noise levels based on different control modes.

This module provides functions to calculate and convert between different
noise parameters (SNR, MI, and MI factor) for Gaussian Mixture Models.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from ..utils.random_utils import RandomState


class NoiseController:
    """
    Utilities for calculating noise levels based on different control modes.
    
    This class provides static methods for calculating and converting
    between different noise parameters (SNR, MI, MI factor) for Gaussian Mixture Models.
    
    Examples
    --------
    >>> weights = np.array([0.3, 0.7])
    >>> means = np.array([[0, 0], [1, 1]])
    >>> # Calculate noise level for SNR = 5dB
    >>> result = NoiseController.calculate_noise_for_snr(weights, means, 5.0)
    >>> sigma = result['sigma']
    >>> # Calculate noise level for MI factor = 0.5 (50% of max possible MI)
    >>> result = NoiseController.calculate_noise_for_mi_factor(weights, means, 0.5)
    >>> sigma = result['sigma']
    """
    
    @staticmethod
    def calculate_noise_for_snr(weights: np.ndarray, means: np.ndarray, snr_db: float) -> Dict[str, Any]:
        """
        Calculate the noise level (sigma) for a given SNR value.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        snr_db : float
            Target signal-to-noise ratio in dB
            
        Returns
        -------
        Dict[str, Any]
            Results including 'sigma' and 'signal_variance'
        """
        # Calculate signal variance
        signal_var = NoiseController._calculate_signal_variance(weights, means)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise variance from SNR
        noise_var = signal_var / snr_linear
        
        # Return the calculated sigma
        sigma = np.sqrt(noise_var)
        return {
            'sigma': sigma,
            'signal_variance': signal_var
        }
    
    @staticmethod
    def calculate_noise_for_mi_target(weights: np.ndarray, means: np.ndarray, mi_target: float,
                                    sigma_low: float = 0.01, sigma_high: float = 10.0) -> Dict[str, Any]:
        """
        Find the noise level (sigma) that achieves a target mutual information.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        mi_target : float
            Target mutual information value
        sigma_low : float, optional
            Lower bound for sigma, by default 0.01
        sigma_high : float, optional
            Upper bound for sigma, by default 10.0
            
        Returns
        -------
        Dict[str, Any]
            Results including 'sigma' and 'mi_estimate'
        """
        # Import here to avoid circular imports
        from mi.estimation.adaptive_estimator import AdaptiveMIEstimator
        from mi.inverse_problem.bisection_solver import find_sigma_for_target_mi
        
        # Convert to torch tensors
        p_tensor = torch.tensor(weights, dtype=torch.float32)
        mus_tensor = torch.tensor(means, dtype=torch.float32)
        
        # Create estimator
        estimator = AdaptiveMIEstimator(p_tensor, mus_tensor)
        return find_sigma_for_target_mi(
            estimator=estimator,
            target_mi=mi_target,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            verbose=False
        )
    
    @staticmethod
    def estimate_mi_for_noise(weights: np.ndarray, means: np.ndarray, noise_std: float) -> Dict[str, Any]:
        """
        Estimate the mutual information for a given noise level.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        noise_std : float
            Noise standard deviation
            
        Returns
        -------
        Dict[str, Any]
            Results including 'mi_estimate'
        """
        # Import here to avoid circular imports
        from mi.estimation.adaptive_estimator import AdaptiveMIEstimator
        
        # Convert to torch tensors
        p_tensor = torch.tensor(weights, dtype=torch.float32)
        mus_tensor = torch.tensor(means, dtype=torch.float32)
        
        # Create estimator
        estimator = AdaptiveMIEstimator(p_tensor, mus_tensor)
        
        # Estimate MI for given sigma
        mi_estimate = estimator.estimate_mutual_information(
            sigma=noise_std,
            n_samples=10000,
            alpha=0.05
        )
        
        return {'mi_estimate': mi_estimate['mi_estimate']}
    
    @staticmethod
    def calculate_snr_from_noise(weights: np.ndarray, means: np.ndarray, noise_std: float) -> float:
        """
        Calculate the SNR value for a given noise level.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        noise_std : float
            Noise standard deviation
            
        Returns
        -------
        float
            SNR value in dB
        """
        # Calculate signal variance
        signal_var = NoiseController._calculate_signal_variance(weights, means)
        
        # Calculate SNR in linear scale
        noise_var = noise_std ** 2
        snr_linear = signal_var / noise_var
        
        # Convert to dB
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
        
    @staticmethod
    def calculate_mi_from_noise(weights: np.ndarray, means: np.ndarray, noise_std: float) -> float:
        """
        Calculate mutual information for a given noise level.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        noise_std : float
            Noise standard deviation
            
        Returns
        -------
        float
            Mutual information value
        """
        result = NoiseController.estimate_mi_for_noise(
            weights=weights,
            means=means,
            noise_std=noise_std
        )
        return result['mi_estimate']
    
    @staticmethod
    def calculate_entropy_of_discrete_distribution(weights: np.ndarray) -> float:
        """
        Calculate the entropy of a discrete distribution.
        
        Parameters
        ----------
        weights : np.ndarray
            Probability vector, shape (K,)
            
        Returns
        -------
        float
            Entropy of the distribution
        """
        # Ensure valid probabilities
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            raise ValueError("Weights must sum to 1.0")
            
        # Entropy = -sum(p_i * log(p_i))
        # Small epsilon (1e-10) added to avoid log(0)
        return -np.sum(weights * np.log(weights + 1e-10))

    @staticmethod
    def calculate_mi_range(weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the achievable MI range [min_mi, max_mi] for a GMM.
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
            
        Returns
        -------
        Tuple[float, float]
            (min_mi, max_mi)
        """
        # Maximum MI = entropy of X
        max_mi = NoiseController.calculate_entropy_of_discrete_distribution(weights)
        
        # Minimum MI is theoretically 0 (as sigma → ∞)
        min_mi = 0.0
        
        return min_mi, max_mi

    @staticmethod
    def mi_factor_to_absolute(mi_factor: float, weights: np.ndarray) -> float:
        """
        Convert normalized MI factor (0-1) to absolute MI value.
        
        Parameters
        ----------
        mi_factor : float
            Normalized MI factor in [0,1]
        weights : np.ndarray
            GMM mixture weights, shape (K,)
            
        Returns
        -------
        float
            Absolute MI value
        """
        mi_factor = np.clip(mi_factor, 0.0, 1.0)
        min_mi, max_mi = NoiseController.calculate_mi_range(weights)
        return min_mi + mi_factor * (max_mi - min_mi)

    @staticmethod
    def absolute_to_mi_factor(mi_absolute: float, weights: np.ndarray) -> float:
        """
        Convert absolute MI value to normalized MI factor (0-1).
        
        Parameters
        ----------
        mi_absolute : float
            Absolute MI value
        weights : np.ndarray
            GMM mixture weights, shape (K,)
            
        Returns
        -------
        float
            Normalized MI factor in [0,1]
        """
        min_mi, max_mi = NoiseController.calculate_mi_range(weights)
        mi_range = max_mi - min_mi
        if mi_range <= 0:
            return 0.0  # Edge case
        
        factor = (mi_absolute - min_mi) / mi_range
        return np.clip(factor, 0.0, 1.0)
    
    @staticmethod
    def calculate_noise_for_mi_factor(weights: np.ndarray, 
                                     means: np.ndarray, 
                                     mi_factor: float,
                                     sigma_low: float = 0.01, 
                                     sigma_high: float = 10.0,
                                     eps: float = 1e-3) -> Dict[str, Any]:
        """
        Find noise level (sigma) for a normalized MI factor (0-1).
        
        Parameters
        ----------
        weights : np.ndarray
            GMM mixture weights, shape (K,)
        means : np.ndarray
            GMM component means, shape (K, D)
        mi_factor : float
            Normalized MI factor in [0,1]
        sigma_low : float, optional
            Lower bound for sigma, by default 0.01
        sigma_high : float, optional
            Upper bound for sigma, by default 10.0
        eps : float, optional
            Threshold for extreme MI factors, by default 1e-3
            
        Returns
        -------
        Dict[str, Any]
            Results including 'sigma' and 'mi_estimate'
        """
        # Ensure mi_factor is in [0,1]
        mi_factor = np.clip(mi_factor, 0.0, 1.0)
        min_mi, max_mi = NoiseController.calculate_mi_range(weights)
        
        # Import for estimation
        from mi.estimation.adaptive_estimator import AdaptiveMIEstimator
        
        # Create estimator for calculating MI
        p_tensor = torch.tensor(weights, dtype=torch.float32)
        mus_tensor = torch.tensor(means, dtype=torch.float32)
        estimator = AdaptiveMIEstimator(p_tensor, mus_tensor)
        
        # Handle extreme MI factors to prevent numerical issues
        if mi_factor < eps:
            # For very small MI factors (near 0), return a very large sigma
            # This corresponds to high noise / low mutual information
            sigma = sigma_high
            
            # Estimate MI at this sigma
            mi_result = estimator.estimate_mutual_information(sigma, n_samples=1000)
            
            return {
                'sigma': sigma,
                'mi_estimate': mi_result['mi_estimate'],
                'mi_factor': mi_factor,
                'mi_range': (min_mi, max_mi)
            }
            
        elif mi_factor > 1.0 - eps:
            # For very high MI factors (near 1), return a very small sigma
            # This corresponds to low noise / high mutual information
            sigma = sigma_low
            
            # Estimate MI at this sigma
            mi_result = estimator.estimate_mutual_information(sigma, n_samples=1000)
            
            return {
                'sigma': sigma,
                'mi_estimate': mi_result['mi_estimate'],
                'mi_factor': mi_factor,
                'mi_range': (min_mi, max_mi)
            }
        
        # For normal MI factors, use the regular process
        # Calculate absolute MI target
        mi_target = NoiseController.mi_factor_to_absolute(mi_factor, weights)
        
        # Import find_sigma_for_target_mi function
        from mi.inverse_problem.bisection_solver import find_sigma_for_target_mi
        
        # Find sigma for this MI target
        result = find_sigma_for_target_mi(
            estimator=estimator,
            target_mi=mi_target,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            verbose=False
        )
        
        # Add MI factor information to the result
        result['mi_factor'] = mi_factor
        result['mi_range'] = (min_mi, max_mi)
        
        return result
    
    @staticmethod
    def _calculate_signal_variance(weights: Union[np.ndarray, torch.Tensor], 
                                 means: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate the variance of the signal (variance due to different cluster means).
        
        For a Gaussian mixture, the total variance is:
        Var(X) = Σ w_i * (μ_i² + σ_i²) - (Σ w_i * μ_i)²
        
        The signal variance (without noise) is:
        Var_signal = Σ w_i * μ_i² - (Σ w_i * μ_i)²
        
        Parameters
        ----------
        weights : Union[np.ndarray, torch.Tensor]
            Mixture weights, shape (K,)
        means : Union[np.ndarray, torch.Tensor]
            Component means, shape (K, D)
            
        Returns
        -------
        float
            The signal variance
        """
        # For multivariate case, compute the average variance across dimensions
        signal_vars = []
        
        # Get dimensionality
        if isinstance(means, np.ndarray):
            dim = means.shape[1]
        else:
            dim = means.size(1)
        
        for d in range(dim):
            # Extract means for this dimension
            if isinstance(means, np.ndarray):
                dim_means = means[:, d]
            else:
                dim_means = means[:, d].numpy()
                
            if isinstance(weights, torch.Tensor):
                weights_np = weights.numpy()
            else:
                weights_np = weights
            
            # Calculate E[X²] = Σ w_i * μ_i²
            exp_x_squared = np.sum(weights_np * np.square(dim_means))
            
            # Calculate (E[X])² = (Σ w_i * μ_i)²
            exp_x_squared_mean = np.square(np.sum(weights_np * dim_means))
            
            # Variance for this dimension
            dim_var = exp_x_squared - exp_x_squared_mean
            signal_vars.append(dim_var)
        
        # Return average variance across dimensions
        return np.mean(signal_vars)