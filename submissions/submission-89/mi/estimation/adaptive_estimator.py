"""Adaptive mutual information estimation for Gaussian Mixture Models."""

import torch
import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional, Union

from ..utils.gmm_utils import (
    log_gmm_density, 
    generate_gmm_samples, 
    h_y_given_x, 
    compute_sample_statistics
)
from ..utils.stats_utils import (
    t_confidence_interval, 
    compute_confidence_width, 
    samples_needed_for_width
)


class AdaptiveMIEstimator:
    """
    Adaptive framework for mutual information estimation using Gaussian Mixture Models.
    
    This class implements:
    1. Fixed-sample confidence interval estimation for I(Y;X)
    2. Adaptive estimation of I(Y;X) with precision guarantees
    """
    
    def __init__(self, p: torch.Tensor, mus: torch.Tensor):
        """
        Initialize the estimator with a GMM distribution.
        
        Parameters
        ----------
        p : torch.Tensor
            Probability vector of shape (K,) for the mixture components
        mus : torch.Tensor
            Centers of the mixture components, shape (K, d)
        """
        # Validate inputs
        if not torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("Probability vector p must sum to 1")
        
        if p.ndim != 1:
            raise ValueError("Probability vector p must be 1-dimensional")
            
        if mus.ndim != 2:
            raise ValueError("Centers mus must be 2-dimensional (K, d)")
            
        if p.shape[0] != mus.shape[0]:
            raise ValueError("Number of probabilities must match number of centers")
        
        self.p = p
        self.mus = mus
        self.d = mus.shape[1]  # Dimensionality
        self.K = p.shape[0]    # Number of mixture components

    def log_density(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Compute ln p_Y(y) using log-sum-exp trick for numerical stability.
        
        Parameters
        ----------
        y : torch.Tensor
            Points to evaluate, shape (N, d) or (d,)
        sigma : float
            Noise standard deviation
            
        Returns
        -------
        torch.Tensor
            Log-density values, shape (N,) or scalar
        """
        return log_gmm_density(y, self.p, self.mus, sigma, self.d)

    def generate_samples(self, sigma: float, n_samples: int) -> torch.Tensor:
        """
        Generate samples from Y = X + sigma * noise.
        
        Parameters
        ----------
        sigma : float
            Noise standard deviation
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        torch.Tensor
            Samples, shape (n_samples, d)
        """
        return generate_gmm_samples(self.p, self.mus, sigma, n_samples)

    def estimate_entropy_y(self, sigma: float, n_samples: int) -> Tuple[float, float]:
        """
        Estimate marginal entropy h(Y) using Monte Carlo samples.
        
        Parameters
        ----------
        sigma : float
            Noise standard deviation
        n_samples : int
            Number of samples to use
            
        Returns
        -------
        Tuple[float, float]
            Estimated entropy and standard error
        """
        # Generate samples
        y_samples = self.generate_samples(sigma, n_samples)
        
        # Compute log-density for all samples at once
        log_densities = self.log_density(y_samples, sigma)
        
        # Convert to numpy array for statistics
        log_py_array = log_densities.numpy()
        
        # Compute entropy estimate and statistics
        entropy_estimate, standard_error = compute_sample_statistics(log_py_array)
        
        return entropy_estimate, standard_error

    def estimate_mutual_information(self, 
                                   sigma: float, 
                                   n_samples: int, 
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """
        Estimate mutual information I(Y;X) = h(Y) - h(Y|X) with confidence interval.
        
        Parameters
        ----------
        sigma : float
            Noise standard deviation
        n_samples : int
            Number of samples to use
        alpha : float, optional
            Significance level (default: 0.05)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results:
            - 'mi_estimate': Mutual information point estimate
            - 'h_y_estimate': Marginal entropy estimate
            - 'h_y_given_x': Conditional entropy (exact)
            - 'standard_error': Standard error of the estimate
            - 'confidence_interval': Tuple of (lower, upper) bounds
            - 'n_samples': Number of samples used
        """
        # Estimate entropy of Y
        h_y_estimate, se = self.estimate_entropy_y(sigma, n_samples)
        
        # Compute h(Y|X) exactly
        h_y_given_x_value = h_y_given_x(sigma, self.d)
        
        # Calculate mutual information
        mi_estimate = h_y_estimate - h_y_given_x_value
        
        # Compute confidence interval
        lower, upper = t_confidence_interval(h_y_estimate, se, n_samples, alpha)
        mi_lower, mi_upper = lower - h_y_given_x_value, upper - h_y_given_x_value
        
        return {
            'mi_estimate': mi_estimate,
            'h_y_estimate': h_y_estimate,
            'h_y_given_x': h_y_given_x_value,
            'standard_error': se,
            'confidence_interval': (mi_lower, mi_upper),
            'n_samples': n_samples
        }

    def adaptive_estimate_mi(self, 
                            sigma: float, 
                            target_width: float, 
                            alpha: float = 0.05,
                            initial_samples: int = 100,
                            min_batch_size: int = 100,
                            max_samples: int = 100000) -> Dict[str, Any]:
        """
        Adaptively estimate mutual information with a target confidence interval width.
        
        Parameters
        ----------
        sigma : float
            Noise standard deviation
        target_width : float
            Target width of the confidence interval
        alpha : float, optional
            Significance level (default: 0.05)
        initial_samples : int, optional
            Initial number of samples (default: 100)
        min_batch_size : int, optional
            Minimum batch size to add in each iteration (default: 100)
        max_samples : int, optional
            Maximum number of samples to use (default: 100000)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results:
            - 'mi_estimate': Mutual information point estimate
            - 'h_y_estimate': Marginal entropy estimate
            - 'h_y_given_x': Conditional entropy (exact)
            - 'standard_error': Standard error of the estimate
            - 'confidence_interval': Tuple of (lower, upper) bounds
            - 'n_samples': Number of samples used
            - 'converged': Boolean indicating if the target width was achieved
            - 'achieved_width': The final confidence interval width
        """
        # Start with initial samples
        n_samples = initial_samples
        
        # Use existing method to estimate MI with initial samples
        result = self.estimate_mutual_information(sigma, n_samples, alpha)
        
        # Calculate current width of the confidence interval
        current_width = result['confidence_interval'][1] - result['confidence_interval'][0]
        
        # Continue until target width is achieved or max samples reached
        while current_width > target_width and n_samples < max_samples:
            # Get current standard error from the result
            se = result['standard_error']
            
            # Calculate variance from standard error
            variance = se**2 * n_samples
            
            # Use samples_needed_for_width to estimate total samples needed
            # Adding a safety factor of 2x to ensure we get enough samples
            total_samples_needed = samples_needed_for_width(variance, n_samples, target_width, alpha)
            total_samples_needed = min(total_samples_needed * 2, max_samples)  # Safety factor
            
            # Make sure we add at least min_batch_size samples
            n_samples = max(n_samples + min_batch_size, total_samples_needed)
            n_samples = min(n_samples, max_samples)
            
            # Use estimate_mutual_information with the new sample count
            result = self.estimate_mutual_information(sigma, n_samples, alpha)
            
            # Recalculate confidence interval width
            current_width = result['confidence_interval'][1] - result['confidence_interval'][0]
        
        # Add convergence information to result
        result['converged'] = current_width <= target_width
        result['achieved_width'] = current_width
        
        return result