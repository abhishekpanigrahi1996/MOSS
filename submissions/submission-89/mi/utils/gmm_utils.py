"""Utility functions for Gaussian Mixture Models."""

import torch
import math
import numpy as np
from typing import Tuple, Union


def log_gaussian(y: torch.Tensor, mu: torch.Tensor, sigma: float, d: int) -> torch.Tensor:
    """
    Compute ln N(y; mu, sigma^2 I_d).
    
    Parameters
    ----------
    y : torch.Tensor
        Points to evaluate, shape (..., d)
    mu : torch.Tensor
        Centers, shape (K, d) or (..., K, d)
    sigma : float
        Noise standard deviation
    d : int
        Dimensionality of the space
        
    Returns
    -------
    torch.Tensor
        Log-probability values
    """
    # Compute Gaussian log-density
    log_normalizer = -0.5 * d * torch.log(torch.tensor(2 * math.pi * sigma**2))
    
    # Compute squared distance, being careful with broadcasting dimensions
    # y has shape (..., d) and mu has shape (K, d) or (..., K, d)
    # We want to broadcast properly to get distance for each center
    
    # Reshape y to (..., 1, d) to broadcast against mu in K dimension
    y_expanded = y.unsqueeze(-2)
    
    # Compute squared distance
    squared_dist = torch.sum((y_expanded - mu)**2, dim=-1)
    
    # Apply Gaussian kernel
    log_kernel = -squared_dist / (2 * sigma**2)
    
    return log_normalizer + log_kernel


def log_gmm_density(y: torch.Tensor, p: torch.Tensor, mus: torch.Tensor, 
                   sigma: float, d: int) -> torch.Tensor:
    """
    Compute ln p_Y(y) for GMM using log-sum-exp trick for numerical stability.
    
    Parameters
    ----------
    y : torch.Tensor
        Points to evaluate, shape (N, d) or (d,)
    p : torch.Tensor
        Mixture weights, shape (K,)
    mus : torch.Tensor
        Mixture centers, shape (K, d)
    sigma : float
        Noise standard deviation
    d : int
        Dimensionality of the space
        
    Returns
    -------
    torch.Tensor
        Log-density values, shape (N,) or scalar
    """
    # Ensure y has correct shape for batch processing
    if y.ndim == 1:
        y = y.unsqueeze(0)  # (d,) -> (1, d)
    
    # Compute log probability for each component
    log_probs = torch.log(p) + log_gaussian(y, mus, sigma, d)
    
    # Use logsumexp for numerical stability
    log_density = torch.logsumexp(log_probs, dim=-1)
    
    # Squeeze output to match input dimensions
    return log_density.squeeze()


def generate_gmm_samples(p: torch.Tensor, mus: torch.Tensor, 
                        sigma: float, n_samples: int) -> torch.Tensor:
    """
    Generate samples from Y = X + sigma * noise where X follows a GMM.
    
    Parameters
    ----------
    p : torch.Tensor
        Mixture weights, shape (K,)
    mus : torch.Tensor
        Mixture centers, shape (K, d)
    sigma : float
        Noise standard deviation
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    torch.Tensor
        Samples, shape (n_samples, d)
    """
    K, d = mus.shape
    
    # Sample component indices according to probability vector p
    indices = torch.multinomial(p, n_samples, replacement=True)
    
    # Get corresponding centers
    x_samples = mus[indices]
    
    # Add Gaussian noise
    noise = torch.randn(n_samples, d) * sigma
    y_samples = x_samples + noise
    
    return y_samples


def h_y_given_x(sigma: float, d: int) -> float:
    """
    Compute conditional entropy h(Y|X) = d/2 * ln(2πe sigma^2).
    
    Parameters
    ----------
    sigma : float
        Noise standard deviation
    d : int
        Dimensionality of the space
        
    Returns
    -------
    float
        Conditional entropy h(Y|X)
    """
    return (d / 2.0) * math.log(2 * math.pi * math.e * sigma**2)


def compute_sample_statistics(log_py_values: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float]:
    """
    Compute sample statistics for entropy estimation.
    
    Parameters
    ----------
    log_py_values : Union[torch.Tensor, np.ndarray]
        Log-density values for samples
        
    Returns
    -------
    Tuple[float, float]
        Negative mean (entropy estimate) and standard error
    """
    # Convert to numpy for statistics if it's a tensor
    if isinstance(log_py_values, torch.Tensor):
        log_py_values = log_py_values.numpy()
        
    # Compute entropy estimate (negative mean of log densities)
    entropy_estimate = -np.mean(log_py_values)
    
    # Compute variance and standard error
    num_samples = len(log_py_values)
    if num_samples > 1:
        variance = np.var(log_py_values, ddof=1)  # Use unbiased estimator
        standard_error = (variance / num_samples)**0.5
    else:
        standard_error = float('inf')
        
    return float(entropy_estimate), float(standard_error)