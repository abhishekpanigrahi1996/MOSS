"""Statistical utility functions for confidence intervals and hypothesis testing."""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional


def t_confidence_interval(mean: float, se: float, n_samples: int, 
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute confidence interval using t-distribution.
    
    Parameters
    ----------
    mean : float
        Sample mean
    se : float
        Standard error
    n_samples : int
        Number of samples
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns
    -------
    Tuple[float, float]
        Lower and upper confidence bounds
    """
    t_crit = stats.t.ppf(1 - alpha/2, n_samples - 1)
    half_width = t_crit * se
    
    return mean - half_width, mean + half_width


def compute_confidence_width(se: float, n_samples: int, alpha: float = 0.05) -> float:
    """
    Compute confidence interval width for a given standard error and sample size.
    
    Parameters
    ----------
    se : float
        Standard error
    n_samples : int
        Number of samples
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns
    -------
    float
        Width of the confidence interval
    """
    t_crit = stats.t.ppf(1 - alpha/2, n_samples - 1)
    return 2 * t_crit * se


def samples_needed_for_width(current_variance: float, current_n: int, 
                           target_width: float, alpha: float = 0.05) -> int:
    """
    Estimate the number of samples needed to achieve a target confidence interval width.
    
    Parameters
    ----------
    current_variance : float
        Current sample variance
    current_n : int
        Current number of samples
    target_width : float
        Target width of the confidence interval
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns
    -------
    int
        Estimated number of samples needed
    """
    t_crit = stats.t.ppf(1 - alpha/2, current_n - 1)
    
    # Estimate required n for target width
    # Width = 2 * t_crit * sqrt(variance / n)
    # n = 4 * (t_crit^2) * variance / (width^2)
    required_n = int(np.ceil(4 * (t_crit**2) * current_variance / (target_width**2)))
    
    return max(required_n, current_n + 1)  # Ensure we at least add one sample