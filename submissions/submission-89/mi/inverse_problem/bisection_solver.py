"""Inverse problem solvers for mutual information using Probabilistic Bisection Algorithm (PBA)."""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union, Callable

from ..estimation.adaptive_estimator import AdaptiveMIEstimator
from ..utils.pba_utils import ProbabilisticBisection


def find_sigma_for_target_mi(estimator: AdaptiveMIEstimator,
                          target_mi: float,
                          mi_threshold: float = 0.05,  # Max width of CI when containing target MI
                          alpha: float = 0.05,         # Significance level for CI calculation
                          sigma_low: float = 0.01,
                          sigma_high: float = 10.0,
                          max_iter: int = 50,
                          early_termination_width: float = 1e-3,
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Solve the inverse problem: find sigma such that I(Y;X) = target_mi using PBA.
    
    Uses Probabilistic Bisection Algorithm with an adaptive grid for nonparametric density representation.
    
    Parameters
    ----------
    estimator : AdaptiveMIEstimator
        The estimator to use for MI calculations
    target_mi : float
        Target mutual information value
    mi_threshold : float, optional
        Maximum width of confidence interval when it contains the target MI.
        The oracle is considered "done" when the CI contains the target and 
        its width is less than this value, by default 0.05
    alpha : float, optional
        Significance level for confidence intervals, by default 0.05.
        This is also used as the PBA correctness probability.
    sigma_low : float, optional
        Lower bound for sigma, by default 0.01
    sigma_high : float, optional
        Upper bound for sigma, by default 10.0
    max_iter : int, optional
        Maximum number of iterations, by default 50
    early_termination_width : float, optional
        Early termination if the credible interval for sigma is narrower 
        than this value, by default 1e-3
    verbose : bool, optional
        Whether to display progress bar, by default False
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing solution results:
        - 'sigma': Found value of sigma (final median)
        - 'converged': Boolean indicating if the method converged
        - 'convergence_reason': String describing the convergence reason
        - 'credible_interval': 95% credible interval for sigma
        - 'mi_estimate': Estimated mutual information at the final sigma
        
    Notes
    -----
    If the target MI is outside the achievable range for the initial sigma bounds:
    - For target MI higher than achievable with sigma_low: Automatically tries smaller 
      sigma values (up to 10 attempts) to achieve higher MI
    - For target MI lower than achievable with sigma_high: Automatically tries larger
      sigma values (up to 10 attempts) to achieve lower MI
    - If target MI is still unachievable after adjustment attempts:
      - Returns sigma=0 with 'converged'=False for too high MI
      - Returns sigma=inf with 'converged'=False for too low MI
    """
    # Try to adjust sigma range when target MI is out of bounds
    original_sigma_low = sigma_low
    original_sigma_high = sigma_high
    max_sigma_adjustments = 10  # Prevent infinite loops
    
    for adjustment_attempt in range(max_sigma_adjustments + 1):
        # Check if the target MI is within the achievable range
        result_low = estimator.adaptive_estimate_mi(sigma_low, mi_threshold, alpha)
        result_high = estimator.adaptive_estimate_mi(sigma_high, mi_threshold, alpha)
        
        # Extract confidence intervals
        low_ci_lower, low_ci_upper = result_low['confidence_interval']
        high_ci_lower, high_ci_upper = result_high['confidence_interval']
        
        # Since MI decreases as sigma increases:
        # - The highest achievable MI is at sigma_low
        # - The lowest achievable MI is at sigma_high
        if target_mi > low_ci_upper:
            # Target MI is higher than what's achievable with smallest sigma
            if adjustment_attempt < max_sigma_adjustments:
                # Shrink sigma_low by half to try reaching higher MI
                sigma_low /= 2
                if verbose:
                    print(f"Target MI {target_mi} higher than achievable with sigma={original_sigma_low}. "
                          f"Trying smaller sigma: {sigma_low}")
            else:
                # After multiple attempts, return special value for "too high MI"
                return {
                    'sigma': 0.0,  # Represents "effectively zero sigma"
                    'converged': False,
                    'convergence_reason': f"Target MI {target_mi} too high even after shrinking sigma to {sigma_low}",
                    'credible_interval': (0.0, sigma_low),
                    'mi_estimate': low_ci_upper
                }
        elif target_mi < high_ci_lower:
            # Target MI is lower than what's achievable with largest sigma
            if adjustment_attempt < max_sigma_adjustments:
                # Double sigma_high to try reaching lower MI
                sigma_high *= 2
                if verbose:
                    print(f"Target MI {target_mi} lower than achievable with sigma={original_sigma_high}. "
                          f"Trying larger sigma: {sigma_high}")
            else:
                # After multiple attempts, return special value for "too low MI"
                return {
                    'sigma': float('inf'),  # Represents "effectively infinite sigma"
                    'converged': False,
                    'convergence_reason': f"Target MI {target_mi} too low even after increasing sigma to {sigma_high}",
                    'credible_interval': (sigma_high, float('inf')),
                    'mi_estimate': high_ci_lower
                }
        else:
            # Target MI is within achievable range, proceed with bisection
            break
    
    # Initialize PBA solver with the specified parameters
    pba_solver = ProbabilisticBisection(
        search_interval=(sigma_low, sigma_high),
        p_correct=1-alpha,  # Use confidence level for PBA probability
        early_termination_width=early_termination_width,
        max_iterations=max_iter,
        verbose=verbose
    )
    
    # Keep track of the MI estimate at the final sigma
    final_mi_result = None
    
    def noisy_oracle(sigma: float) -> Tuple[bool, bool]:
        """
        Noisy oracle for estimating mutual information.
        
        Parameters
        ----------
        sigma : float
            The sigma value to evaluate
            
        Returns
        -------
        Tuple[bool, bool]
            (response, done_flag) where:
            - response is True if the estimated MI is GREATER than the target (sigma is too LOW)
            - response is False if the estimated MI is LESS than the target (sigma is too HIGH)
            - done_flag is True if the target MI is within the confidence interval
              and the confidence interval is narrow enough
        """
        nonlocal final_mi_result
        
        # Start with our target MI threshold precision
        current_width = mi_threshold
        
        while True:
            # Calculate MI estimate with current target width
            result = estimator.adaptive_estimate_mi(
                sigma=sigma,
                target_width=current_width,
                alpha=alpha
            )
            
            # Update final_mi_result (will be overwritten as PBA progresses)
            final_mi_result = result
            
            # Parse the result
            mi_estimate = result['mi_estimate']
            ci_lower, ci_upper = result['confidence_interval']
            ci_width = ci_upper - ci_lower
            
            # Case 1: CI is below target - we have a confident response
            if ci_upper < target_mi:
                # The MI is too low, so sigma is too high (MI decreases as sigma increases)
                response = False  # sigma is too high
                done_flag = False
                break
                
            # Case 2: CI is above target - we have a confident response
            elif ci_lower > target_mi:
                # The MI is too high, so sigma is too low (MI decreases as sigma increases)
                response = True  # sigma is too low
                done_flag = False
                break
                
            # Case 3: CI contains target
            else:
                # If CI is narrow enough, we're done
                if ci_width < mi_threshold:
                    # For the final decision, use the point estimate
                    # True if MI is too high (sigma too low), False if MI is too low (sigma too high)
                    response = mi_estimate > target_mi
                    done_flag = True
                    break
                    
                # Otherwise, refine the estimate by reducing the target width
                current_width /= 2
        
        return response, done_flag
    
    # Run the PBA solver with our oracle
    result = pba_solver.solve(noisy_oracle)
    
    # Get the final sigma
    final_sigma = result['root']
    
    # Format the result with only essential information
    return {
        'sigma': final_sigma,
        'converged': result['converged'],
        'convergence_reason': result['convergence_reason'],
        'credible_interval': result['credible_interval'],
        'mi_estimate': final_mi_result['mi_estimate'] if final_mi_result else None
    }