"""Probabilistic Bisection Algorithm (PBA) utilities for stochastic root-finding problems."""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from tqdm import tqdm


class ProbabilisticBisection:
    """
    Implements the Probabilistic Bisection Algorithm (PBA) for stochastic root-finding.
    
    PBA is an iterative method designed to solve one-dimensional stochastic root-finding 
    problems by maintaining and successively updating a probability distribution over 
    the location of the root.
    """
    
    def __init__(self, 
                 search_interval: Tuple[float, float],
                 p_correct: float = 0.6,
                 early_termination_width: float = 1e-3,
                 max_iterations: int = 50,
                 verbose: bool = False):
        """
        Initialize a PBA solver.
        
        Parameters
        ----------
        search_interval : Tuple[float, float]
            Initial search interval [a, b]
        p_correct : float, optional
            Assumed probability that the oracle's response is correct (>0.5), by default 0.6
        early_termination_width : float, optional
            Early termination if the credible interval is narrower than this value, by default 1e-3
        max_iterations : int, optional
            Maximum number of iterations, by default 50
        verbose : bool, optional
            Whether to display progress bar, by default False
        """
        self.a, self.b = sorted(search_interval)
        self.p_correct = p_correct
        self.early_termination_width = early_termination_width
        self.max_iterations = max_iterations
        self.verbose = verbose
        
    def compute_total_mass(self, bounds: List[float], densities: List[float]) -> float:
        """
        Compute the total mass (integral) of a piecewise constant density.

        Parameters
        ----------
        bounds : List[float]
            Boundary points (length N+1).
        densities : List[float]
            Density values on each interval (length N).

        Returns
        -------
        float
            Total mass.
        """
        total = 0.0
        for i in range(len(densities)):
            total += densities[i] * (bounds[i+1] - bounds[i])
        return total

    def compute_quantile(self, bounds: List[float], densities: List[float], quantile: float = 0.5) -> float:
        """
        Compute the specified quantile of a piecewise constant density.

        Parameters
        ----------
        bounds : List[float]
            Boundary points (length N+1).
        densities : List[float]
            Density values on each interval (length N).
        quantile : float, optional
            Desired quantile (e.g., 0.5 for the median), by default 0.5

        Returns
        -------
        float
            Quantile value.
        """
        total_mass = self.compute_total_mass(bounds, densities)
        target = quantile * total_mass
        cum_mass = 0.0
        for i in range(len(densities)):
            seg_mass = densities[i] * (bounds[i+1] - bounds[i])
            if cum_mass + seg_mass >= target:
                L, R = bounds[i], bounds[i+1]
                return L + (target - cum_mass) / densities[i]
            cum_mass += seg_mass
        return bounds[-1]

    def compute_credible_interval(self, bounds: List[float], densities: List[float], fraction: float = 0.95) -> Tuple[float, float]:
        """
        Compute the lower and upper quantiles defining a central credible interval.

        Parameters
        ----------
        bounds : List[float]
            Boundary points.
        densities : List[float]
            Density values on the intervals.
        fraction : float, optional
            The fraction of the total mass to include, by default 0.95

        Returns
        -------
        Tuple[float, float]
            (lower quantile, upper quantile)
        """
        lower_target = (1 - fraction) / 2
        upper_target = 1 - lower_target
        return (self.compute_quantile(bounds, densities, lower_target), 
                self.compute_quantile(bounds, densities, upper_target))

    def update_posterior(self, bounds: List[float], densities: List[float], 
                         median: float, response: bool, p: float) -> Tuple[List[float], List[float]]:
        """
        Update the piecewise constant posterior density given a query at the median.

        If an interval spans the median, it is split. The density values are updated by multiplying by:
        - p for segments on the side indicated by the oracle and (1-p) for the other side.
        The update is performed in a Bayesian fashion and the density is renormalized.

        Parameters
        ----------
        bounds : List[float]
            Current boundaries (length N+1).
        densities : List[float]
            Current density values (length N).
        median : float
            The query point.
        response : bool
            Oracle response; True indicates f(sigma) > f_target (sigma is too low).
            False indicates f(sigma) < f_target (sigma is too high).
        p : float
            Assumed probability that the oracle's response is correct (> 0.5).

        Returns
        -------
        Tuple[List[float], List[float]]
            (new_bounds, new_densities) representing the updated density (normalized to mass 1).
        """
        q = 1 - p
        new_bounds = []
        new_densities = []
        
        for i in range(len(densities)):
            L, R = bounds[i], bounds[i+1]
            d = densities[i]
            if R <= median:
                factor = p if (not response) else q
                new_bounds.append(L)
                new_bounds.append(R)
                new_densities.append(d * factor)
            elif L >= median:
                factor = q if (not response) else p
                new_bounds.append(L)
                new_bounds.append(R)
                new_densities.append(d * factor)
            else:
                # Interval straddles the median; split it.
                factor_left = p if (not response) else q
                factor_right = q if (not response) else p
                new_bounds.append(L)
                new_bounds.append(median)
                new_densities.append(d * factor_left)
                new_bounds.append(median)
                new_bounds.append(R)
                new_densities.append(d * factor_right)
        
        # Simplify to remove duplicates
        simplified_bounds = [new_bounds[0]]
        simplified_densities = []
        
        current_idx = 0
        for i in range(len(new_densities)):
            if i < len(new_densities) - 1 and np.isclose(new_densities[i], new_densities[i+1]):
                # Skip this boundary if the densities are the same
                continue
            simplified_bounds.append(new_bounds[current_idx + 1])
            simplified_densities.append(new_densities[i])
            current_idx += 2
        
        # Normalize the posterior
        total_mass = self.compute_total_mass(simplified_bounds, simplified_densities)
        simplified_densities = [d / total_mass for d in simplified_densities]
        
        return simplified_bounds, simplified_densities

    def solve(self, oracle_func: Callable[[float], Tuple[bool, bool]]) -> Dict[str, Any]:
        """
        Run the PBA algorithm to find the root of a noisy function.
        
        Parameters
        ----------
        oracle_func : Callable[[float], Tuple[bool, bool]]
            Function taking a point x and returning (response, done_flag)
            where response is True if f(x) < target (x is too low)
            and done_flag is True if |f(x) - target| < threshold (x is close enough)
            
        Returns
        -------
        Dict[str, Any]
            Results containing:
            - 'root': Final estimate of the root
            - 'iterations': Number of iterations used
            - 'credible_interval': 95% credible interval for the root
            - 'convergence_reason': String describing the convergence reason
            - 'converged': Boolean indicating if the method converged
        """
        # Initialize posterior
        bounds = [self.a, self.b]
        densities = [1.0 / (self.b - self.a)]
        medians = []
        
        # Run the PBA algorithm
        iteration_range = tqdm(range(self.max_iterations), desc="PBA Search") if self.verbose else range(self.max_iterations)
        
        for _ in iteration_range:
            # Compute median of current posterior
            median = self.compute_quantile(bounds, densities, 0.5)
            medians.append(median)
            
            # Query the oracle
            response, done_flag = oracle_func(median)
            
            # If the oracle reports that the current point is already good enough, break out
            if done_flag:
                convergence_reason = 'threshold_reached'
                break
            
            # Update posterior
            bounds, densities = self.update_posterior(
                bounds, densities, median, response, self.p_correct
            )
            
            # Check if credible interval is narrow enough
            ci_left, ci_right = self.compute_credible_interval(bounds, densities)
            if (ci_right - ci_left) <= self.early_termination_width:
                convergence_reason = 'credible_interval_width'
                break
        else:
            # If we exit the loop normally, we reached max iterations
            convergence_reason = 'max_iterations'
        
        # Final estimated root is the last median computed
        final_estimate = medians[-1] if medians else None
        
        # Compute the 95% credible interval
        ci_left, ci_right = self.compute_credible_interval(bounds, densities)
        
        return {
            'root': final_estimate,
            'iterations': len(medians),
            'credible_interval': (ci_left, ci_right),
            'convergence_reason': convergence_reason,
            'converged': (done_flag if 'done_flag' in locals() else False) or convergence_reason == 'credible_interval_width'
        }