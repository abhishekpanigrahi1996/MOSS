"""Inverse problem solvers for GMM mutual information."""

from .bisection_solver import find_sigma_for_target_mi

__all__ = ["find_sigma_for_target_mi"]