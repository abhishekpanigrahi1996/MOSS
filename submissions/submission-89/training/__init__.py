"""
Training components for GMM transformer framework.

This package provides the training infrastructure for the transformer-based
GMM framework, including trainer, experiment management, metrics, and utilities.
"""

from .trainer import GMMTrainer
from .experiment import ExperimentManager

__all__ = [
    'GMMTrainer',
    'ExperimentManager'
]