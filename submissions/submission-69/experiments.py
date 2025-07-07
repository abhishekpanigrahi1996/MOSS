import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, Any, Tuple, Optional

from training import train
from evals import EvalVisualiser


def run(
    tries: int,
    setup_fn: Callable[[], Dict[str, Any]],
    visualisers: Dict[str, EvalVisualiser],
    seed: int = 0,
    *args: Any,
    **kwargs: Any
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run multiple training trials with visualisation and reproducible results.
    
    This function orchestrates multiple training runs with the same setup,
    collecting results for statistical analysis and visualisation. It ensures
    reproducibility by setting all relevant random seeds and provides progress
    tracking via tqdm.
    
    Args:
        tries: Number of training trials to run
        setup_fn: Function that returns experiment setup parameters as a dictionary.
                 Called once per trial to allow for trial-specific configurations.
        visualisers: Dictionary of visualiser objects that will collect and display
                    results. Each visualiser should have update() and display() methods.
        seed: Random seed for reproducibility across NumPy, Python random, and PyTorch
        *args: Additional positional arguments to pass to the train function
        **kwargs: Additional keyword arguments to pass to the train function
        
    Returns:
        Tuple containing:
        - visualisers: Dictionary of updated visualiser objects with collected results
        - result: The result dictionary from the final training run, or None if no runs
        
    Example:
        >>> def setup():
        ...     return {'model': MyModel(), 'optimizer': optimizer}
        >>> visualisers = {'loss': LineVisualiser(...)}
        >>> vis, last_result = run(10, setup, visualisers, seed=42)
        >>> # All visualisers will have collected data from 10 runs
    """
    # Disable progress bars and verbose output for individual training runs
    kwargs['show_pbar'] = False
    kwargs['verbose'] = False

    # Set random seeds for reproducibility across all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Additional settings for deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    result = None
    
    # Run multiple training trials with progress tracking
    for i in tqdm(range(tries), desc="Running trials"):
        # Get fresh setup for this trial (allows for trial-specific configurations)
        trial_setup = setup_fn()
        
        # Run training with combined arguments
        result = train(*args, **kwargs, **trial_setup)
        
        # Update all visualisers with the result from this trial
        for key in visualisers:
            visualisers[key].update(result)
    
    # Generate final visualisations and statistics
    for key in visualisers:
        visualisers[key].display()
    
    return visualisers, result
