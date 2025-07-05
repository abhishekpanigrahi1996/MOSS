"""
Factory functions for creating GMM generators from configuration.

This module provides factory functions to create parameter and data generators
from configuration dictionaries, with support for state management.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

from ..core import ParameterGenerator, DataGenerator


def create_generators(config_dict: Dict[str, Any], 
                     state: Optional[Dict[str, Any]] = None) -> Tuple[ParameterGenerator, DataGenerator]:
    """
    Create parameter and data generators from a config dictionary.
    
    This function creates generators with optional state restoration in one unified operation.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary for generator parameters
    state : Optional[Dict[str, Any]], optional
        Optional state dictionary to restore generator state, by default None
        
    Returns
    -------
    Tuple[ParameterGenerator, DataGenerator]
        Tuple of (param_generator, data_generator)
        
    Examples
    --------
    >>> config = {
    ...    "dim": 2,
    ...    "control_mode": "snr",
    ...    "cluster_params": {"type": "fixed", "value": 3},
    ...    "snr_db_params": {"type": "uniform", "min": 3.0, "max": 10.0},
    ...    "sample_count_distribution": {"type": "fixed", "value": 1000},
    ...    "vary_parameter_in_batch": True,
    ...    "random_seed": 42
    ... }
    >>> # Create fresh generators
    >>> param_gen, data_gen = create_generators(config)
    >>> 
    >>> # Save state after generating some data
    >>> state = save_state(param_gen, data_gen, "state.json")
    >>> 
    >>> # Restore generators with saved state
    >>> state = load_state("state.json")
    >>> param_gen, data_gen = create_generators(config, state)
    """
    from ..utils.distribution_utils import resolve_preset
    
    # Extract base configuration
    dim = config_dict.get("dim", 3)
    cluster_params = config_dict.get("cluster_params", {"type": "fixed", "value": 3})
    control_mode = config_dict.get("control_mode", "snr")
    alpha_dirichlet = config_dict.get("alpha_dirichlet", 1.0)
    
    # Get random seed, prioritizing state over config
    random_seed = None
    
    # If state is provided, try to get seed from there
    if state is not None and "param_generator" in state:
        param_state = state["param_generator"]
        if "random_state" in param_state and "seed" in param_state["random_state"]:
            random_seed = param_state["random_state"]["seed"]
    
    # If no seed from state, use the one from config
    if random_seed is None:
        random_seed = config_dict.get("random_seed")
    
    # Resolve string presets for cluster_params
    cluster_params = resolve_preset(cluster_params, 'cluster_params')
    
    # Create appropriate control configs based on mode
    snr_config = None
    mi_config = None
    mi_factor_config = None
    
    if control_mode == "snr":
        snr_config = config_dict.get("snr_db_params", {"type": "fixed", "value": 5.0})
        snr_config = resolve_preset(snr_config, 'snr_db_params')
    elif control_mode == "mi_factor":
        mi_factor_config = config_dict.get("mi_factor_params", {"type": "fixed", "value": 0.8})
    
    # Get sample count configuration
    sample_count_config = config_dict.get("sample_count_distribution", {"type": "fixed", "value": 1000})
    sample_count_config = resolve_preset(sample_count_config, 'sample_count_distribution')
    
    # Create parameter generator
    param_generator = ParameterGenerator(
        dim=dim,
        cluster_config=cluster_params,
        snr_config=snr_config,
        mi_config=mi_config,
        mi_factor_config=mi_factor_config,
        sample_count_config=sample_count_config,
        alpha_dirichlet=alpha_dirichlet,
        seed=random_seed
    )
    
    # Extract data generation settings - check for both old and new key names
    vary_control = config_dict.get("vary_parameter_in_batch", 
                                  config_dict.get("vary_snr_in_batch", False))
    
    # Create data generator
    data_generator = DataGenerator(
        param_generator=param_generator,
        vary_clusters_in_batch=False,  # Always false now
        vary_control_in_batch=vary_control,
        seed=random_seed
    )
    
    # Apply state if provided
    if state is not None:
        # Apply parameter generator state first
        if "param_generator" in state:
            param_generator.set_state(state["param_generator"])
        
        # Then apply data generator state
        if "data_generator" in state:
            # Ensure param_generator state is available to data_generator
            if "param_generator" in state and "param_generator_state" not in state["data_generator"]:
                state["data_generator"]["param_generator_state"] = state["param_generator"]
            
            data_generator.set_state(state["data_generator"])
    
    return param_generator, data_generator


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder with support for NumPy types.
    
    Extends the standard JSON encoder to handle NumPy arrays and scalars.
    """
    def default(self, obj: Any) -> Any:
        """
        Convert NumPy types to JSON-serializable values.
        
        Parameters
        ----------
        obj : Any
            Object to encode
            
        Returns
        -------
        Any
            JSON-serializable representation
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_state(param_generator: ParameterGenerator, 
              data_generator: DataGenerator, 
              file_path: Union[str, Path],
              additional_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Save the complete state to a file for later restoration.
    
    Parameters
    ----------
    param_generator : ParameterGenerator
        ParameterGenerator instance
    data_generator : DataGenerator
        DataGenerator instance
    file_path : Union[str, Path]
        Path to save state to
    additional_state : Optional[Dict[str, Any]], optional
        Additional state to include, by default None
        
    Returns
    -------
    Dict[str, Any]
        Complete state dictionary that was saved
        
    Examples
    --------
    >>> # Create generators
    >>> param_gen, data_gen = create_generators(config)
    >>> 
    >>> # Generate some data and save state
    >>> for _ in range(5):
    ...     batch = data_gen.generate_training_batch(batch_size=4)
    >>> 
    >>> # Save state
    >>> state = save_state(param_gen, data_gen, "state.json")
    """
    # Unified state structure
    state = {
        "param_generator": param_generator.get_state(),
        "data_generator": data_generator.get_state()
    }
    
    # Add additional state if provided
    if additional_state is not None:
        state.update(additional_state)
    
    # Ensure directory exists
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)
    
    # Save state to file
    with open(file_path, 'w') as f:
        json.dump(state, f, cls=NumpyEncoder)
    
    return state


def load_state(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load state from a file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to load state from
        
    Returns
    -------
    Dict[str, Any]
        Complete state dictionary
        
    Raises
    ------
    FileNotFoundError
        If the state file doesn't exist
        
    Examples
    --------
    >>> # Load state and create generators
    >>> state = load_state("state.json")
    >>> param_gen, data_gen = create_generators(config, state)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"State file not found: {file_path}")
    
    # Load state from file
    with open(file_path, 'r') as f:
        state = json.load(f)
    
    return state