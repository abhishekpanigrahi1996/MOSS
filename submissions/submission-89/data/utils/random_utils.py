"""
Utility functions for random number generation and state management.

This module provides classes and functions for managing random state
to ensure reproducibility in GMM data generation.
"""

import numpy as np
from typing import Dict, Any, Optional


class RandomState:
    """
    Wrapper for numpy's random number generator with reliable state management.
    
    This class ensures reproducibility by managing the bit generator state directly,
    which is more reliable than using the higher-level state management methods.
    
    Examples
    --------
    >>> random_state = RandomState(seed=42)
    >>> random_numbers = random_state.rng.random(5)
    >>> state = random_state.get_state()
    >>> restored_state = RandomState.from_state(state)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random state with a seed.
        
        Parameters
        ----------
        seed : Optional[int]
            Random seed for initialization (None for random seed)
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Save a copy of the original state for debugging/testing
        self._bit_generator_type = type(self.rng.bit_generator)
        self._initial_state = self._copy_bit_generator_state(self.rng.bit_generator.state)
    
    def _copy_bit_generator_state(self, state: Dict) -> Dict:
        """
        Create a deep copy of the bit generator state.
        
        This is important because the state dict can contain nested mutable objects.
        
        Parameters
        ----------
        state : Dict
            The bit generator state
            
        Returns
        -------
        Dict
            A deep copy of the bit generator state
        """
        # Create a new dict with copied entries
        result = {}
        for key, value in state.items():
            if isinstance(value, dict):
                # Recursively copy nested dicts
                result[key] = self._copy_bit_generator_state(value)
            elif isinstance(value, (list, np.ndarray)):
                # Copy lists and arrays
                result[key] = value.copy() if hasattr(value, 'copy') else value[:]
            else:
                # Primitive values (ints, strings, etc.) are copied by assignment
                result[key] = value
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state for checkpointing.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with state information
        """
        # Use direct bit generator state for better reproducibility
        # Important: Make a deep copy to prevent modification of the returned state
        bg_state = self._copy_bit_generator_state(self.rng.bit_generator.state)
        return {
            "bit_generator_state": bg_state,
            "seed": self.seed
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a checkpoint.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from get_state
            
        Notes
        -----
        This implementation supports the modern bit_generator_state format.
        It properly handles conversions between array formats for PCG64.
        """
        # Store the seed value
        self.seed = state.get("seed", self.seed)
        
        # Handle bit generator state
        if "bit_generator_state" in state:
            # Make a deep copy of the state to prevent issues with shared references
            bg_state = self._copy_bit_generator_state(state["bit_generator_state"])
            
            # Handle the state attribute, which is critical for PCG64
            if "state" in bg_state:
                # Handle nested state dictionary (PCG64 format)
                if isinstance(bg_state["state"], dict):
                    # Process state value which must be uint64
                    if "state" in bg_state["state"] and isinstance(bg_state["state"]["state"], list):
                        # Convert list to numpy array with correct dtype
                        try:
                            bg_state["state"]["state"] = np.array(bg_state["state"]["state"], dtype=np.uint64)
                        except (ValueError, TypeError):
                            print("WARNING: Could not convert state array to uint64")
                    
                    # Convert other arrays back to numpy arrays
                    for k, v in bg_state["state"].items():
                        if k != "state" and isinstance(v, list):
                            try:
                                bg_state["state"][k] = np.array(v)
                            except (ValueError, TypeError):
                                print(f"WARNING: Could not convert state[{k}] array")
            
            # Set the processed state on the bit generator
            try:
                self.rng.bit_generator.state = bg_state
            except (ValueError, TypeError) as e:
                print(f"ERROR: Failed to set bit generator state: {e}")
                # Try to recreate the generator with the seed as a fallback
                if self.seed is not None:
                    print(f"Falling back to seed-based initialization with seed={self.seed}")
                    self.rng = np.random.default_rng(self.seed)
    
    def reset_to_initial(self) -> None:
        """
        Reset the random state to its initial state.
        
        This can be useful for reproducibility in tests.
        """
        if self._initial_state:
            self.rng.bit_generator.state = self._copy_bit_generator_state(self._initial_state)
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'RandomState':
        """
        Create a new random state from a checkpoint.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from get_state
            
        Returns
        -------
        RandomState
            Initialized RandomState object
        """
        random_state = cls(state.get("seed"))
        random_state.set_state(state)
        return random_state