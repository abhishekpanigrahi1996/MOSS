"""
Contains utility functions for multichain RL.
"""

import numpy as np
from itertools import product


def iterate_binary_arrays(n):
    """
    Iterate through all possible binary arrays of length n composed of 0s and 1s.
    
    Parameters:
        n (int): The length of the arrays.
        
    Yields:
        numpy.ndarray: A binary array of length n.
        This saves memory, only processing a numpy array into memory as needed.
    """
    for binary_tuple in product([0, 1], repeat=n):
        yield np.array(binary_tuple, dtype=int)


def iterate_back_forth_arrays(n):
    """
    Iterate through all possible arrays of length n composed of 0s, 1s, and -1s.

    Parameters:
        n (int): The length of the arrays.
        
    Yields:
        numpy.ndarray: An array of length n composed of 0s, 1s, and -1s.
        This saves memory, only processing a numpy array into memory as needed.
    """
    for back_forth_tuple in product([0, 1, -1], repeat=n):
        yield np.array(back_forth_tuple, dtype=int)



def multi_hot_policy(pi: np.ndarray, A: int):
    """
    Turns a policy vector (of len S) into a policy matrix (of size S x A)

    This function will multi-hot the pi to make it a 2D array of size S_h x A_h
    (so each row=state has a 1 corresponding to the action=column to take at 
    the state).

    Parameters:
    -----------
    pi: array-like or np.ndarray 
        1D arrays of length S, where each element is the action to take at 
        the state (which equals the index). 

    A: int
        number of possible actions (underlying the transition matrix from 
        which pi was made).

    Returns:
    --------
    pi_hot: np.ndarray
        2D array of size S_h x A_h (so each row=state has a 1 corresponding 
        to the action=column to take at the state).
    """
    pi_hot = np.eye(A)[pi]  # Use one-hot encoding by indexing into identity matrix
    return pi_hot