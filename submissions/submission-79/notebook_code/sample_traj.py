"""
File containing functions for sampling the human trajectory
or human trajectory under AI intervention assuming that the
human's state is fully known.
"""
import numpy as np
from notebook_code.rl_utils import * 


def sample_trajectory(optimal_policy: np.ndarray, transition_matrix: np.ndarray, 
                      start_state: list, dims: list, n_steps=100, seed: int=None):
    """
    Samples a trajectory using the optimal policy and its corresponding 
    transition matrix.

    NOTE: this works both for human agents' transition matrices + policy
    as well as for AI agents' transition matrices + policy
    
    Parameters:
    -----------
    optimal_policy: np.ndarray
        numpy array of dimensionality [S]. Each element is the optimal 
        action at state s. 

    transition_matrix: np.ndarray
        for the human agent: np array of shape S_h x A_h x S'_h
        for the AI agent: np array of shape S_AI x A_AI x S'_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    start_state: list
        the initial state
        for the human agent: is multiple items: each progress chain state, 
        followed by stuck chain state
        for the AI agent: the above, followed by the human agent's
        state the previous step (so this is arbitrary)

    dims: list or tuple (list-like)
        for the human agent: contains the length of progress chains 1 through 
        the last one, followed by the length of the stuck chain
        for the AI agent: the above but followed by the size of the human 
        agent's action space, A_h

    n_steps: int
        maximum number of steps to roll out a trajectory for, if do not reach
        an absorbing state before this (if reach absorbing state before this,
        trajectory rollout stops there).

    seed: int
        seed for setting randomness
    
    Returns:
    --------
    trajectory_states: np.array ints
        Sequence of states visited during the trajectory.
        Note that this sequence contains the state indices, rather than tuples.
        NOTE: for AI agent: state takes form [s^{(h)}_t, a^{(h)}_{t-1}]

    trajectory_actions: np.array of ints
        Sequence of actions taken during the trajectory.
        The index of the action corresponds to the index of the state it
        was taken at in the trajectory_states list. (So this list will be
        of length 1 less than trajectory_states if reached absorbing state)
    """
    if seed is not None:
        np.random.seed(seed)
    trajectory_states = []
    trajectory_actions = []
    current_state = state_to_index(start_state, dims)
    
    for i in range(n_steps): # halt if don't reach absorbing state after many steps
        trajectory_states.append(current_state)
        action = optimal_policy[current_state]
        trajectory_actions.append(action)
        next_state_probs = transition_matrix[current_state, action, :]
        next_state = np.random.choice(range(len(next_state_probs)), p=next_state_probs)
        current_state = next_state

        # Terminate if current state is absorbing
        if np.allclose(transition_matrix[current_state, :, current_state], 1.0):
            trajectory_states.append(current_state)
            break

    trajectory_states = np.array(trajectory_states)
    trajectory_actions = np.array(trajectory_actions)
    
    return trajectory_states, trajectory_actions