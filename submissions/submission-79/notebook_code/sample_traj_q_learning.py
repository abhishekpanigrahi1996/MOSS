"""
File containing functions for sampling the human trajectory
or human trajectory under AI intervention through Q-learning
(assuming that the human agent's transitions are not known).
"""
import numpy as np
from notebook_code.rl_utils import * 
import random


def sample_traj_q_learning_episodic_decay(transition_matrix: np.ndarray, init_Q: np.ndarray,
                      R: np.ndarray, start_state: list, dims: list, n_actions: int,
                      epsilon: float=1,
                      alpha: float=0.9, gamma: float=0.9,
                      n_steps=100, seed: int=None):
    """
    Samples a trajectory using Q-learning.  This runs 1 episode of Q learning.
    NOTE: for the episodic decay, need loop calling this to update epsilon
    between iterations of calling this and pass in updated epsilon.

    NOTE: this is for the AI agents' transition matrices + policy (true state
    transitions are determined by the human/AI transition matrix).

    NOTE: this version decays epsilon at the end of every episode.
    
    Parameters:
    -----------
    transition_matrix: np.ndarray
        for the AI agent: np array of shape S_AI x A_AI x S'_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    init_Q: np.ndarray
        initial Q matrix that this episode of Q-learning is starting off with.
        Q: numpy array of dimensionality [S, A]. Each element is the action-value 
        at a given state. 
        for the AI agent: np array of shape S_AI x A_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    R: np.ndarray
        numpy array of dimensionality [S, A, S]
        for the AI agent: is of shape S_AI x A_AI x S_AI
        is the reward for taking AI action a at AI state s and transitioning
        to AI state s' for any states s, s' and action a.

    start_state: list
        the initial state
        for the human agent: is multiple items: each progress chain state, 
        followed by stuck chain state
        for the AI agent: the above, followed by the human agent's
        state the previous step (so this is arbitrary)

    dims: list or tuple (list-like)
        for the AI agent: contains the length of progress chains 1 through 
        the last one, followed by the length of the stuck chain, followed by 
        the size of the human agent's action space, A_h

    n_actions: int
        number of possible actions in the action space of the MDP whose
        trajectory is being sampled (for human is |A_h|, for AI is |A_AI|)

    alpha: float
        learning rate, is in [0, 1]

    gamma: float
        discount factor

    n_steps: int
        maximum number of steps to roll out a trajectory for, if do not reach
        an absorbing state before this (if reach absorbing state before this,
        trajectory rollout stops there).

    seed: int
        seed for setting randomness

    epsilon: float
        epsilon value for epsilon-greedy Q-learning.
        If decay_rate != 0, will decay to the min_epsilon value
        (if decay_rate == 0, will remain constant)

    epsilon_decay: float
        rate by which to decay epsilon by (factor which is subtracted
        off epsilon at every timestep).  Should be calculated as:
            (epsilon - min_epsilon) / total_steps
        where
            total_steps = num_episodes * max_steps_per_episode

    min_epsilon: float
        minimum value of epsilon to decay to (cannot go below this)
    
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

    Q: np.ndarray
        final Q matrix that this episode of Q-learning is resulted in.
        Q: numpy array of dimensionality [S, A]. Each element is the action-value 
        at a given state. 
        for the AI agent: np array of shape S_AI x A_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    total_reward: float
        total discounted reward accumulated by the MDP
    """
    if seed is not None:
        np.random.seed(seed)
    trajectory_states = []
    trajectory_actions = []
    current_state = state_to_index(start_state, dims)
    total_reward = 0

    Q = init_Q

    gamma_d = 1
    
    for i in range(n_steps): # halt if don't reach absorbing state after many steps
        trajectory_states.append(current_state)

        # ε-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions-1)
        else:
            action = np.argmax(Q[current_state])

        trajectory_actions.append(action)

        # Get next state from current action
        next_state_probs = transition_matrix[current_state, action, :]
        next_state = np.random.choice(range(len(next_state_probs)), p=next_state_probs)

        # Update Q-value
        best_next_action_return = np.max(Q[next_state])
        reward = R[current_state, action, next_state]
        Q[current_state][action] += alpha * (reward + gamma * best_next_action_return - Q[current_state][action])

        total_reward += reward * gamma_d
        gamma_d *= gamma
        
        # Set up for the next iteration
        current_state = next_state

        # Terminate if next state is absorbing
        if np.allclose(transition_matrix[current_state, :, current_state], 1.0):
            trajectory_states.append(current_state)
            break

    trajectory_states = np.array(trajectory_states)
    trajectory_actions = np.array(trajectory_actions)
    
    return trajectory_states, trajectory_actions, Q, total_reward