"""
File containing functions for sampling the human trajectory
under AI intervention using the stuck estimators.
"""

import numpy as np
from notebook_code.rl_utils import * 
from notebook_code.estimators import MLE_stuck_one_step_under_AI, prob_last_state_stuck_all_start_to_cur_ind_under_AI



def sample_trajectory_under_iter_est(optimal_AI_policy: np.ndarray, T_AI: np.ndarray, 
                      start_AI_state: list, AI_dims: list, n_steps=100, seed: int=None):
    """
    Samples a trajectory using the optimal AI policy and its corresponding 
    transition matrix.  Does so by estimating the stuck state using the
    iterative estimator.  The action is chosen from the policy under the 
    estimated state.  Then this action and the true state are used to get
    the transition to the next state.

    NOTE: this works for the AI agents' transition matrices + policy
    
    Parameters:
    -----------
    optimal_AI_policy: np.ndarray
        numpy array of dimensionality [S]. Each element is the optimal 
        action at state s. NOTE: this is the AI's optimal policy.

    T_AI: np.ndarray
        np array of shape S_AI x A_AI x S'_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    start_AI_state: list
        the initial state
        is multiple items: each progress chain state, followed by stuck 
        chain state, followed by the human agent's state the previous step 
        (so this is arbitrary)

    AI_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one, 
        followed by the length of the stuck chain, followed by the size 
        of the human agent's action space, A_h

    n_steps: int
        maximum number of steps to roll out a trajectory for, if do not reach
        an absorbing state before this (if reach absorbing state before this,
        trajectory rollout stops there).

    seed: int
        seed for setting randomness
    
    Returns:
    --------
    trajectory_states: np.array ints
        Sequence of AI states visited during the trajectory.
        Note that this sequence contains the state indices, rather than tuples.
        NOTE: for AI agent: state takes form [s^{(h)}_t, a^{(h)}_{t-1}]

    trajectory_actions: np.array of ints
        Sequence of AI actions taken during the trajectory.
        The index of the action corresponds to the index of the state it
        was taken at in the trajectory_states list. (So this list will be
        of length 1 less than trajectory_states if reached absorbing state)

    num_overestimates: int
        number of times the stuck estimator overestimated
    num_underestimates: int
        number of times the stuck estimator underestimated
    total_amt_overestimate: int
        over all cases where estimator overestimated: sum of difference
        between estimate and true stuck value
    total_amt_underestimate: int
        over all cases where estimator underestimated: sum of difference
        between true stuck value and estimate 
    """
    if seed is not None:
        np.random.seed(seed)
    # containers for the AI states and actions in the trajectory rolled out
    trajectory_states = []
    trajectory_actions = []
    current_state = state_to_index(start_AI_state, AI_dims)
    prev_stuck_iter = 0

    num_overestimates = 0
    num_underestimates = 0
    total_amt_overestimate = 0
    total_amt_underestimate = 0
    
    for i in range(n_steps): # halt if don't reach absorbing state after many steps
        trajectory_states.append(current_state)
        if i == 0:
            # we know the initial stuck state is 0 so can just directly
            # take AI action from current state
            action = optimal_AI_policy[current_state]
        else:
            # need to estimate the current stuck state, don't know it
            # cur_state_idx = traj_dataset["states"][k]
            cur_state_AI_idx = trajectory_states[i]

            prev_state_AI_idx = trajectory_states[i-1]
            prev_state_AI_tuple = index_to_state(prev_state_AI_idx, AI_dims)
            prev_est_AI_state_tuple = (prev_state_AI_tuple[0], prev_state_AI_tuple[1], prev_stuck_iter, prev_state_AI_tuple[3])
            prev_est_AI_state_idx = state_to_index(prev_est_AI_state_tuple, AI_dims)

            prev_AI_action = trajectory_actions[i-1]
        
            # this should be based on previous state and action taken from previous state to current state plus current state
            estim_cur_stuck_iter, max_likeli_iter = MLE_stuck_one_step_under_AI(T_AI, None, cur_state_AI_idx, prev_est_AI_state_idx, prev_AI_action, True, AI_dims)
            
            # Now use estimated stuck state to get the current AI action
            # plug estimated stuck state into current state
            cur_state_AI_tuple = index_to_state(cur_state_AI_idx, AI_dims)
            cur_est_AI_state_tuple = (cur_state_AI_tuple[0], cur_state_AI_tuple[1], estim_cur_stuck_iter, cur_state_AI_tuple[3])
            cur_est_AI_state_idx = state_to_index(cur_est_AI_state_tuple, AI_dims)
            action = optimal_AI_policy[cur_est_AI_state_idx]

            # check whether over or underestimate
            if estim_cur_stuck_iter > cur_state_AI_tuple[2]:
                # overestimated
                num_overestimates += 1
                total_amt_overestimate += estim_cur_stuck_iter - cur_state_AI_tuple[2]
            elif estim_cur_stuck_iter < cur_state_AI_tuple[2]:
                # underestimated
                num_underestimates += 1
                total_amt_underestimate += cur_state_AI_tuple[2] - estim_cur_stuck_iter

            prev_stuck_iter = estim_cur_stuck_iter # set up for next iter: use estimated stuck to estimate next stuck

        trajectory_actions.append(action)
        # transition based on real human state
        next_state_probs = T_AI[current_state, action, :]
        next_state = np.random.choice(range(len(next_state_probs)), p=next_state_probs)
        current_state = next_state

        # Terminate if current state is absorbing
        if np.allclose(T_AI[current_state, :, current_state], 1.0):
            trajectory_states.append(current_state)
            break

    trajectory_states = np.array(trajectory_states)
    trajectory_actions = np.array(trajectory_actions)
    
    return trajectory_states, trajectory_actions, num_overestimates, num_underestimates, total_amt_overestimate, total_amt_underestimate


def sample_trajectory_under_seq_est(optimal_AI_policy: np.ndarray, T_AI: np.ndarray, 
                      start_state: list, AI_dims: list, n_steps=100, seed: int=None):
    """
    Samples a trajectory using the optimal AI policy and its corresponding 
    transition matrix.  Does so by estimating the stuck state using the
    sequential estimator.  The action is chosen from the policy under the 
    estimated state.  Then this action and the true state are used to get
    the transition to the next state.

    NOTE: this works for the AI agents' transition matrices + policy
    
    Parameters:
    -----------
    optimal_AI_policy: np.ndarray
        numpy array of dimensionality [S]. Each element is the optimal 
        action at state s. NOTE: this is the AI's optimal policy.

    T_AI: np.ndarray
        np array of shape S_AI x A_AI x S'_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.

    start_AI_state: list
        the initial AI state
        is multiple items: each progress chain state, followed by stuck 
        chain state, followed by the human agent's state the previous step 
        (so this is arbitrary)

    AI_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one, 
        followed by the length of the stuck chain, followed by the size 
        of the human agent's action space, A_h

    n_steps: int
        maximum number of steps to roll out a trajectory for, if do not reach
        an absorbing state before this (if reach absorbing state before this,
        trajectory rollout stops there).

    seed: int
        seed for setting randomness
    
    Returns:
    --------
    trajectory_states: np.array ints
        Sequence of AI states visited during the trajectory.
        Note that this sequence contains the state indices, rather than tuples.
        NOTE: for AI agent: state takes form [s^{(h)}_t, a^{(h)}_{t-1}]

    trajectory_actions: np.array of ints
        Sequence of AI actions taken during the trajectory.
        The index of the action corresponds to the index of the state it
        was taken at in the trajectory_states list. (So this list will be
        of length 1 less than trajectory_states if reached absorbing state)

    num_overestimates: int
        number of times the stuck estimator overestimated
    num_underestimates: int
        number of times the stuck estimator underestimated
    total_amt_overestimate: int
        over all cases where estimator overestimated: sum of difference
        between estimate and true stuck value
    total_amt_underestimate: int
        over all cases where estimator underestimated: sum of difference
        between true stuck value and estimate 
    """
    if seed is not None:
        np.random.seed(seed)

    current_AI_state = state_to_index(start_state, AI_dims)
    AI_traj_dataset = {
        "states": [],
        "actions": [],
    }

    num_overestimates = 0
    num_underestimates = 0
    total_amt_overestimate = 0
    total_amt_underestimate = 0
    
    for i in range(n_steps): # halt if don't reach absorbing state after many steps
        if i == 0:
            # we know the initial stuck state is 0 so can just directly
            # take AI action from current state
            AI_action = optimal_AI_policy[current_AI_state]
            AI_traj_dataset["states"].append(current_AI_state)
        else:
            # need to estimate the current stuck state, don't know it
            AI_traj_dataset["states"].append(current_AI_state)

            probs_stuck_arr_ind = prob_last_state_stuck_all_start_to_cur_ind_under_AI(AI_traj_dataset, T_AI, None, i, optimal_AI_policy, AI_dims)
            estim_cur_stuck_seq = np.argmax(probs_stuck_arr_ind)

            # Now use estimated stuck state to get the current AI action
            # plug estimated stuck state into current state
            cur_state_AI_tuple = index_to_state(current_AI_state, AI_dims)
            cur_est_AI_state_tuple = (cur_state_AI_tuple[0], cur_state_AI_tuple[1], estim_cur_stuck_seq, cur_state_AI_tuple[3])
            cur_est_AI_state_idx = state_to_index(cur_est_AI_state_tuple, AI_dims)

            # check whether over or underestimate
            if estim_cur_stuck_seq > cur_state_AI_tuple[2]:
                # overestimated
                num_overestimates += 1
                total_amt_overestimate += estim_cur_stuck_seq - cur_state_AI_tuple[2]
            elif estim_cur_stuck_seq < cur_state_AI_tuple[2]:
                # underestimated
                num_underestimates += 1
                total_amt_underestimate += cur_state_AI_tuple[2] - estim_cur_stuck_seq

            AI_action = optimal_AI_policy[cur_est_AI_state_idx]

        AI_traj_dataset['actions'].append(AI_action)
        # transition based on real human state
        next_state_probs = T_AI[current_AI_state, AI_action, :]
        next_AI_state = np.random.choice(range(len(next_state_probs)), p=next_state_probs)

        # Update current state for next iter
        current_AI_state = next_AI_state

        # Terminate if current state is absorbing
        if np.allclose(T_AI[current_AI_state, :, current_AI_state], 1.0):
            AI_traj_dataset['states'].append(current_AI_state)
            break

    trajectory_states = np.array(AI_traj_dataset["states"])
    trajectory_actions = np.array(AI_traj_dataset["actions"])
    
    return trajectory_states, trajectory_actions, num_overestimates, num_underestimates, total_amt_overestimate, total_amt_underestimate
