"""
File which contains code for estimating things like the current value on the stuck chain.
"""

import numpy as np
from notebook_code.rl_utils import state_to_index, index_to_state
import math


def prob_last_state_stuck_all_start_to_cur_ind(traj_dataset: dict, T: np.ndarray, Q: np.ndarray, 
                                           cur_state: int, pi: np.ndarray, chain_dims: list):
    """
    A DP approach for computing the sequential estimator. 
    
    Function which computes the probabilities of what the stuck chain
    value at a given timestep (state) is by aggregating the probabilities
    of the sequences of stuck chain values from the initial through current 
    timestep's stuck chain value together which end in the same stuck chain
    value (i.e. if at timestep k, since we know s_0 = (0, 0, 0), will jointly
    estimate s_1^{(S)}, ..., s_k^{S}.  For each x that s_k^{S} could be: we
    will sum the probabilities of all sequences ending in s_k^{S} = x to get
    the probability that s_k^{S} = x.

    This is the sequential estimator.  NOTE: this means this version multiplies
    the probability by the indicator that the AI would've taken its previous
    action at the previous stuck state.  Assumes deterministic policy.

    NOTE: this version finds the probability of the last stuck state in
    a sequence being x for any x by using dynamic programming, making it
    significantly faster than the BFS version (it is extremely fast).

    NOTE: for long trajectories/sequences, a version of this function which
    computes the log of the probabilities should be made.

    Parameters:
    -----------
    traj_dataset: dict
        holds a trajectory rolled out from the MDP.  Contains the following
        key-value pairs:
        "states": 1-D np.ndarray of states (as ints in [0, S-1])
        "actions": 1-D np.ndarry of actions
        The index of the action corresponds to the index of the state it
        was taken at in the "states" list. (So this list will be of length 
        1 less than trajectory_states if reached absorbing state).
        NOTE: this contains just 1 trajectory, not multiple (this function
        is invalid if traj_dataset contains more than one dataset).
        NOTE: this function only relies on the progress chain states.

    T: numpy array of dimensionality [S, A, S]
        the transition matrix for the MDP, A is the size of the action
        space, S is the size of the state space.

    Q: numpy array of dimensionality [S, A]. 
        each element is the action-value at a given state.
        can be None if `determ_pol` is True. 

    cur_state: int
        the state/timestep for which we are estimating the stuck chain 
        value (viable state values: starts from 0 and ends at 
        len(traj_dataset[states]) - 1 = n - 1).
        NOTE: cur_state must be > 0 for this fn to work

    pi: np.ndarray
        np array containing the policy the agent is following.
        numpy array of dimensionality [S]. Each element is the optimal action at state s.

    chain_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one
        followed by the length of the stuck chain

    Returns:
    --------
    cur_stuck_ts_probs: np.ndarry of ints
        array of the values of the stuck chain estimates for the stuck
        state at timestep k (which is the current stuck state, s_k).
        Indices in the array corresponding to the stuck state value (0 through
        the end of the stuck chain), and elements are the probabilities of the
        current stuck state being the index value.
    """
    # arrays to store DP solution (iterating from estimating the stuck state
    # at timestep (ts) t=0 through the current timestep: cur_state)
    prev_stuck_ts_probs = np.zeros(chain_dims[-1])
    cur_stuck_ts_probs = np.zeros(chain_dims[-1])

    # s_0^{(S)} = 0 (the stuck state at timestep 0/the first state tuple is 0)
    prev_stuck_ts_probs[0] = 1
    cur_stuck_ts_probs[0] = 1

    # truncate trajectory dataset to just states 0 through cur_state
    states_to_cur = traj_dataset["states"][:cur_state+1]
    actions_to_cur = traj_dataset["actions"][:cur_state]

    # convert states from state indices to tuples
    state_tuples_list = [index_to_state(state_idx, chain_dims) for state_idx in states_to_cur]

    # goal is to compute: P[t][s] = at time t, probability of being at stuck state s
    # meaningless when s > t (since this isn't possible) or is past the end of the
    # stuck chain, so we don't compute those values (for example, P[0][1] is left undefined)
    # however, we only need access to the current timestep (t) and the previous timestep (t-1) so we
    # are working with 2 1D arrays corresponding to these 2 timesteps rather than a 2D array
    for t in range(1, cur_state+1): # timestep
        prev_stuck_ts_probs = list(cur_stuck_ts_probs)
        cur_stuck_ts_probs = np.zeros(chain_dims[-1])
        for s in range(min(t+1, chain_dims[-1])): # stuck state value for current timestep
            for ps in {s - 1, s, s + 1}: # previous stuck state
                if ps < 0 or ps > t-1 or ps > chain_dims[-1]-1:
                    # invalid: out of bounds
                    continue


                # P[t][s] += Pr(state s | previous state ps, time=t-1) * P[t-1][ps]

                # Get progress chain states for previous timestep, and make the
                # stuck state corresponding to it the previous stuck state being considered
                prev_state_tuple = state_tuples_list[t-1]
                prev_state_tuple = prev_state_tuple[:-1] + (ps,)
                prev_state_idx = state_to_index(prev_state_tuple, chain_dims)
                prev_action = actions_to_cur[t-1]

                # Get progress chain states for current timestep, and make the
                # stuck state corresponding to it the current stuck state being considered
                cur_state_tuple = state_tuples_list[t]
                cur_state_tuple = cur_state_tuple[:-1] + (s,)
                cur_state_idx = state_to_index(cur_state_tuple, chain_dims)

                # To get the probability of the stuck state being s given the previous
                # state tuple (where progress chain states are as observed in the trajectory
                # and the previous stuck chain state is the one we're considering, ps)
                # and current progress chain state (as observed in the trajectory),
                # we need to divide the probability of the current progress chain state
                # (as observed from our trajectory) and current stuck chain we're considering
                # given the previous stuck chain state and action (taken in the trajectory)
                # by the probability of the current progress chain state given the previous
                # state tuple (AKA definition of conditional probability w extra conditioning)
                cond_prob_numerator = T[prev_state_idx, prev_action, cur_state_idx]

                # need to sum over all possible stuck chain values for the
                # current progress chain state to get its marginal probability
                if cond_prob_numerator != 0:
                    cond_prob_denominator = 0
                    for stuck in range(chain_dims[-1]):
                        state_tuple = cur_state_tuple[:-1] + (stuck,)
                        state_idx = state_to_index(state_tuple, chain_dims)
                        cond_prob_denominator += T[prev_state_idx, prev_action, state_idx]
                else:
                    cond_prob_denominator = 1 # doesn't matter what value it is, 
                    # 0 probability of event happening, just made it non-zero so get
                    # 0 / 1 = 0 (what will happen later)

                # need to multiply the current stuck chain transition probability by the
                # probability of the previous sequence of stuck chain values (leading up
                # to the previous stuck chain value we were at) given the progress chains 
                # we observed and actions we took

                cur_stuck_ts_prob_to_add = (cond_prob_numerator / cond_prob_denominator
                                            ) * prev_stuck_ts_probs[ps] * (pi[prev_state_idx] == prev_action)

                cur_stuck_ts_probs[s] += cur_stuck_ts_prob_to_add

    return cur_stuck_ts_probs



def MLE_stuck_one_step(T: np.ndarray, Q: np.ndarray, cur_state_idx: int, prev_state_idx: int, 
                       prev_action: int, determ_pol: bool, chain_dims: list): 
    """
    The iterative estimator.
    
    Function which estimates the stuck chain value at the current state based on the previous 
    state's stuck chain value. 

    Parameters:
    -----------
    T: numpy array of shape [S, A, S]
        The transition matrix for the MDP.

    Q: numpy array of shape [S, A] (or None if `determ_pol` is True).
        The action-value function.

    cur_prog_state: int
        The current state for which we would like to estimate
        the stuck chain value for, as an index s \in [0, S-1].
    
    prev_state_idx: int
        The state at the timestep before the current state, as an
        index s \in [0, S-1].

    prev_action: int
        the action taken at the previous state (to transition from
        the previous state to the current state).

    determ_pol: bool
        - True: Uses the deterministic policy likelihood function.
        - False: Uses the stochastic policy likelihood function.

    chain_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one
        followed by the length of the stuck chain

    Returns:
    --------
    best_stuck_value: int
        MLE (estimate) of the stuck chain value at the current state.

    max_likelihood: float
        Likelihood of the observed trajectory dataset using the best estimate. 

    """
    # NOTE: this does not cur take into acct indicator that would take the action
    # that human did from the stuck state...
    prev_state_tuple = index_to_state(prev_state_idx, chain_dims)  
    prev_stuck_value = prev_state_tuple[-1]  

    curr_stuck_values = [prev_stuck_value - 1, prev_stuck_value, prev_stuck_value + 1]
    max_likelihood = -1
    best_stuck_value = None

    if not determ_pol:
        A = list(range(Q.shape[1]))
    
    # iterate through all viable stuck state values for the current state
    # to find the one which maximizes the probability from the previous
    # state to the current state
    for i in curr_stuck_values:
        if i < 0 or i > chain_dims[-1]-1:
            continue
        
        cur_state_tuple = index_to_state(cur_state_idx, chain_dims)
        updated_cur_state_tuple = cur_state_tuple[:-1] + (i,)
        updated_cur_state_idx = state_to_index(updated_cur_state_tuple, chain_dims)

        likeli = T[prev_state_idx, prev_action, updated_cur_state_idx]

        if not determ_pol:
            likeli *= math.exp(Q[prev_state_idx, prev_action]) / np.sum(np.exp(Q[prev_state_idx, A]))


        if likeli > max_likelihood:
            max_likelihood = likeli
            best_stuck_value = i

    return best_stuck_value, max_likelihood



def MLE_stuck_one_step_under_AI(T_AI: np.ndarray, Q: np.ndarray, cur_AI_state_idx: int, prev_AI_state_idx: int, 
                       prev_AI_action: int, determ_pol: bool, AI_dims: list): 
    """
    Computing the iterative estimator using DP under the AI's interventions.
    
    Function which estimates the stuck chain value at the current state based on the previous 
    state's stuck chain value.  Does so under the AI's interventions.

    Parameters:
    -----------
    T_AI: numpy array of shape [S, A, S]
        The transition matrix for the AI MDP.

    Q: numpy array of shape [S, A] (or None if `determ_pol` is True).
        The action-value function for the AI.

    cur_AI_state_idx: int
        The current state for which we would like to estimate
        the stuck chain value for, as an index s \in [0, S-1].
        Note that this is an AI state (consisting of the
        human's current state and previous action)
    
    prev_AI_state_idx: int
        The state at the timestep before the current state, as an
        index s \in [0, S-1].

    prev_AI_action: int
        the action taken by the AI at the previous state (to transition 
        from the previous state to the current state).

    determ_pol: bool
        - True: Uses the deterministic policy likelihood function.
        - False: Uses the stochastic policy likelihood function.

    AI_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one
        followed by the length of the stuck chain followed by the
        number of possible human actions.

    Returns:
    --------
    best_stuck_value: int
        MLE (estimate) of the stuck chain value at the current state.

    max_likelihood: float
        Likelihood of the observed trajectory dataset using the best estimate. 

    """
    prev_state_tuple = index_to_state(prev_AI_state_idx, AI_dims)  
    prev_stuck_value = prev_state_tuple[2]  

    curr_stuck_values = [prev_stuck_value - 1, prev_stuck_value, prev_stuck_value + 1]
    max_likelihood = -1
    best_stuck_value = None

    if not determ_pol:
        A = list(range(Q.shape[1]))
    
    # iterate through all viable stuck state values for the current state
    # to find the one which maximizes the probability from the previous
    # state to the current state
    for i in curr_stuck_values:
        if i < 0 or i > AI_dims[2]-1:
            continue
        
        cur_state_tuple = index_to_state(cur_AI_state_idx, AI_dims)
        updated_cur_state_tuple = cur_state_tuple[:2] + (i, ) + (cur_state_tuple[3],)
        updated_cur_state_idx = state_to_index(updated_cur_state_tuple, AI_dims)

        likeli = T_AI[prev_AI_state_idx, prev_AI_action, updated_cur_state_idx]

        if not determ_pol:
            likeli *= math.exp(Q[prev_AI_state_idx, prev_AI_action]) / np.sum(np.exp(Q[prev_AI_state_idx, A]))


        if likeli > max_likelihood:
            max_likelihood = likeli
            best_stuck_value = i

    return best_stuck_value, max_likelihood


def prob_last_state_stuck_all_start_to_cur_ind_under_AI(AI_traj_dataset: dict, T_AI: np.ndarray, Q: np.ndarray, 
                                           cur_AI_state: int, pi_AI: np.ndarray, AI_dims: list):
    """
    Computing the sequential estimator using DP under the AI's interventions.

    Function which computes the probabilities of what the stuck chain
    value at a given timestep (state) is by aggregating the probabilities
    of the sequences of stuck chain values from the initial through current 
    timestep's stuck chain value together which end in the same stuck chain
    value (i.e. if at timestep k, since we know s_0 = (0, 0, 0), will jointly
    estimate s_1^{(S)}, ..., s_k^{S}.  For each x that s_k^{S} could be: we
    will sum the probabilities of all sequences ending in s_k^{S} = x to get
    the probability that s_k^{S} = x.  *Does so under the AI's interventions.*

    This is the sequential estimator.  NOTE: this means this version multiplies
    the probability by the indicator that the AI would've taken its previous
    action at the previous stuck state.  Assumes deterministic policy.

    NOTE: this version finds the probability of the last stuck state in
    a sequence being x for any x by using dynamic programming, making it
    significantly faster than the BFS version (it is extremely fast).

    NOTE: for long trajectories/sequences, a version of this function which
    computes the log of the probabilities should be made.

    Parameters:
    -----------
    AI_traj_dataset: dict
        holds a trajectory rolled out from the AI MDP.  Contains the 
        following key-value pairs:
        "states": 1-D np.ndarray of AI states (as ints in [0, S-1])
        "actions": 1-D np.ndarry of AI actions
        The index of the action corresponds to the index of the state it
        was taken at in the "states" list. (So this list will be of length 
        1 less than trajectory_states if reached absorbing state).
        NOTE: this contains just 1 trajectory, not multiple (this function
        is invalid if traj_dataset contains more than one dataset).
        NOTE: this function only relies on the progress chain states.

    T_AI: numpy array of dimensionality [S, A, S]
        the transition matrix for the AI MDP, A is the size of the AI action
        space, S is the size of the AI state space.

    Q: numpy array of dimensionality [S, A]. 
        each element is the action-value at a given state.
        can be None if `determ_pol` is True. 

    cur_AI_state: int
        the state/timestep for which we are estimating the stuck chain 
        value (viable state values: starts from 0 and ends at 
        len(traj_dataset[states]) - 1 = n - 1).
        NOTE: cur_state must be > 0 for this fn to work and is an AI state.

    pi_AI: np.ndarray
        np array containing the policy the AI agent is following.
        numpy array of dimensionality [S]. Each element is the optimal AI 
        action at AI state s.

    AI_dims: list or tuple (list-like)
        contains the length of progress chains 1 through the last one
        followed by the length of the stuck chain followed by the number of
        possible human actions.

    Returns:
    --------
    cur_stuck_ts_probs: np.ndarry of ints
        array of the values of the stuck chain estimates for the stuck
        state at timestep k (which is the current stuck state, s_k).
        Indices in the array corresponding to the stuck state value (0 through
        the end of the stuck chain), and elements are the probabilities of the
        current stuck state being the index value.
    """
    # arrays to store DP solution (iterating from estimating the stuck state
    # at timestep (ts) t=0 through the current timestep: cur_state)
    prev_stuck_ts_probs = np.zeros(AI_dims[2])
    cur_stuck_ts_probs = np.zeros(AI_dims[2])

    # s_0^{(S)} = 0 (the stuck state at timestep 0/the first state tuple is 0)
    prev_stuck_ts_probs[0] = 1
    cur_stuck_ts_probs[0] = 1

    # truncate trajectory dataset to just states 0 through cur_state
    AI_states_to_cur = AI_traj_dataset["states"][:cur_AI_state+1]
    AI_actions_to_cur = AI_traj_dataset["actions"][:cur_AI_state]

    # convert states from state indices to tuples
    AI_state_tuples_list = [index_to_state(state_idx, AI_dims) for state_idx in AI_states_to_cur]

    # goal is to compute: P[t][s] = at time t, probability of being at stuck state s
    # meaningless when s > t (since this isn't possible) or is past the end of the
    # stuck chain, so we don't compute those values (for example, P[0][1] is left undefined)
    # however, we only need access to the current timestep (t) and the previous timestep (t-1) so we
    # are working with 2 1D arrays corresponding to these 2 timesteps rather than a 2D array
    for t in range(1, cur_AI_state+1): # timestep through (including) current timestep
        prev_stuck_ts_probs = list(cur_stuck_ts_probs)
        cur_stuck_ts_probs = np.zeros(AI_dims[2])
        for s in range(min(t+1, AI_dims[2])): # stuck state value for current timestep
            for ps in {s - 1, s, s + 1}: # previous stuck state
                if ps < 0 or ps > t-1 or ps > AI_dims[2]-1:
                    # invalid: out of bounds
                    continue

                # P[t][s] += Pr(state s | previous state ps, time=t-1) * P[t-1][ps]

                # Get progress chain states for previous timestep, and make the
                # stuck state corresponding to it the previous stuck state being considered
                prev_AI_state_tuple = AI_state_tuples_list[t-1]
                prev_AI_state_tuple = prev_AI_state_tuple[:2] + (ps,) + (prev_AI_state_tuple[3],)
                prev_AI_state_idx = state_to_index(prev_AI_state_tuple, AI_dims)
                prev_AI_action = AI_actions_to_cur[t-1]

                # Get progress chain states for current timestep, and make the
                # stuck state corresponding to it the current stuck state being considered
                cur_AI_state_tuple = AI_state_tuples_list[t]
                cur_AI_state_tuple = cur_AI_state_tuple[:2] + (s,) + (cur_AI_state_tuple[3],)
                cur_AI_state_idx = state_to_index(cur_AI_state_tuple, AI_dims)

                # To get the probability of the stuck state being s given the previous
                # state tuple (where progress chain states are as observed in the trajectory
                # and the previous stuck chain state is the one we're considering, ps)
                # and current progress chain state (as observed in the trajectory),
                # we need to divide the probability of the current progress chain state
                # (as observed from our trajectory) and current stuck chain we're considering
                # given the previous stuck chain state and action (taken in the trajectory)
                # by the probability of the current progress chain state given the previous
                # state tuple (AKA definition of conditional probability w extra conditioning)
                cond_prob_numerator = T_AI[prev_AI_state_idx, prev_AI_action, cur_AI_state_idx]

                # need to sum over all possible stuck chain values for the
                # current progress chain state to get its marginal probability
                if cond_prob_numerator != 0:
                    cond_prob_denominator = 0
                    for stuck in range(AI_dims[2]):
                        AI_state_tuple = cur_AI_state_tuple[:2] + (stuck,) + (cur_AI_state_tuple[3],)
                        AI_state_idx = state_to_index(AI_state_tuple, AI_dims)
                        cond_prob_denominator += T_AI[prev_AI_state_idx, prev_AI_action, AI_state_idx]
                else:
                    cond_prob_denominator = 1 # doesn't matter what value it is, 
                    # 0 probability of event happening, just made it non-zero so get
                    # 0 / 1 = 0 (what will happen later)

                # need to multiply the current stuck chain transition probability by the
                # probability of the previous sequence of stuck chain values (leading up
                # to the previous stuck chain value we were at) given the progress chains 
                # we observed and actions we took

                cur_stuck_ts_prob_to_add = (cond_prob_numerator / cond_prob_denominator
                                            ) * prev_stuck_ts_probs[ps] * (pi_AI[prev_AI_state_idx] == prev_AI_action)

                cur_stuck_ts_probs[s] += cur_stuck_ts_prob_to_add

    return cur_stuck_ts_probs