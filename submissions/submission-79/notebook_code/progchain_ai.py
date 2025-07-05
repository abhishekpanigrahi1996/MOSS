"""
File containing code for making AI agent's transition and reward matrices.
"""

import numpy as np
from scipy.special import softmax
from notebook_code.multichain_utils import multi_hot_policy
from notebook_code.progchainworld import make_rewards_stuck_nchains

##########
# AI MDP #
##########

def select_softmax(Q, noise = 0.1): 
    Q_ = Q / noise
    return  softmax(Q_, axis = 1)

def select_max(Q): 
     # Find the index of the maximum Q-value for each state
    best_actions = np.argmax(Q, axis=1)

    # Create a one-hot encoding matrix for the best actions
    policy = np.eye(Q.shape[1])[best_actions]

    return policy


def build_T_AI_from_human_policy(T_human: np.ndarray, pi_human: np.ndarray):
    """
    Builds the AI agent's transition matrix under a particular intervention
    action (either doing nothing or intervening in a particular way).  Takes
    in the human agent's transition matrix and policy under the human parameter
    updates resulting from the AI intervention.

    Parameters:
    -----------
    T_human: np.ndarray
        the human agent's transition matrix under the AI intervention action
        for which the AI's transition matrix is being made.  Is of shape
        S_h x A_h x S'_h (= N_states x N_actions x N_states)

    pi_human: np.ndarray
        the human agent's policy under the AI intervention action for
        which the AI's transition matrix is being made.  Is of shape
        S_h x A_h (= N_states x N_actions)

    Returns:
    --------
    T_app_a: np.ndarray
        the AI agent's transition matrix for the particular intervention
        action being considered.  Is of shape S_AI x S_AI (= S_app x S_app
        = (S_human * A_human) x (S'_human * A'_human))
    """
    N_states, N_actions = pi_human.shape
    S_app = N_states * N_actions
    T_app_a =  (pi_human.flatten().reshape(-1, 1) * T_human.reshape(S_app, N_states)) # S_app x W', p(w' | w, a_user')p(a_user' | w, a_app)
    T_app_a =  T_app_a.reshape(N_states, N_actions, N_states) # W x A_user' x W'
    T_app_a = np.repeat(T_app_a[None, :], N_actions, axis = 0) # A_user x W x A_user' x W' 
    T_app_a = np.transpose(T_app_a, axes = (1, 0, 3, 2)) # W x A_user x W' x A_user'
    T_app_a = T_app_a.reshape(S_app, S_app) # S_app x S_app = (S_human * A_human x S_human' * A_human')
    
    return T_app_a


def build_T_ai(T_humans: list, pi_humans: list, multi_hot_pis: bool):
    """
    Builds the AI's transition matrix based on the human transition matrix
    and policy under each AI intervention (or lack of intervention).

    Parameters:
    -----------
    T_humans: list of np.arrays
        list of human transition matrices under the different AI interventions
        the 0th element should be the human transition matrix under no invention
        each element i should correspond to AI action i
        human transition matrices should all be of size S_h x A_h x S_h
        where h represents human, S is the number of states, A is the number
        of actions.

    pi_humans: list of np.arrays
        list of human plicies under the different AI interventions
        each element i corresponds to T_humans[i]
    
    multi_hot_pis: bool
        if True: pi_humans are 1D arrays of length S_h, where each element
        is the action to take at the state (which equals the index). 
        This function will multi-hot the pi_humans to make them 2D arrays 
        of size S_h x A_h (so each row=state has a 1 corresponding to the
        action=column to take at the state).

        else: pi_humans are already these multi-hotted 2D arrays of S_h x A_h 

    Returns:
    --------
    T_AI: np.ndarray
        np array of shape S_AI x A_AI x S'_AI
        which is the transition matrix for each AI state and action
        NOTE: AI states take the form (s^{(h)}_t, a_^{(h)}_{t-1}) where 
        s^h is the human state at time t, a^h is the human action at time
        t-1.
    """
    if multi_hot_pis:
        pi_humans = pi_humans.copy()
        A_h = T_humans[0].shape[1]
        for i in range(len(pi_humans)):
            pi_humans[i] = multi_hot_policy(pi_humans[i], A_h)
    
    # Get transition matrix from S_AI to S'_AI for each AI action separately
    T_AI = [build_T_AI_from_human_policy(T_humans[i], pi_humans[i]) for i in range(len(pi_humans))]

    # Rearrange so that T_AI takes form S_AI x A_AI x S_AI
    T_AI = np.transpose(np.array(T_AI), axes = (1, 0, 2))
    return T_AI


def build_R_ai(n_AI_actions: int, R_human: np.ndarray, 
               prog_chain_dims: list, stuck_chain_dim: int, 
                 r_goal: float = 1, r_dropout: float = -1,
                 r_intervention: float = -0.1, r_timestep: float = 0):
    """
    Creates the AI's reward matrix.  It shares the same structure as the human's
    reward matrix (with a reward for the human dropping out or achieving their goal),
    but has a (penalty) reward for the AI intervening.

    Parameters:
    -----------
    n_AI_actions: int
        the number of actions in the AI agent's action space

    R_human: np.ndarray
        the human's reward matrix, is of shape S_h x A_h x S'_h, where S_h
        is the human's state space, A_h is the human's action space
        
    prog_chain_dims: list of ints
        number of states in each progress chain

    stuck_chain_dim: int
        number of states in the stuck chain

    r_goal: float
        reward for achieving the absorbing goal state: reaching the
        end of both chain 1 and chain 2

    r_dropout: float
        reward for reaching the dropout state at the end of the stuck
        chain.  Should be negative (to be a penalty).

    r_intervention: float
        reward for the AI taking an intervening action (not doing nothing).
        Should be negative (to be a penalty).

    r_timestep: float
        reward for each timestep.  Should be negative (to be a penalty for
        taking more timesteps). NOTE: applying this to the human MDP created
        weird behavior so it may be better to leave this 0.

    Returns:
    --------
    R_ai: np.ndarray of the form S_AI x A_AI x S'_AI
        AI's reward matrix
        a_ai = 0 corresponds to doing nothing
        a_ai = i, i \neq 0 corresponds to taking intervention action i
    """
    N_states, N_actions, _ = R_human.shape
    # Initialize AI reward matrix with cost of intervention
    R_ai = np.ones((N_states, n_AI_actions, N_states)) * (r_timestep + r_intervention)
    # Remove cost of intervention for AI action of do nothing (which is action 0)
    R_ai[:, 0, :] = r_timestep

    # update AI reward matrix to get reward when human agent reaches goal 
    # state or dropout state, and 0 reward once hit either (since they are absorbing)
    R_ai = make_rewards_stuck_nchains(prog_chain_dims, stuck_chain_dim, r_goal, r_dropout, R_ai)

    # Rearrange to match AI state space S_AI = S_h * A_h (where S_h is human
    # state space and A_h is human action space)
    R_ai = np.array([R_ai for i in range(N_actions * N_actions)])
    R_ai = R_ai.reshape(N_actions, N_actions, N_states, n_AI_actions, N_states)
    R_ai = np.transpose(R_ai, axes = (2, 0, 3, 4, 1)).reshape(N_actions * N_states, n_AI_actions, N_actions * N_states)
    return R_ai