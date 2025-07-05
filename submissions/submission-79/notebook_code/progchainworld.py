"""
File containing code for making human agent transition and reward matrices.
"""

import numpy as np
import math
from itertools import product
from notebook_code.rl_utils import index_to_state, state_to_index
from notebook_code.multichain_utils import iterate_binary_arrays

A = 3

def make_nchains_transitions_simplified(prog_chain_dims: list, stuck_chain_dim: int,
                   p_fwd_prog_chains: np.array, p_back_prog_chains: np.array,
                   p_fwd_stuck: float, p_back_stuck: float):
    """
    Makes a progress chain world consisting of n progress chains, 1 through n,
    and 1 stuck chain.  There are 2 absorbing states: one when hit the goal state
    (last state) on all of the progress chains simultaneously, one when hit the
    dropout state (last state) on the stuck chain.
    
    Action 0 is no work is done on either chain, and action i is work on chain i
    for i \in [1, n].

    NOTE: in this simplified version, when the human works on a progress chain,
    cannot move backwards on this progress chain: can only move fwd or stay put.
    
    There is also a stuck chain, where the last state of this chain is absorbing,
    which is probabilistic:

    If you work on a chain but fail to make progress or do no work, 
    you move forward on the stuck chain (if possible) with some probability
    or stay put; otherwise, (if you work on a chain and make progress), 
    you move back 1 on the stuck chain with some probability or stay put.
    NOTE: edge cases: working on a chain you're at the end of counts as progress;
    if at 0 on stuck chain can't go backwards on it (stay put)
    Makes the transition matrix for this setup.

    NOTE: f_i is probability of moving forward 1 on chain i when do work on it
    and r_i is probability of moving back 1 on chain i when do no work action

    Parameters:
    -----------
    prog_chain_dims: list of ints
        number of states in each progress chain

    stuck_chain_dim: int
        number of states in the stuck chain

    p_fwd_prog_chains: np.array of floats in [0, 1]
        probability of moving forward on each chain i if work is done on chain i
        when work is done on chain i: have an f_i probability of moving forward on
        chain i (if not already at an end), have a 1 - f_i probability of staying 
        put on chain i.

    p_back_prog_chains: np.array of floats in [0, 1]
        probability of moving back 1 (relapsing) on each chain i
        if not 0: 
            when no work is done on chain i, have a r_i probability 
            of moving backwards 1 state on chain 1, and a 1 - r_i
            probability of staying put on chain i.

        otherwise: will just stay put on chain i if do not move forward.

    p_fwd_stuck: float in [0, 1]
        probability of moving forward 1 on the stuck chain if did not
        work or did work but failed to make progress.

    p_back_stuck: float in [0, 1]
        probability of moving back 1 on the stuck chain if did work
        and made progress on the chain worked on.

    Returns:
    --------
    T: 3D list of the form S x A x S'
        transition matrix
    """
    S = math.prod(prog_chain_dims) * stuck_chain_dim

    end_prog_chains = [prog_chain_dim - 1 for prog_chain_dim in prog_chain_dims]
    end_stuck = stuck_chain_dim - 1

    # Assuming if not numpy array, is list, turn to numpy array
    if not isinstance(p_fwd_prog_chains, np.ndarray):
        p_fwd_prog_chains = np.array(p_fwd_prog_chains)
    if not isinstance(p_back_prog_chains, np.ndarray):
        p_back_prog_chains = np.array(p_back_prog_chains)

    # list of transition matrices, one per action (do no work, followed by work on
    # progress chain 1 through n)
    T_by_action = [np.zeros((S, S)) for _ in range(1 + len(prog_chain_dims) )]


    # FIRST: fill in the no action transition matrix
    # print(f"no work transitions")
    T_by_action[0] = make_nchains_no_work_transitions(prog_chain_dims, stuck_chain_dim,
                                     p_back_prog_chains, p_fwd_stuck)
    

    # SECOND: Fill in transition matrices for working on progress chain i
    # print(f"work transitions")
    # Iterate through all possible current states, disregarding stuck chain for now
    # gen the range of possible states for each chain
    all_chain_dims = prog_chain_dims + [stuck_chain_dim]
    chain_ranges = [range(c) for c in all_chain_dims]
    # iter through all possible states (combos of state on each chain)
    all_states_tuples = list(product(*chain_ranges))
    for cur_state in all_states_tuples:
        cur_state_idx = state_to_index(cur_state, all_chain_dims)
        cur_state = np.array(cur_state)
        cur_prog_state = cur_state[:-1]

        # at positions where started at 0: have to stay at 0 in next state
        # update transition probability for not going back on these progress
        # chains accordingly (set to 1)
        unworked_no_back_probs = 1 - p_back_prog_chains
        unworked_no_back_probs[cur_prog_state == 0] = 1

        # The n-1 chains not being worked on: can stay put or move back,
        # iter thru all possible combos of next states for these n-1 chains
        for next_unworked_progmove in iterate_binary_arrays(len(prog_chain_dims)-1):
            # got potential moves for the n-1 unworked chains: now consider
            # working on each chain i, i \in [1, n]
            for i in range(len(prog_chain_dims)):
                # Create a mask to exclude index i (to exclude chain i)
                mask = np.arange(len(prog_chain_dims)) != i
                # Check if valid next state for the unworked chains
                next_unworked_progstate = cur_prog_state[mask] - next_unworked_progmove
                if np.isin(-1, next_unworked_progstate):
                    # invalid state
                    continue
                # valid state 
                # get probability of moving to the next state on the unworked chains
                # (these are all progress chains except chain i)
            
                ## fill in probability of moving back 1 in each unworked chain where do so
                unworked_prog_back_probs = next_unworked_progmove * p_back_prog_chains[mask]
                ## fill in probability of staying put in each unworked chain where do so
                unworked_prog_put_probs = (1-next_unworked_progmove) * unworked_no_back_probs[mask]
                ## put the probabilities together into 1 vector
                next_unworked_progstate_probs = unworked_prog_back_probs + unworked_prog_put_probs
                ## multiply the probability of transitioning to each chain's next
                ## state to get the compound probability of transitioning to the next state
                next_uworked_progstate_prob = np.prod(next_unworked_progstate_probs)

                # now handle each possible move on the chain being worked on:
                # can go forward (if not at the end of it) or stay put
                if cur_state[i] == end_prog_chains[i]:
                    # cannot go forward, would stay put in place of moving forward,
                    # so only option is stay put

                    # stay put:
                    p_put_worked_chain = 1
                    next_progstate = np.insert(next_unworked_progstate, i, cur_state[i])
                    next_progstate_prob = next_uworked_progstate_prob * p_put_worked_chain
                    # handle stuck chain transitions and update transition 
                    # matrix for working on chain i in this case
                    T_by_action[i+1] = update_work_transition(False, i, T_by_action[i+1], all_chain_dims, cur_state, 
                                                              cur_state_idx, next_progstate, next_progstate_prob,
                                                              p_fwd_stuck, p_back_stuck)

                else:
                    # can go forward or stay put with the standard probabilities

                    # moved forward: succeeded to mv forward
                    p_fwd_worked_chain = p_fwd_prog_chains[i]
                    next_progstate = np.insert(next_unworked_progstate, i, cur_state[i]+1)
                    next_progstate_prob = next_uworked_progstate_prob * p_fwd_worked_chain
                    # handle stuck chain transitions and update transition 
                    # matrix for working on chain i in this case
                    T_by_action[i+1] = update_work_transition(True, i, T_by_action[i+1], all_chain_dims, cur_state, 
                                                              cur_state_idx, next_progstate, next_progstate_prob,
                                                              p_fwd_stuck, p_back_stuck)
                    
                    # stayed put: failed to mv forward 
                    p_put_worked_chain = 1-p_fwd_prog_chains[i]
                    next_progstate = np.insert(next_unworked_progstate, i, cur_state[i])
                    next_progstate_prob = next_uworked_progstate_prob * p_put_worked_chain
                    # handle stuck chain transitions and update transition 
                    # matrix for working on chain i in this case
                    T_by_action[i+1] = update_work_transition(False, i, T_by_action[i+1], all_chain_dims, cur_state, 
                                                              cur_state_idx, next_progstate, next_progstate_prob,
                                                              p_fwd_stuck, p_back_stuck)
                    

    # THIRD: handle absorbing states
    T = np.transpose(T_by_action, axes = [1, 0, 2])

    # Make goal state absorbing
    for s_stuck in range(stuck_chain_dim):
        goal_state = end_prog_chains + [s_stuck]
        s_goal = state_to_index(goal_state, all_chain_dims) 

        # If hit goal state on both chains, stay there
        T[s_goal, :, :] =  0
        T[s_goal, :, s_goal] = 1

    # Make dropout state at end of stuck chain absorbing:
    progchain_ranges = [range(c) for c in prog_chain_dims]
    # iter through all possible states (combos of state on each chain)
    all_progstates_tuples = list(product(*progchain_ranges))
    for cur_progstate in all_progstates_tuples:
        dropout_state = cur_progstate + (end_stuck,)
        s_dropout = state_to_index(dropout_state, all_chain_dims)

        # If hit dropout state at end of stuck chain, stay there
        T[s_dropout, :, :] = 0
        T[s_dropout, :, s_dropout] = 1

    return T




def make_nchains_no_work_transitions(prog_chain_dims: list, stuck_chain_dim: int,
                                     p_back_prog_chains: np.array, p_fwd_stuck: float):
    """
    Helper function for making a progress chain world consisting of n progress chains, 
    1 through n, and 1 stuck chain where there are 2 absorbing states: one when hit 
    the goal state (last state) on all of the progress chains simultaneously, one when 
    hit the dropout state (last state) on the stuck chain.

    This helper function creates the transition matrix for doing no work on any chain
    (action 0).
    
    There is also a stuck chain, where the last state of this chain is absorbing,
    which is probabilistic:

    If you work on a chain but fail to make progress or do no work, 
    you move forward on the stuck chain (if possible) with some probability
    or stay put; otherwise, (if you work on a chain and make progress), 
    you move back 1 on the stuck chain with some probability or stay put.
    NOTE: edge cases: working on a chain you're at the end of counts as progress;
    if at 0 on stuck chain can't go backwards on it (stay put)
    Makes the transition matrix for this setup.

    NOTE: f_i is probability of moving forward 1 on chain i when do work on it
    and r_i is probability of moving back 1 on chain i when do no work action or
    (optionally) work on it but fail to move forward

    Parameters:
    -----------
    prog_chain_dims: list of ints
        number of states in each progress chain

    stuck_chain_dim: int
        number of states in the stuck chain

    p_back_prog_chains: np.array of floats in [0, 1]
        probability of moving back 1 (relapsing) on each chain i
        if not 0: 
            when work is done on chain i: have a 1 - f_i probability
            of not moving forward.  This is optionally broken into a 
            (1-f_i) * r_i probability of moving 1 state
            backwards on chain i and a (1-f_i) * (1-r_i) 
            probability of staying put on chain i, or in the simplified
            version, (1-f_i) probability of staying put on chain i.
        
            when no work is done on chain i, have a r_i probability 
            of moving backwards 1 state on chain 1, and a 1 - r_i
            probability of staying put on chain i.

        otherwise: will just stay put on chain i if do not move forward.

    p_fwd_stuck: float in [0, 1]
        probability of moving forward 1 on the stuck chain if did not
        work or did work but failed to make progress.

    Returns:
    --------
    T_no_work: 2D list of the form S x S'
        transition matrix for action 0 (do no work)
    """
    S = math.prod(prog_chain_dims) * stuck_chain_dim
    T_no_work = np.zeros((S, S))
    end_stuck = stuck_chain_dim - 1
    # Iterate through all possible current states, disregarding stuck chain for now
    # gen the range of possible states for each chain
    all_chain_dims = prog_chain_dims + [stuck_chain_dim]
    chain_ranges = [range(c) for c in all_chain_dims]
    # iter through all possible states (combos of state on each chain)
    all_states_tuples = list(product(*chain_ranges))
    for cur_state in all_states_tuples:
        cur_state_idx = state_to_index(cur_state, all_chain_dims)
        cur_state = np.array(cur_state)
        cur_prog_state = cur_state[:-1]
        cur_stuck_state = cur_state[-1]
        # at positions where started at 0: have to stay at 0 in next state
        # update transition probability for not going back on these progress
        # chains accordingly (set to 1)
        updated_no_back_probs = 1 - p_back_prog_chains
        updated_no_back_probs[cur_prog_state == 0] = 1

        # iterate thru all possible combinations of moving back 1 and
        # staying put on each chain, skipping invalid states (going back
        # 1 when started at 0)
        for next_progmove in iterate_binary_arrays(len(prog_chain_dims)):
            next_progstate = cur_prog_state - next_progmove
            if np.isin(-1, next_progstate):
                # invalid state
                continue

            # valid state: compute transition probability
            ## fill in probability of moving back 1 in each chain where do so
            prog_back_probs = next_progmove * p_back_prog_chains
            ## fill in probability of staying put in each chain where do so
            prog_put_probs = (1-next_progmove) * updated_no_back_probs
            ## put the probabilities together into 1 vector
            next_progstate_probs = prog_back_probs + prog_put_probs
            ## multiply the probability of transitioning to each chain's next
            ## state to get the compound probability of transitioning to the next state
            next_progstate_prob = np.prod(next_progstate_probs)

            # now handle next stuck state, but if currently at end of stuck
            # chain, cannot move fwd on it, must stay put
            if cur_stuck_state == end_stuck:
                next_state = np.append(next_progstate, cur_stuck_state)
                next_state_idx = state_to_index(next_state, all_chain_dims)
                next_state_prob = next_progstate_prob * 1
                
                T_no_work[cur_state_idx, next_state_idx] = next_state_prob

            else:
                # not at end of stuck chain, can move forward on it or stay put
                # move forward on stuck chain
                next_state_stuck_fwd = np.append(next_progstate, cur_stuck_state + 1)
                next_state_idx = state_to_index(next_state_stuck_fwd, all_chain_dims)
                next_state_prob = next_progstate_prob * p_fwd_stuck
                T_no_work[cur_state_idx, next_state_idx] = next_state_prob

                # stay put on stuck chain
                next_state_stuck_put = np.append(next_progstate, cur_stuck_state)
                next_state_idx = state_to_index(next_state_stuck_put, all_chain_dims)
                next_state_prob = next_progstate_prob * (1-p_fwd_stuck)
                T_no_work[cur_state_idx, next_state_idx] = next_state_prob

    return T_no_work


def update_work_transition(made_progress: bool, chain_worked_on: int, 
                           T: np.array, all_chain_dims: list,
                           cur_state: np.array, cur_state_idx: int, 
                           next_progstate: np.array, next_progstate_prob: float,
                           p_fwd_stuck: float, p_back_stuck: float):
    """
    Helper function for making a progress chain world consisting of n progress chains, 
    1 through n, and 1 stuck chain where there are 2 absorbing states: one when hit 
    the goal state (last state) on all of the progress chains simultaneously, one when 
    hit the dropout state (last state) on the stuck chain.

    This helper function updates the transition matrix for working on chain i,
    handling the different stuck state transitions.
    
    The stuck chain, where the last state of this chain is absorbing, 
    is probabilistic:

    If you work on a chain but fail to make progress or do no work, 
    you move forward on the stuck chain (if possible) with some probability
    or stay put; otherwise, (if you work on a chain and make progress), 
    you move back 1 on the stuck chain with some probability or stay put.
    NOTE: edge cases: working on a chain you're at the end of (and not losing
    progress on it, if this is an option) results in being guaranteed to stay
    put on the stuck chain;
    if at 0 on stuck chain can't go backwards on it (stay put)
    Makes the transition matrix for this setup.

    NOTE: f_i is probability of moving forward 1 on chain i when do work on it
    and r_i is probability of moving back 1 on chain i when do no work action or
    (optionally) work on it but fail to move forward

    Parameters:
    -----------
    made_progress: bool
        if True: succeeded to make progress on the chain worked on
        otherwise: failed to make progres on the chain worked on

    chain_worked_on: int
        progress chain worked on to get to next state (a number between 
        0 and n-1, assuming there are n progress chains, 1st progress
        chain is idx 0)

    T: 2D numpy array of size S x S
        where S is the number of states in the progress chain world
        current transition matrix for doing work on a particular chain

    all_chain_dims: list
        list where each index i is the length of chain i (includes all
        progress chains and stuck chain)

    cur_state: 1D numpy array of ints
        the current state on each chain i, where the first n chains are 
        the progress chains, and the last chain is the stuck chain

    cur_state_idx: int
        the current state as an integer between 0 and (SxS) - 1

    next_progstate: 1D numpy array of ints
        the next state on each progress chain i (excludes the stuck chain)
    
    next_progstate_prob: float
        the probability of transitioning to the next state on the progress
        chains (excludes the stuck chain)

    p_fwd_stuck: float in [0, 1]
        probability of moving forward 1 on the stuck chain if did not
        work or did work but failed to make progress.

    p_back_stuck: float in [0, 1]
        probability of moving back 1 on the stuck chain if did work
        and made progress on the chain worked on.

    Returns:
    --------
    T: 2D list of the form S x S'
        updated transition matrix for action i, i \in [1, n]
        (work on progress chain i), updated from inputted T
    """
    cur_stuck_state = cur_state[-1]
    end_stuck = all_chain_dims[-1] - 1 # last state index in the stuck chain

    if made_progress:
        # move back 1 or stay put on the stuck chain
        if cur_stuck_state == 0:
            # cannot move back 1: must stay put on stuck chain
            next_state = np.append(next_progstate, cur_stuck_state)
            next_state_idx = state_to_index(next_state, all_chain_dims)
            next_state_prob = next_progstate_prob * 1
            T[cur_state_idx, next_state_idx] = next_state_prob

        else:
            # can move back 1 or stay put on stuck chain

            # stay put
            next_state_put_stuck = np.append(next_progstate, cur_stuck_state)
            next_state_idx = state_to_index(next_state_put_stuck, all_chain_dims)
            next_state_prob = next_progstate_prob * (1-p_back_stuck)
            T[cur_state_idx, next_state_idx] = next_state_prob

            # move back 1
            next_state_back_stuck = np.append(next_progstate, cur_stuck_state-1)
            next_state_idx = state_to_index(next_state_back_stuck, all_chain_dims)
            next_state_prob = next_progstate_prob * p_back_stuck
            T[cur_state_idx, next_state_idx] = next_state_prob

    else:
        # failed to make progress when worked on progress chain:
        # move forward 1 or stay put on the stuck chain
        if (cur_stuck_state == end_stuck) or (
            (cur_state[chain_worked_on] == (all_chain_dims[chain_worked_on] - 1)) and (
                next_progstate[chain_worked_on] == cur_state[chain_worked_on])):
            # in first case: cannot move forward 1 on stuck chain: must stay put on the stuck chain
            # in second case: at end of prog chain being worked on and stayed there: 
            # decided means stay put on stuck chain (the thought was that you  
            # technically failed to make progress but you couldn't make any
            # more so don't let it negatively impact the stuckness level)
            next_state = np.append(next_progstate, cur_stuck_state)
            next_state_idx = state_to_index(next_state, all_chain_dims)
            next_state_prob = next_progstate_prob * 1
            T[cur_state_idx, next_state_idx] = next_state_prob

        else:
            # can move forward 1 or stay put on stuck chain

            # stay put
            next_state_put_stuck = np.append(next_progstate, cur_stuck_state)
            next_state_idx = state_to_index(next_state_put_stuck, all_chain_dims)
            next_state_prob = next_progstate_prob * (1-p_fwd_stuck)
            T[cur_state_idx, next_state_idx] = next_state_prob

            # move back 1
            next_state_fwd_stuck = np.append(next_progstate, cur_stuck_state+1)
            next_state_idx = state_to_index(next_state_fwd_stuck, all_chain_dims)
            next_state_prob = next_progstate_prob * p_fwd_stuck
            T[cur_state_idx, next_state_idx] = next_state_prob

    return T


def make_rewards_stuck(chain_1_dim: int, chain_2_dim: int, 
                 stuck_chain_dim: int, r_goal: float = 10,
                 r_dropout: float = -10):
    """
    Makes the rewards for 2 goal chains and 1 stuck chain where action 0
    is no work is done on either chain, action 1 is work is done on chain 
    1 and no work done on chain 2, and action 2 is work is done on chain 2 
    and no work done on chain 1.

    Reward for goal given if reach end of both goal chains.
    Penalty for dropping out if reach end of stuck chain.

    Parameters:
    -----------
    chain_1_dim: int
        number of states in chain 1

    chain_2_dim: int
        number of states in chain 2

    stuck_chain_dim: int
        number of states in the stuck chain

    r_goal: float
        reward for achieving the absorbing goal state: reaching the
        end of both chain 1 and chain 2

    r_dropout: float
        reward for reaching the dropout state at the end of the stuck
        chain.  Should be negative (to be a penalty).

    Returns:
    --------
    R: 3D list of the form S x A x S'
        reward matrix
    """
    S = chain_1_dim * chain_2_dim * stuck_chain_dim
    dims = (chain_1_dim, chain_2_dim, stuck_chain_dim)
    
    R = np.zeros((S, A, S))
    
    end_1 = chain_1_dim - 1
    end_2 = chain_2_dim - 1
    end_stuck = stuck_chain_dim - 1

    # Dropout reward
    for s1 in range(chain_1_dim):
        for s2 in range(chain_2_dim):
            s_dropout = state_to_index((s1, s2, end_stuck), dims)
            R[:, :, s_dropout] = r_dropout

            # Dropout state is absorbing
            R[s_dropout, :, :] = 0

    # Goal reward 
    for s3 in range(stuck_chain_dim):
        s_goal = state_to_index((end_1, end_2, s3), dims)
        # as a note, this includes (end 1, end 2, end stuck chain)
        # ((but if reached stuck chain b4 reached end 1 and end 2,
        #  would've already terminated; so this is if reach end of
        #  all 3 at same time))
        R[:, :, s_goal] = r_goal

        # goal state is absorbing
        R[s_goal, :, :] = 0


    return R


def make_rewards_stuck_nchains(prog_chain_dims: list, stuck_chain_dim: int, 
                                r_goal: float = 10, r_dropout: float = -10,
                                R: np.ndarray = None):
    """
    Makes the rewards for n progress chains and 1 stuck chain where action 0
    is no work is done on either chain, and each action i is do work on
    progress chain i.

    Parameters:
    -----------
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

    R: np.ndarray or None
        if None: instantiates R as a 0s array of shape S x A x S'
        where S is the number of states in the human MDP, A is the
        number of actions in the human's action space (do no work or
        work on progress chain i for any i in [1, num progress chains])

        else: uses the inputted R--which should be of shape S x A x S'--
        as the initial R matrix, which gets modified to handle the absorbing
        goal and dropout states.
    """
    S = math.prod(prog_chain_dims) * stuck_chain_dim

    if R is None:
        # for making human agent's reward matrix
        # actions are do no work or work on progress chain i, for each progress chain
        A_h = len(prog_chain_dims) + 1 
        R = np.zeros((S, A_h, S))

    end_prog_chains = [prog_chain_dim - 1 for prog_chain_dim in prog_chain_dims]
    end_stuck = stuck_chain_dim - 1
    all_chain_dims = prog_chain_dims + [stuck_chain_dim]


    # Make penalty for reaching end of stuck chain (and 0 reward
    # once get there since its absorbing), noting that can be
    # at end of stuck chain for any possible combo of progress chain states
    progchain_ranges = [range(c) for c in prog_chain_dims]
    # iter through all possible states (combos of state on each chain)
    all_progstates_tuples = list(product(*progchain_ranges))
    for cur_progstate in all_progstates_tuples:
        if list(cur_progstate) == end_prog_chains:
            # successfully reached end of progress chains (if this case
            # happens, it would be reaching end of prog chains and dropping
            # out at same time bc they're individually absorbing--did suceed
            # in getting to goal state so should get only the positive rwd
            # for doing so)
            continue
        dropout_state = cur_progstate + (end_stuck,)
        s_dropout = state_to_index(dropout_state, all_chain_dims)

        # reward (penalty) for dropping out
        R[:, :, s_dropout] += r_dropout

        # Dropout state is absorbing
        R[s_dropout, :, :] = 0


    # Make reward for reaching goal state (and 0 reward once get there since
    # its abosrbing), noting that goal state occurs when at end of all progress
    # chains, for any stuck state
    for s_stuck in range(stuck_chain_dim):
        goal_state = end_prog_chains + [s_stuck]
        s_goal = state_to_index(goal_state, all_chain_dims) 

        # reward for hitting goal state
        R[:, :, s_goal] += r_goal

        # goal state is absorbing
        R[s_goal, :, :] = 0

    return R


def human_pol_from_ai_pol(pi_ai: np.ndarray, pi_humans: list, chain_dims: list, A_h: int):
    """
    Gets the human agent's policy under the AI agent's intervention policy.

    Let the human policy being constructed be called pi_hai. 
    At a human state s_h, 
    if the AI agent takes action = 0 (do nothing): 
        pi_hai(s_h) = action dictated by human policy learned under 
        no intervention at s_h
    if the AI agent takes action = i (do intervention i):
        pi_hai(s_h) = action dictated by human policy learned under
        intervention i at s_h

    Parameters:
    -----------
    pi_ai: np.ndarray
        np array of shape S_AI x A_AI, where S_AI = S_h * A_h (the size of
        the human agent's state space times its action space), A_AI is the
        AI agent's action space (0: do nothing, i: do intervention i)
    pi_humans: list of np.ndarrays
        list of human policies under the different AI interventions
        each element i corresponds to the ith element of pi_ai
        (that is: element 0 is pi_human learned under the human agent's
        baseline transition matrix and discount factor where the AI did 
        not intervene, and each element i is pi_human learned under the 
        AI doing intervention i)
    chain_dims: list
        array of length k,
        each index i corresponds to the number of states in chain i
    A_h: int
        number of actions in the human's action space (under which
        the human policies and AI policy were learned)

    Returns:
    --------
    pi_hai: np.ndarray
        1D array of length equal to S_h, where each index is a human state
        s_h and the value is what action the human agent takes at that state,
        as dictated by the optimal human policy under the AI's intervention for 
        that state (see function description above).
    """
    S_h = math.prod(chain_dims)
    pi_hai = np.zeros(S_h)
    ai_dims = chain_dims + [A_h]
    # index of pi_ai is the AI agent's state
    for s_ai in range(len(pi_ai)):
        a_ai = pi_ai[s_ai]
        # get human's state
        s_ai_tuple = index_to_state(s_ai, ai_dims)
        # remove a_h t-1 (human's action at the previous timestep) from the state
        # to get s_h t (human's action at the current timestep)
        s_h_tuple = s_ai_tuple[:-1]
        s_h_idx = state_to_index(s_h_tuple, chain_dims)
        # get human's action under human policy selected by AI agent for 
        # the current human state
        a_h = pi_humans[a_ai][s_h_idx]

        # under AI policy, human policy takes action a_h at state s_h
        pi_hai[s_h_idx] = a_h

    return pi_hai


def human_states_w_ai_inter(pi_ai: np.ndarray, pi_humans: list, chain_dims: list, A_h: int):
    """
    For each human state, gets what the AI action is for it.

     Parameters:
    -----------
    pi_ai: np.ndarray
        np array of shape S_AI x A_AI, where S_AI = S_h * A_h (the size of
        the human agent's state space times its action space), A_AI is the
        AI agent's action space (0: do nothing, i: do intervention i)
    pi_humans: list of np.ndarrays
        list of human policies under the different AI interventions
        each element i corresponds to the ith element of pi_ai
        (that is: element 0 is pi_human learned under the human agent's
        baseline transition matrix and discount factor where the AI did 
        not intervene, and each element i is pi_human learned under the 
        AI doing intervention i)
    chain_dims: list
        array of length k,
        each index i corresponds to the number of states in chain i
    A_h: int
        number of actions in the human's action space (under which
        the human policies and AI policy were learned)

    Returns:
    --------
    pi_aih: np.ndarray
        1D array of length equal to S_h, where each index is a human state
        s_h and the value is what action the AI agent takes at that state
        under the AI agent's optimal policy.

    ai_action_at_human_state: dict
        keys are the AI actions (0 is do nothing, i is intervention i)
        values are the human states, as tuples, that the AI action occurred in
        under the AI agent's optimal policy.
    """
    S_h = math.prod(chain_dims)
    pi_aih = np.zeros(S_h)
    ai_dims = chain_dims + [A_h]
    ai_action_at_human_state = {i: [] for i in range(np.max(pi_ai)+1)}
    # index of pi_ai is the AI agent's state
    for s_ai in range(len(pi_ai)):
        a_ai = pi_ai[s_ai]
        # get human's state
        s_ai_tuple = index_to_state(s_ai, ai_dims)

        # remove a_h (human's action at the previous timestep) from the state
        # to get s_h (human's action at the current timestep)
        s_h_tuple = s_ai_tuple[:-1]
        s_h_idx = state_to_index(s_h_tuple, chain_dims)

        # AI took action a_ai at human state s_h (note: this holds regardless
        # of what a_h at the previous timestep was)
        pi_aih[s_h_idx] = a_ai
        if s_h_tuple not in ai_action_at_human_state[a_ai]: 
            # if to deal w s_h repeats in AI policy (once per a_h, but for the
            # same s_h, each (s_h, a_h) pair corresponds to the same AI action)
            ai_action_at_human_state[a_ai].append(s_h_tuple)

    return pi_aih, ai_action_at_human_state