import numpy as np

def state_to_index(s, dim_sizes): 
    '''
    Given a discrete state (e.g. [1, 1, 0]) returns single index (e.g. the index of the flattened state space)

    Inputs: 
    - s: a tuple or array representing the state. 
    - dim_sizes: the size of each (discrete) dimension

    Output: an int representing the (flattened) state index
    '''
    return np.ravel_multi_index(s, dim_sizes)


def index_to_state(ind, dim_sizes): 
    '''
    Given a single index (e.g. the index of the flattened state space), returns a discrete state tuple
    
    Inputs: 
    - s: an int representing the (flattened) state index 
    - dim_sizes: the size of each (discrete) dimension

    Output: a tuple representing the state
    '''
    return np.unravel_index(ind, dim_sizes)  


def value_iteration(T, R, gamma, delta = 0.1, verbose = False): 
    '''
    Value iteration to solve for the optimal value function. 

    Inputs: 
    - T: numpy array of dimensionality [S, A, S]
    - R: numpy array of dimensionality [S, A, S]
    - gamma: float discount factor from (0, 1)
    
    Outputs:
    - pi: numpy array of dimensionality [S]. Each element is the optimal action at state s.  
    - Q: numpy array of dimensionality [S, A]. Each element is the action-value at a given state. 
    - V: numpy array of dimensionality [S]. Each element is the value at a given state. 
    '''
    n_actions = T.shape[1]
    n_states = T.shape[0]
    V = np.zeros((n_states, ))
    Q = np.zeros((n_states, n_actions))
    max_change = delta
    i = 0
    while max_change >= delta: 
        max_change = 0.
        for s in range(Q.shape[0]):
            for a in range(n_actions): 
                '''
                indices, s_next_probs = T.get_transition_probabilities(s, a)
                r_next = np.array([r(s, a, s_next) for s_next in indices])
                Q[s, a] = np.dot(gamma * V[indices] + r_next, s_next_probs)
                '''
                Q[s, a] = np.dot(T[s, a, :], gamma * V + R[s, a, :])
                
            v = V[s]
            V[s] = np.amax(Q[s, :])
            max_change = max(max_change, np.abs(v - V[s]))
           

        if verbose: 
            print("iteration {}: max change {}".format(i, max_change))
        i+=1  

    pi = np.argmax(Q, axis = 1)

    return pi, Q, V