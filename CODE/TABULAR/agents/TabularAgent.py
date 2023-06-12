import numpy as np

'''
tabular_dim: shape of state transition 'tensor' for the MDP.
             See David Silver Lecture on Markov Decision Processes
'''

class TabularAgent:
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0, alpha=0.1, map_state=None):
        self.n_actions = tabular_dim[0]
        self.n_states = np.prod(tabular_dim)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros(tabular_dim)
        self.map_state = map_state # Maps state to tabular entries

    def choose_action(self, state, decay_factor=1):
        if self.map_state: state=self.map_state(state)
        if np.random.rand() < self.epsilon*decay_factor:
            # Choose a random action with probability epsilon
            action = np.random.randint(0, self.n_actions)
        else:            
            # Choose the action with the highest estimated action-value with probability 1-epsilon
            i = state[0]  # row index
            j = state[1]  # column index
            
            # Find the maximum value in the matrix
            q_values=self.q[:, i, j] 
            max_value = np.max(q_values)

            # Find indices of all occurrences of the maximum value in the matrix
            indices = np.argwhere(q_values == max_value)
            # Choose random action. This is to prevent bias when multiple actions have the same q value
            action = np.random.choice(np.ravel(indices))
        return action
    
    def update(self, episode_states, episode_actions, episode_rewards):
        raise NotImplementedError("Agent requires update function.")
