from Easy21 import Easy21
import numpy as np

'''
tabular_dim: shape of state transition 'tensor' for the MDP.
             See David Silver Lecture on Markov Decision Processes
'''

# Implementation of First-Visit Monte Carlo Method Agent
class MonteCarloAgent:
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0, map_state=None):
        self.n_actions = tabular_dim[0]
        self.n_states = np.prod(tabular_dim)
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros(tabular_dim)
        self.n_visits = np.zeros(tabular_dim)
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
            # Choose random action
            action = np.random.choice(np.ravel(indices))
        return action

    def update(self, episode_states, episode_actions, episode_rewards): # Control
        G = 0 # Sample return
        for t in reversed(range(len(episode_states))): #reverse to compute returns
            state = episode_states[t]
            if self.map_state: state=self.map_state(state)
            action = episode_actions[t]
            reward = episode_rewards[t]
            G = self.gamma * G + reward

            if state not in episode_states[:t]:
                # First visit to the state in the episode
                self.n_visits[action, state[0], state[1]] += 1 # -1 to deal with indexes
                alpha = 1 / self.n_visits[action, state[0], state[1]]
                # alpha as a constant is used to 'forget' earlier action-states values computed with worse policies. 
                self.q[action, state[0], state[1]] += alpha * (G - self.q[action, state[0], state[1]])