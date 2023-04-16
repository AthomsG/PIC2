from Easy21 import Easy21
import numpy as np

'''
tabular_dim: shape of state transition 'tensor' for the MDP.
             See David Silver Lecture on Markov Decision Processes
'''

# Implementation of First-Visit Monte Carlo Method Agent
class MonteCarloAgent:
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0):
        self.n_actions = tabular_dim[0]
        self.n_states = np.prod(tabular_dim)
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros(tabular_dim)
        self.n_visits = np.zeros(tabular_dim)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Choose a random action with probability epsilon
            action = np.random.randint(0, self.n_actions)
        else:
            # This might introduce bias, since I'm not sure if argmax chooses at random if all values are equal
            
            # Choose the action with the highest estimated action-value with probability 1-epsilon
            
            i = state[0]-1  # row index
            j = state[1]-1  # column index
            
            matrix_with_argmax = self.q[:, i, j]  # Extract along axis 0 (first axis)
            argmax_matrix_idx = np.argmax(matrix_with_argmax) # ----> ACTION index
            
            action = argmax_matrix_idx
        return action

    def update(self, episode_states, episode_actions, episode_rewards): # Control
        G = 0 # Sample return
        for t in reversed(range(len(episode_states))): #reverse to compute returns
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            G = self.gamma * G + reward

            if state not in episode_states[:t]:
                # First visit to the state in the episode
                self.n_visits[action, state[0]-1, state[1]-1] += 1 # -1 to deal with indexes
                alpha = 1 / self.n_visits[action, state[0]-1, state[1]-1]
                # alpha as a constant is used to 'forget' earlier action-states values computed with worse policies. 
                self.q[action, state[0]-1, state[1]-1] += alpha * (G - self.q[action, state[0]-1, state[1]-1])