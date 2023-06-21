import numpy as np
from agents.TabularAgent import TabularAgent

# Implementation of Vanilla Off-Policy QLearning 
class QLearningAgent(TabularAgent):
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0, alpha=0.1, map_state=None):
        super().__init__(tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma, alpha=alpha, map_state=map_state)

    def update(self, state, action, reward, next_state, done):
        if self.map_state:
            state = self.map_state(state)
            next_state = self.map_state(next_state)

        max_next_value = np.max(self.q[:, next_state[0], next_state[1]])
        # Compute TD target
        td_target = reward + self.gamma * max_next_value * (1 - done)  # Include done flag in TD target
        # Compute TD error
        td_error = td_target - self.q[action, state[0], state[1]]
        # Update action-value estimates
        self.q[action, state[0], state[1]] += self.alpha * td_error