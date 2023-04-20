import numpy as np
from agents.TabularAgent import TabularAgent

class TDLambdaAgent(TabularAgent):
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0, alpha=0.1, lambd=0.5, map_state=None):
        super().__init__(tabular_dim, epsilon, gamma, alpha, map_state)
        self.lambd = lambd
        self.e = np.zeros(self.q.shape) # Eligibility traces

    def update(self, state, action, reward, next_state): # Control
        if self.map_state:
            state = self.map_state(state)
            next_state = self.map_state(next_state)
        # Compute TD target
        td_target = reward + self.gamma * np.max(self.q[:, next_state[0], next_state[1]])
        # Compute TD error
        td_error = td_target - self.q[action, state[0], state[1]]
        # Update eligibility traces - accumulating
        self.e *= self.gamma * self.lambd
        self.e[action, state[0], state[1]] += 1 # see fig 7.9 Sutton Barto for the three kinds of traces (accumulating, dutch and replacing)
        # Update action-value estimates
        self.q += self.alpha * td_error * self.e
        # Decay eligibility traces
        self.e *= self.gamma * self.lambd
