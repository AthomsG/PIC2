import numpy as np
from agents.TabularAgent import TabularAgent

# Implementation of First-Visit Monte Carlo Method Agent
class MonteCarloAgent(TabularAgent):
    def __init__(self, tabular_dim, epsilon=0.1, gamma=1.0, map_state=None):
        super().__init__(tabular_dim, epsilon, gamma, map_state)
        self.n_visits = np.zeros(tabular_dim)

    def update(self, episode_states, episode_actions, episode_rewards):
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
                self.alpha = 1 / self.n_visits[action, state[0], state[1]]
                # alpha as a constant is used to 'forget' earlier action-states values computed with worse policies. 
                self.q[action, state[0], state[1]] += self.alpha * (G - self.q[action, state[0], state[1]])
