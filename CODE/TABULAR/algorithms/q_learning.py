import numpy as np
from tqdm import tqdm #progress bar
# local libraries
import sys
sys.path.extend(['..'])

from agents.TabularAgent import TabularAgent
from agents.QLearningAgent import QLearningAgent

def q_learning(env, episodes, tabular_dim, epsilon=0.1, gamma=1.0, alpha=0.1, map_state=None):
    n_actions = tabular_dim[0]
    agent = QLearningAgent(tabular_dim, epsilon, gamma, alpha, map_state)

    for episode in tqdm(range(episodes), desc="Progress", leave=False):
        state = env.start()
        decay_factor = 1 - (episode / episodes)

        while not env.terminate:
            action = agent.choose_action(state, decay_factor=decay_factor)
            next_state, reward, terminate = env.step(action)
            agent.update(state, action, reward, next_state, terminate)
            state = next_state

    return agent.q