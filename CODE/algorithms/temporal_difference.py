import numpy as np
from tqdm import tqdm #progress bar
# local libraries
import sys
sys.path.extend(['..'])

from agents.TemporalDifferenceAgent import TDAgent

#Sarsa algorithm i
def temporal_difference(env, episodes, tabular_dim, alpha=0.1, epsilon=0.1, gamma=1.0, map_state=None):
    n_actions = tabular_dim[0]
    agent = TDAgent(tabular_dim, alpha, epsilon, gamma, map_state)

    for episode in tqdm(range(episodes), desc="Progress", leave=False):
        state = env.start()
        action = agent.choose_action(state)

        # GLIE
        decay_factor=1-(episode/episodes) # MAYBE DEFINE AS A FUNCTION INPUT

        while not env.terminate:
            next_state, reward, terminate = env.step(action)
            next_action = agent.choose_action(next_state, decay_factor=decay_factor)

            agent.update(state, action, reward, next_state)

            state = next_state
            action = next_action

    return agent.q