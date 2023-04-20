import numpy as np
from tqdm import tqdm #progress bar
# local libraries
import sys
sys.path.extend(['..'])

from agents.TDLambdaAgent import TDLambdaAgent

def temporal_difference_lambda(env, episodes, tabular_dim, alpha=0.1, epsilon=0.1, gamma=1.0, lambd=0.5, map_state=None):
    n_actions = tabular_dim[0]
    agent = TDLambdaAgent(tabular_dim, alpha, epsilon, gamma, lambd, map_state)

    for episode in tqdm(range(episodes), desc="Progress", leave=False):
        state = env.start()
        action = agent.choose_action(state)

        # GLIE
        decay_factor = 1 - (episode / episodes)

        while not env.terminate:
            next_state, reward, terminate = env.step(action)
            next_action = agent.choose_action(next_state, decay_factor=decay_factor)

            agent.update(state, action, reward, next_state)

            state = next_state
            action = next_action

        # Decay eligibility traces at end of episode
        agent.e *= agent.gamma * agent.lambd

    return agent.q
