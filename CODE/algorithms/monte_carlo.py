import numpy as np
from tqdm import tqdm #progress bar
# local libraries
import sys
sys.path.extend(['..'])

from agents.MonteCarloAgent import MonteCarloAgent

def monte_carlo(env, episodes, tabular_dim, epsilon=0.1, gamma=1.0, map_state=None):
    n_actions = tabular_dim[0]
    agent = MonteCarloAgent(tabular_dim, epsilon, gamma, map_state)

    for episode in tqdm(range(episodes), desc="Progress", leave=False):
        episode_states = []
        episode_actions = []
        episode_rewards = []

        state = env.start()
        
        # GLIE
        decay_factor=1-(episode/episodes) # MAYBE DEFINE AS A FUNCTION INPUT

        while not env.terminate:
            action = agent.choose_action(state, decay_factor=decay_factor)
            episode_states.append(state)
            episode_actions.append(action)

            next_state, reward, terminate = env.step(action)
            episode_rewards.append(reward)

            if terminate:
                agent.update(episode_states, episode_actions, episode_rewards)
                
            state = next_state

    return agent.q