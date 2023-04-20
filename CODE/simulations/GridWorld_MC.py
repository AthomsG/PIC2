import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.extend(['.', '..'])

from envs.GridWorld import GridWorld, plot_policy
from algorithms.monte_carlo import monte_carlo

epsilon =1
episodes=100
gamma   =0.9

grid_size=10

#GridWorld
tabular_dim = (4, grid_size, grid_size)  # Action Set Cardinality, Dealer's card (1-10), Player's sum (1-21);

env = GridWorld([tabular_dim[1], tabular_dim[2]])

q_values = monte_carlo(env=env, episodes=episodes, tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma)
# The resulting q_values is a 2D array of shape (n_states, n_actions), where q_values[state][action] represents the estimated action-value for the given state and action.

v_values = np.mean(q_values, axis=0)
plt.imshow(v_values, cmap='gray')
plt.show()

plot_policy(q_values)