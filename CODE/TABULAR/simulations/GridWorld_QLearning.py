import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.extend(['.', '..'])

from envs.GridWorld import GridWorld, plot_policy
from algorithms.q_learning import q_learning

epsilon = 0.5
episodes = 300
gamma = 0.99
alpha = 0.01

grid_size = (15, 15)

# GridWorld
tabular_dim = (4, grid_size[0], grid_size[1])  # Action Set Cardinality, Dealer's card (1-10), Player's sum (1-21);

env = GridWorld([tabular_dim[1], tabular_dim[2]])

q_values = q_learning(env=env, episodes=episodes, tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma, alpha=alpha)
# The resulting q_values is a 3D array of shape (n_actions, grid_size[0], grid_size[1]),
# where q_values[action][row][column] represents the estimated action-value for the given action, row, and column.

v_values = np.max(q_values, axis=0)
plt.title(r'V$_{\pi}$ for optimal policy')
plt.imshow(v_values, cmap='gray')
plt.show()

plot_policy(q_values)