import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.extend(['.', '..'])

from envs.GridWorld import plot_policy
from envs.FourRooms import FourRoomEnv
from algorithms.temporal_difference import temporal_difference
from algorithms.monte_carlo import monte_carlo

grid_size=11

env=FourRoomEnv(grid_size)
env.plot()

epsilon =0.5
episodes=50000
gamma   =1.0

#GridWorld
tabular_dim = (4, grid_size, grid_size)  # Action Set Cardinality, Dealer's card (1-10), Player's sum (1-21);

q_values = monte_carlo(env=env, episodes=episodes, tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma)
# The resulting q_values is a 2D array of shape (n_states, n_actions), where q_values[state][action] represents the estimated action-value for the given state and action.

v_values = np.mean(q_values, axis=0)
plt.imshow(v_values, cmap='gray')
plt.show()

plot_policy(q_values)