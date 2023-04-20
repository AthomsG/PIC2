import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.extend(['.', '..'])

from envs.GridWorld import GridWorld, plot_policy
from algorithms.temporal_difference import temporal_difference

epsilon = 0.8
episodes= 100000
gamma   = 1.0
alpha   = 0.01

grid_size=(5, 5)

#GridWorld
tabular_dim = (4, grid_size[0], grid_size[1])  # Action Set Cardinality, Dealer's card (1-10), Player's sum (1-21);

env = GridWorld([tabular_dim[1], tabular_dim[2]])

q_values = temporal_difference(env=env, episodes=episodes, tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma, alpha=alpha)
# The resulting q_values is a 2D array of shape (n_states, n_actions), where q_values[state][action] represents the estimated action-value for the given state and action.

v_values = np.mean(q_values, axis=0)
plt.title(r'V$_{\pi}$ for optimal policy')
plt.imshow(v_values, cmap='gray')
plt.show()

plot_policy(q_values)

# Save value to .txt
with open('v_values.txt', 'w') as f:
    np.savetxt(f, np.round(v_values, 4), fmt='%.4f')