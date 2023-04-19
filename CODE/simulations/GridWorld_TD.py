import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import sys
sys.path.extend(['.', '..'])

from envs.GridWorld import GridWorld
from algorithms.temporal_difference import temporal_difference

def plot_policy(q_values):
    n_actions, n_rows, n_cols = q_values.shape
    fig, ax = plt.subplots()

    # plot squares for each state
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j-0.5, n_rows-i-1-0.5), 1, 1, fill=False, linewidth=1))

    # plot arrows for each state
    for i in range(n_rows):
        for j in range(n_cols):
            action = np.argmax(q_values[:, i, j])
            if action == 0:  # up
                ax.arrow(j, n_rows-i-1, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif action == 1:  # down
                ax.arrow(j, n_rows-i-1, 0, -0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif action == 2:  # left
                ax.arrow(j, n_rows-i-1, -0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif action == 3:  # right
                ax.arrow(j, n_rows-i-1, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                
    # set axis labels and limits
    ax.set_xlim([-0.5, n_cols-0.5])
    ax.set_ylim([-0.5, n_rows-0.5])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    plt.show()


epsilon =1
episodes=100000
gamma   =1

grid_size=5

#GridWorld
tabular_dim = (4, grid_size, grid_size)  # Action Set Cardinality, Dealer's card (1-10), Player's sum (1-21);

env = GridWorld([tabular_dim[1], tabular_dim[2]])

q_values = temporal_difference(env=env, episodes=episodes, tabular_dim=tabular_dim, epsilon=epsilon, gamma=gamma)
# The resulting q_values is a 2D array of shape (n_states, n_actions), where q_values[state][action] represents the estimated action-value for the given state and action.

v_values = np.mean(q_values, axis=0)
plt.title(r'V$_{\pi}$ for optimal policy')
plt.imshow(v_values, cmap='gray')
plt.show()

plot_policy(q_values)

# Save value to .txt
with open('v_values.txt', 'w') as f:
    np.savetxt(f, np.round(v_values, 4), fmt='%.4f')