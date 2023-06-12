import sys
sys.path.extend(['.', '..'])

from envs.Easy21 import Easy21
from algorithms.monte_carlo import monte_carlo

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def map_state(state):
    return [i-1 for i in state]

# PARAMETERS
epsilon =0.1
episodes=100000
gamma   =1.0 #Uniscounted as all episodes contain terminal state

# action-state value function
env=Easy21()
q_values = monte_carlo(env=env, 
                       episodes=episodes, 
                       epsilon=epsilon, 
                       gamma=gamma, 
                       tabular_dim=(2, 10, 21), 
                       map_state=map_state)

# state value function
v_values = np.maximum(q_values[0], q_values[1])

# state value function plot
m, n = v_values.shape

x = np.arange(1, n+1)
y = np.arange(1, m+1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, v_values, cmap='plasma')

ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_zlabel(r'V$_{\pi}$(s)')
ax.set_title("Monte Carlo {} Episodes".format(episodes))

#plt.savefig('easy21_MC_value.pdf')
plt.show()

# policy plot

policy = (q_values[1] > q_values[0]).astype(int)

ax = plt.gca()

# Major ticks
ax.set_xticks(np.arange(0, 21, 1))
ax.set_yticks(np.arange(0, 10, 1))

# Labels for major ticks
ax.set_xticklabels(np.arange(1, 22, 1))
ax.set_yticklabels(np.arange(1, 11, 1))

# Minor ticks
ax.set_xticks(np.arange(-.5, 21, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='salmon', linestyle='-', linewidth=2)

# Remove minor ticks
ax.tick_params(which='minor', bottom=False, left=False)

# Labels
ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_title('Policy')

plt.imshow(policy, aspect='equal', cmap='gray')
plt.show()
#plt.savefig('easy_21_MC_policy.pdf')