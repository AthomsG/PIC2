import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#
#                       AUXILIARY FUNCTIONS
#

def get_matrix(env_draw):
    matrix=list()
    values=dict({'':1, 'A':0, 'G':0.9})
    for i in reversed(range(len(env_draw))):
        row=list()
        for j in range(len(env_draw[i])):
            row.append(values[env_draw[i][j]])
        matrix.append(row)
    return matrix

def plot_env(env_draw, episode=None):
    matrix = get_matrix(env_draw)
    
    # Set the figure size
    plt.figure(figsize=(6, 6))

    # Display the matrix as an image with gray colormap
    plt.imshow(matrix, cmap='gray', origin='lower', extent=[0, len(matrix[0]), 0, len(matrix)])

    # Add grid lines at 0, 1, 2, 3...
    plt.xticks(range(len(matrix) + 1))
    plt.yticks(range(len(matrix[0]) + 1))
    plt.grid(color='black', linewidth=1)
    
    # Remove ticks
    plt.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    if(episode):
        plt.savefig(str(episode)+'.png')
    # Show the plot
    plt.show()


def plot_policy(q_values):
    stochastic_policy=False
    n_actions, n_rows, n_cols = q_values.shape
    fig, ax = plt.subplots()

    # plot squares for each state
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j-0.5, n_rows-i-1-0.5), 1, 1, fill=False, linewidth=1))

    # plot arrows for each state
    for i in range(n_rows):
        for j in range(n_cols):

            if i == n_rows-1 and j == n_cols-1: break # We don't care about the policy in the terminal state.

            state_q_values=q_values[:, i, j]
            max_value = np.max(state_q_values)

            # Find indices of all occurrences of the maximum value in the matrix
            indices = np.argwhere(state_q_values == max_value)
            # Choose random action. This is to prevent bias when multiple actions have the same q value
            if len(indices)>1: stochastic_policy=True
            for action in indices:
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
    if stochastic_policy:
        ax.set_title('Stochastic Policy')
    else:
        ax.set_title('Deterministic Policy')
    
    plt.show()

#
#                         GRIDWORLD CLASS
#

class GridWorld:
    def __init__(self, grid_size):
        self.agent_pos = np.array([0, 0])  # agent's initial position
        self.terminate = False
        if type(grid_size)==int:
            self.grid_size = np.array([grid_size, grid_size])
            self.goal_pos = np.array([grid_size - 1, grid_size - 1])  # goal position
        else:
            self.grid_size = np.array(grid_size)
            self.goal_pos = np.array([grid_size[0] - 1, grid_size[1] - 1])  # goal position
        

    def start(self):
        self.agent_pos = np.array([0, 0])  # reset agent's position
        self.terminate = False
        return tuple(self.agent_pos)

    def step(self, action):
        if action == 0:  # "up" action
            next_pos = self.agent_pos + np.array([-1, 0])
        elif action == 1:  # "down" action
            next_pos = self.agent_pos + np.array([1, 0])
        elif action == 2:  # "left" action
            next_pos = self.agent_pos + np.array([0, -1])
        elif action == 3:  # "right" action
            next_pos = self.agent_pos + np.array([0, 1])
        else:
            raise ValueError("Invalid action: {}. Must be 0 (up), 1 (down), 2 (left), or 3 (right).".format(action))

        # Check if the next position is within the grid boundaries ----------> HAS TO BE UPDATED
        if (next_pos >= 0).all() and (next_pos < self.grid_size).all():
            self.agent_pos = next_pos

        if (self.agent_pos == self.goal_pos).all():
            self.terminate = True
            return tuple(self.agent_pos), 0, self.terminate  # Goal reached, episode terminates with +1 reward
        else:
            return tuple(self.agent_pos), -1, self.terminate  # Episode continues with -0.1 reward

    def draw(self):
        # Draw the gridworld environment with the agent's current position and the goal position
        grid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=str)
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # agent's position
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'  # goal position

        return grid