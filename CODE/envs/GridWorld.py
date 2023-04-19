import numpy as np
import matplotlib.pyplot as plt

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