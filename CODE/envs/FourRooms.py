import numpy as np
import matplotlib.pyplot as plt

class FourRoomEnv:
    def __init__(self, grid_size):
        self.agent_pos = np.array([0, 0])  # agent's initial position
        self.grid_size = np.array(grid_size)
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])  # goal position

        # Define walls for the four rooms
        room_size = self.grid_size // 2
        self.walls = np.zeros([self.grid_size, self.grid_size], dtype=bool)
        self.walls[room_size, :] = True
        self.walls[:, room_size] = True
        self.walls[room_size:room_size+1, :room_size] = True
        self.walls[room_size+1:room_size+1, room_size:] = True
        self.walls[room_size:, room_size+1:room_size+1] = True

        self.walls[room_size, (room_size)//2]     = False
        self.walls[room_size, 1+3*(room_size)//2] = False

        self.walls[(room_size)//2, room_size] = False
        self.walls[1+3*(room_size)//2, room_size] = False

        self.terminate = False

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

        # Check if the next position is within the grid boundaries and not a wall
        if (next_pos >= 0).all() and (next_pos < self.grid_size).all() and not self.walls[tuple(next_pos)]:
            self.agent_pos = next_pos

        if (self.agent_pos == self.goal_pos).all():
            self.terminate = True
            return tuple(self.agent_pos), 0, self.terminate  # Goal reached, episode terminates with +1 reward
        else:
            return tuple(self.agent_pos), -1, self.terminate  # Episode continues with -0.1 reward

    def plot(self):
        # Draw the gridworld environment with the agent's current position and the goal position
        grid = np.ones((self.grid_size, self.grid_size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 0.5  # agent's position
        grid[self.goal_pos[0], self.goal_pos[1]] = 0.9 # goal position
        grid[self.walls] = 0 # draw walls

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
        ax.grid(True, color='black', linewidth=1.5)

         # Add grid lines at 0, 1, 2, 3...
        plt.xticks(range(self.grid_size + 1))
        plt.yticks(range(self.grid_size + 1))
        plt.grid(color='black', linewidth=1)
    
        # Remove ticks
        plt.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.show()