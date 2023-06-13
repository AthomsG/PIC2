import numpy as np

class GridWorld:
    def __init__(self, size=10):
        self.grid_size = (size, size)
        self.agent_pos = (0, 0)
        self.target_pos = (size-1, size-1)
        self.num_actions = 4  # Up, Down, Left, Right
        
    def get_state(self):
        return self.agent_pos
        
    def reset(self):
        self.agent_pos = (0, 0)
        return np.array(self.agent_pos)
    
    def step(self, action):
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Down
            self.agent_pos = (min(self.grid_size[0] - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # Right
            self.agent_pos = (self.agent_pos[0], min(self.grid_size[1] - 1, self.agent_pos[1] + 1))
        
        done = self.agent_pos == self.target_pos
        reward = 0 if done else -1
        return np.array(self.agent_pos), reward, done
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.agent_pos] = 1
        grid[self.target_pos] = 2
        print(grid)