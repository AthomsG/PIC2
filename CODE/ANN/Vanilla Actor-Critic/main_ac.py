from tqdm import tqdm
from agent import Agent
from networks import ActorNetwork, CriticNetwork
from gridworld import GridWorld
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F

#discount factor for future utilities
DISCOUNT_FACTOR=0.99

#number of episodes to run
NUM_EPISODES=1000

#max steps per episode
MAX_STEPS=100

#learning rate
lr=1e-5

#grid size
GRIDSIZE=5

#track scores
SCORES = []

env=GridWorld(size=GRIDSIZE)
agent=Agent(n_actions=4, env=env, gamma=DISCOUNT_FACTOR, h_size=10, alpha=1e-5, beta=1e-5)

#run episodes
for episode in tqdm(range(NUM_EPISODES)):
    SCORES.append(agent.learn(max_len=MAX_STEPS))
    if episode%10==0:
        print(agent.choose_action(np.array([GRIDSIZE-1, GRIDSIZE-1])))

#plt.plot(SCORES)
#plt.show()