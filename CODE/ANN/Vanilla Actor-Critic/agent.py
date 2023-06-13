import os
import numpy as np

import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical

from networks import ActorNetwork, CriticNetwork

class Agent():
    '''
    input_dim  - Dimention of state
    n_actions  - Number of actions performed in the environment
    max_action - Upper bound of continuous action
    h_size     - Size of hidden layer
    alpha      - Adam optimizer learninng rate
    '''
    def __init__(self, n_actions,  alpha=3e-4, beta=3e-4, input_dims=2,
            env=None, gamma=0.99, h_size=256):
        # store init values internally
        self.gamma = gamma
        self.n_actions = n_actions
        self.env=env

        # instanciate function approximators
        self.actor = ActorNetwork(observation_space=input_dims, action_space=n_actions,
                    h_size=256, lr=3e-4, name='actor', checkpoint_dir='tmp/ac')
        self.critic = CriticNetwork(observation_space=input_dims,
                    h_size=256, lr=3e-4, name='critic', checkpoint_dir='tmp/ac')
        
    def set_env(self, env):
        self.env=env

    # sample action from policy π
    def choose_action(self, observation):
        #convert state to float tensor, add 1 dimension, allocate tensor on self.actor.device
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        
        #use network to predict action probabilities
        action_probs = self.actor(state)
        state = state.detach()
        
        #sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()
        
        #return action
        return action.item(), m.log_prob(action)

    # saves model in \dir
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    # loads model from \dir
    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    # learns throughout one episode or until reaching 'max_len' move
    def learn(self, max_len=1000, gamma=0.99):
        done  = False
        score = 0
        state = self.env.reset()

        I = 1
        i = 0
        while (not done and i < max_len):
            # compute action and log probability
            action, lp = self.choose_action(state)
            
            #step with action
            new_state, reward, done = self.env.step(action)
            
            #update episode score
            score += reward
            
            #get state value of current state
            state_tensor = T.from_numpy(state).float().unsqueeze(0).to(self.actor.device)
            state_val    = self.critic(state_tensor)
            
            #get state value of next state
            new_state_tensor = T.from_numpy(new_state).float().unsqueeze(0).to(self.actor.device)        
            new_state_val = self.critic(new_state_tensor)
            
            #if terminal state, next state val is 0
            if done:
                new_state_val = T.tensor([0]).float().unsqueeze(0).to(self.actor.device)
            
            #calculate value function loss with MSE
            val_loss = F.mse_loss(reward + gamma * new_state_val, state_val)
            val_loss *= I
            
            #calculate policy loss
            advantage = reward + gamma * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I
            
            self.actor.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.actor.optimizer.step()
            
            self.critic.optimizer.zero_grad()
            val_loss.backward(retain_graph=True)
            self.critic.optimizer.step()

            #move into new state, discount I
            state = new_state
            I *= gamma
            i+=1
        
        return score