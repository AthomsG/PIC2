import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

# Network architectures were chosen to be as in the original paper. See Appendix D. Hyperparameters

# Critic Networks -> State Action Value function Approximator
class CriticNetwork(nn.Module):
    '''
    input_dim  - Dimention of state
    n_actions  - Number of actions performed in the environment
    max_action - Upper bound of continuous action
    h_size     - Size of hidden layer
    alpha      - Adam optimizer learninng rate
    '''
    def __init__(self, beta, input_dims, n_actions, hidden_layer_1_dims=256, hidden_layer_2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_1_dims = hidden_layer_1_dims
        self.hidden_layer_2_dims = hidden_layer_2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.hidden_layer_1 = nn.Linear(self.input_dims[0]+n_actions, self.hidden_layer_1_dims)
        self.hidden_layer_2 = nn.Linear(self.hidden_layer_1_dims, self.hidden_layer_2_dims)
        self.q = nn.Linear(self.hidden_layer_2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    # ANN forward pass
    def forward(self, state, action):
        action_value = self.hidden_layer_1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.hidden_layer_2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    # save models weights after training
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    # load model weights
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# Value Network -> State Value Function Approximator. Could be computed through critic
# but having a separate state value function helps with convergence (akin to ensemble methods)
class ValueNetwork(nn.Module):
    '''
    input_dim  - Dimention of state
    n_actions  - Number of actions performed in the environment
    max_action - Upper bound of continuous action
    h_size     - Size of hidden layer
    alpha      - Adam optimizer learninng rate
    '''
    def __init__(self, beta, input_dims, hidden_layer_1_dims=256, hidden_layer_2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_1_dims = hidden_layer_1_dims
        self.hidden_layer_2_dims = hidden_layer_2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.hidden_layer_1 = nn.Linear(*self.input_dims, self.hidden_layer_1_dims)
        self.hidden_layer_2 = nn.Linear(self.hidden_layer_1_dims, hidden_layer_2_dims)
        self.v = nn.Linear(self.hidden_layer_2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.hidden_layer_1(state)
        state_value = F.relu(state_value)
        state_value = self.hidden_layer_2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
    # save models weights after training
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    # load model weights
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    '''
    input_dim  - Dimention of state
    n_actions  - Number of actions performed in the environment
    max_action - Upper bound of continuous action
    h_size     - Size of hidden layer
    alpha      - Adam optimizer learninng rate
    '''
    def __init__(self, alpha, input_dims, max_action, hidden_layer_1_dims=256, 
            hidden_layer_2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_1_dims = hidden_layer_1_dims
        self.hidden_layer_2_dims = hidden_layer_2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        # squashing function used in SAC to enforce action bounds -> Appendix C.
        self.max_action = max_action
        # for numerical stability (on evaluating log(0))
        self.reparam_noise = 1e-6

        self.hidden_layer_1 = nn.Linear(*self.input_dims, self.hidden_layer_1_dims)
        self.hidden_layer_2 = nn.Linear(self.hidden_layer_1_dims, self.hidden_layer_2_dims)
        # Gaussian Parameters -> Family of Distributions chosen in the original paper
        self.mu = nn.Linear(self.hidden_layer_2_dims, self.n_actions)
        self.sigma = nn.Linear(self.hidden_layer_2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # forward pass
    def forward(self, state):
        prob = self.hidden_layer_1(state)
        prob = F.relu(prob)
        prob = self.hidden_layer_2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
    # sample action from policy
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        # VER ISTO COM A ZITA! NÃO PERCEBI AINDA
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # enforcing action bounds -> Appendix C.
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

