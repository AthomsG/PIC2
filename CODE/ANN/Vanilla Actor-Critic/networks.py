import torch.nn as nn
import torch.nn.functional as F
import torch as T
import torch.optim as optim
import os

# SETUP FOR DISCRETE ACTION SPACE ENVIRONMENTS

# policy function approximator -> returns categorical distribution probabilities
class ActorNetwork(nn.Module):

    def __init__(self, observation_space, action_space, h_size=256, lr=3e-4,
                 name='actor', checkpoint_dir='tmp/ac'):
        super(ActorNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, h_size)
        self.output_layer = nn.Linear(h_size, action_space)
        self.checkpoint_file = os.path.join(checkpoint_dir, name+'_ac')
        
        # initialize the output layer weights to zero -> uniform random policy
        nn.init.ones_(self.output_layer.weight)
        nn.init.ones_(self.output_layer.bias)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # forward pass
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        actions = self.output_layer(x)
        # assume softmax policy
        action_probs = F.softmax(actions, dim=0)
        return action_probs
    
    # save models weights after training
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    # load model weights
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# state value function approximator -> returns scalar
class CriticNetwork(nn.Module):
    
    def __init__(self, observation_space, h_size=256, lr=3e-4,
                 name='critic', checkpoint_dir='tmp/ac'):
        super(CriticNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, h_size)
        self.output_layer = nn.Linear(h_size, 1)
        self.checkpoint_file = os.path.join(checkpoint_dir, name+'_ac')

        # initialize the output layer weights to zero -> uniform random policy
        nn.init.ones_(self.output_layer.weight)
        nn.init.ones_(self.output_layer.bias)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
    # forward pass
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        state_value = self.output_layer(x)
        
        return state_value
    
    # save models weights after training
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    # load model weights
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))