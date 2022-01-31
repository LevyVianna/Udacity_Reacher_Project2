import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size_1, hidden_size_2, output_gate=None):
        super(FCNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)
        self.output_gate = output_gate
        self.reset_parameters()

    def reset_parameters(self):
        u_range = 1. / np.sqrt(self.linear1.weight.data.size()[0])
        self.linear1.weight.data.uniform_(-u_range, u_range)
        u_range = 1. / np.sqrt(self.linear2.weight.data.size()[0])
        self.linear2.weight.data.uniform_(-u_range, u_range)
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x

class Policy(nn.Module):

    def __init__(self, action_size, ActorBody, CriticBody):
        super(Policy, self).__init__()
        self.actor_body = ActorBody
        self.critic_body = CriticBody
        self.std = torch.Tensor(nn.Parameter(torch.ones(1, action_size)))

    def forward(self, states, actions=None):
        values = self.critic_body(states)
        dist = torch.distributions.Normal(self.actor_body(states), self.std)
        if actions is None:
            dim = 1
            actions = dist.sample()
        else:
            dim = 2
        log_prob = torch.sum(dist.log_prob(actions), dim=dim, keepdim=True)
        entropy = torch.sum(dist.entropy(), dim=dim)[:, np.newaxis]
        return actions, log_prob, entropy, values
