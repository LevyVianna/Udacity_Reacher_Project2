import numpy as np
import random
import copy
import os
import yaml
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from utils import Batcher

NUM_AGENTS = 20
SEED = 314
LR = 1.0e-4
SGD_EPOCH = 4
DISCOUNT = 1
GAE_LAMBDA = 0.95
BATCH_SIZE = 64

class Agent(object):

    def __init__(self, state_size, action_size):
        self.seed = SEED
        self.action_size = action_size
        self.__name__ = 'PPO'
        
        actor = FCNetwork(state_size, action_size, 500, 250, F.tanh)
        critic = FCNetwork(state_size, 1, 500, 250)  
        
        self.policy = Policy(action_size, actor, critic)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def step(self, t):
        states = torch.Tensor(t['states'])
        rewards = torch.Tensor(t['rewards'])
        old_probs = torch.Tensor(t['probs'])
        actions = torch.Tensor(t['actions'])
        old_values = torch.Tensor(t['values'])
        dones = torch.Tensor(t['dones'])
        
        for i in range(SGD_EPOCH):
            self.learn(states, rewards, old_probs, actions, old_values, dones)

    def act(self, states):
       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        states = torch.from_numpy(states).float().to(device)
        self.policy.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        return actions, log_probs, values

    def learn(self, states, rewards, old_probs, actions, old_values, dones):

        num_steps = len(states)
        rollout_length = [None for _ in range(num_steps)]
        advantages = torch.Tensor(np.zeros((NUM_AGENTS, 1)))
        for i in reversed(range(num_steps)):
            
            s = {
                'reward'     : torch.Tensor(rewards[i]).unsqueeze(1),
                'done'       : torch.Tensor(1 - dones[i]).unsqueeze(-1),
                'value'      : torch.Tensor(old_values[i]),
                'next_value' : old_values[min(num_steps - 1, i + 1)]
            }
            td_error = s['reward'] + DISCOUNT * s['done'] * s['next_value'].detach() -s['value'].detach()
            advantages = advantages * GAE_LAMBDA * DISCOUNT * s['done'] + td_error
            rollout_length[i] = advantages
            
        advantages = torch.stack(rollout_length).squeeze(2)
        advantages = ((advantages - advantages.mean()) / advantages.std())[:, :, np.newaxis]
        #advantages[:, :, np.newaxis]
        
        batcher = Batcher(BATCH_SIZE, [np.arange(states.size(0))])
        batcher.shuffle()
        while not batcher.end():
            indices = torch.Tensor(batcher.next_batch()[0]).long()
            L = clipped_surrogate(self.policy,
                                          old_probs[indices],
                                          states[indices],
                                          actions[indices],
                                          rewards[indices],
                                          advantages[indices])
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L

def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      advantages, epsilon=0.1, beta=0.0025):

        discount = DISCOUNT**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        actions = torch.tensor(actions, dtype=torch.float, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        _, new_probs, entropy, values = policy(states, actions)
        ratio = (new_probs - old_probs).exp()

        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrog = torch.min(ratio*advantages, clip*advantages)

        return  -torch.mean(clipped_surrog + beta*entropy)
