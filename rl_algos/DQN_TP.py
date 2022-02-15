from copy import copy, deepcopy
from operator import index
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from MEMORY import Memory
from CONFIGS import DQN_CONFIG

class DQN():
    '''
    DQN to fill for didactic purposes.
    '''

    def __init__(self, memory : Memory, action_value : nn.Module, metrics = [], config = DQN_CONFIG):
        self.memory = 
        self.step = 

        self.action_value = 
        self.action_value_target = 
        self.opt = 

        self.gamma =
        self.sample_size = 
        self.reward_scaler = #(mean, std), R <- (R-mean)/std
        self.update_method =    # "soft" or "periodic"
        
        self.exploration_timesteps = 
        self.exploration_initial = 
        self.exploration_final = 
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))
        self.metrics = list(Metric(self) for Metric in metrics)

    def act(self, observation, greedy=False, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Skip frames:
        # optional /
        
        #Batching observation : numpy of shape (n_obs,) to torch tensor of shape (1, n_obs)
        ...
        observations = 
    
        # Q(s) (shape = (1, n_action))
        Q_a = 

        #Choose action greedy
        epsilon = 
        if greedy or np.random.rand() > epsilon:
            ...
            action =
    
        #Exploration
        else :
            ...
            action = 
    
        # Action (int) is returned
        return action


    def learn(self):
        '''Do one step of learning.
        return : metrics, a list of metrics computed during this learning step.
        '''
        metrics = list()
        
        #Skip frames:
        # optional /

        #Learn only every train_freq steps
        # optional /

        #Learn only after learning_starts steps 
        # optional /

        #Sample trajectories from memory
        observations, actions, rewards, dones, next_observations = 
        ...
        actions = actions.to(dtype = torch.int64)
        #print(observations, actions, rewards, dones, sep = '\n\n')
    

        #Normalizing the rewards : (mean, std) = (100, 200) -> (0, 1)
        # optional /
        
        # Estimated Q values : 
        # method is Bootstrapping : Q(s,a) = r + gamma * max_a'(Q_target(s_next, a')) * (1-d)  | s_next and r being the result of action a taken in observation s
        ...
        Q_s_predicted = 
                   
        
        #Gradient descent on Q network
        ...
        
        #Update target network
        ...

        #Metrics
        return list(metric.on_learn(critic_loss = loss.detach().numpy(), value = Q_s.mean().detach().numpy()) for metric in self.metrics)

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        ...
        #Metrics
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)

        



if __name__ == "__main__":
    print('ratio')