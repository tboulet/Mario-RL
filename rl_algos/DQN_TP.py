from copy import copy, deepcopy
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
from METRICS import *
from rl_algos.AGENT import AGENT

class DQN_TP(AGENT):
    '''DQN to fill for didactic purposes
    '''
    
    def __init__(self, action_value : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Reward, Metric_Total_Reward, Metric_Performances, Metric_Action_Frequencies]
        super().__init__(config = DQN_CONFIG, metrics = metrics)
        self.memory = ...
        
        self.action_value = ...
        self.action_value_target = ...
        self.opt = ...
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))
        
        
    def act(self, observation, greedy=False, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Batching observation
        observations = ...
    
        # Q(s)
        Q = ...

        #Greedy policy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            ...
    
        #Exploration
        else :
            ...
        
        #Save metrics
        self.add_metric(mode = 'act')
    
        # Action
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1

        #Learn only every train_freq steps
        #\optional

        #Learn only after learning_starts steps 
        #\optional

        #Sample trajectories
        observations, actions, rewards, dones, next_observations = ...
        actions = actions.to(dtype = torch.int64)

        #Scaling the rewards
        #\optional
        
        # Estimated Q values
        Q_s_predicted = ...

        #Gradient descent on Q network
        criterion = nn.SmoothL1Loss()
        for _ in range(self.gradients_steps):
            ...
        
        #Update target network
        if self.update_method == "periodic":
            ...
        elif self.update_method == "soft":
            ...
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Save metrics*
        values["critic_loss"] = loss.detach().numpy()
        values["value"] = Q_s.mean().detach().numpy()
        self.add_metric(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        ...
        
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done, "next_obs" : next_observation}
        self.add_metric(mode = 'remember', **values)
    