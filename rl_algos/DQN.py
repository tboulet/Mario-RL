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

class DQN(AGENT):

    def __init__(self, action_value : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward]
        super().__init__(config = DQN_CONFIG, metrics = metrics)
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation'])
        
        self.action_value = action_value
        self.action_value_target = deepcopy(action_value)
        self.opt = optim.Adam(lr = self.learning_rate, params=action_value.parameters())
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))
        
        
    def act(self, observation, greedy=False, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
    
        # Q(s)
        Q = self.action_value(observations) # (1, action_space)

        #Greedy policy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            with torch.no_grad():
                if mask is not None:
                    Q = Q - 10000.0 * torch.Tensor([mask])      # So that forbidden action won't ever be selected by the argmax.
                action = torch.argmax(Q, axis = -1).numpy()[0]  
    
        #Exploration
        else :
            if mask is None:
                action = torch.randint(size = (1,), low = 0, high = Q.shape[-1]).numpy()[0]     #Choose random action
            else:
                authorized_actions = [i for i in range(len(mask)) if mask[i] == 0]              #Choose random action among authorized ones
                action = random.choice(authorized_actions)
        
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
        if self.step % self.train_freq != 0:
            return

        #Learn only after learning_starts steps 
        if self.step < self.learning_starts:
            return

        #Sample trajectories
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            sample_size=self.sample_size,
            method = "random",
            )
        actions = actions.to(dtype = torch.int64)
        
        # print(observations.shape, actions, rewards, dones, sep = '\n\n')
        # raise

        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards = rewards / self.reward_scaler
        
        # Estimated Q values
        if not self.double_q_learning:
            #Simple learning : Q(s,a) = r + gamma * max_a'(Q_target(s_next, a')) * (1-d)  | s_next and r being the result of action a taken in observation s
            future_Q_s_a = self.action_value_target(next_observations)
            future_Q_s, bests_a = torch.max(future_Q_s_a, dim = 1, keepdim=True)
            Q_s_predicted = rewards + self.gamma * future_Q_s * (1 - dones)  #(n_sampled,)
        else:
            #Double Q Learning : Q(s,a) = r + gamma * Q_target(s_next, argmax_a'(Q(s_next, a')))
            future_Q_s_a = self.action_value(next_observations)
            future_Q_s, bests_a = torch.max(future_Q_s_a, dim = 1, keepdim=True)
            future_Q_s_a_target = self.action_value_target(next_observations)
            future_Q_s_target = torch.gather(future_Q_s_a_target, dim = 1, index= bests_a)
            
            Q_s_predicted = rewards + self.gamma * future_Q_s_target * (1 - dones)
        
        #Gradient descent on Q network
        criterion = nn.SmoothL1Loss()
        for _ in range(self.gradients_steps):
            self.opt.zero_grad()
            Q_s_a = self.action_value(observations)
            Q_s = Q_s_a.gather(dim = 1, index = actions)
            loss = criterion(Q_s, Q_s_predicted)
            loss.backward(retain_graph = True)
            if self.clipping is not None:
                for param in self.action_value.parameters():
                    param.grad.data.clamp_(-self.clipping, self.clipping)
            self.opt.step()
        
        #Update target network
        if self.update_method == "periodic":
            if self.step % self.target_update_interval == 0:
                self.action_value_target = deepcopy(self.action_value)
        elif self.update_method == "soft":
            for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
                phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data    
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
        self.memory.remember((observation, action, reward, done, next_observation))
        
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done, "next_obs" : next_observation}
        self.add_metric(mode = 'remember', **values)
    
