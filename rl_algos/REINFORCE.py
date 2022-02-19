from copy import copy, deepcopy
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt
from div.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

from MEMORY import Memory
from CONFIGS import REINFORCE_CONFIG
from METRICS import Metric_Performances, Metric_Total_Reward, MetricS_On_Learn
from rl_algos.AGENT import AGENT

class REINFORCE(AGENT):
    '''REINFORCE agent is an actor RL agent that performs gradient ascends on the estimated objective function to maximize.
    NN trained : Actor
    Policy used : On-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : discrete / continuous
    '''

    def __init__(self, actor : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Performances]
        super().__init__(config = REINFORCE_CONFIG, metrics = metrics)
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done'])
        
        self.policy = actor
        self.opt = optim.Adam(lr = 1e-4, params=self.policy.parameters())
                
        
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped nummpy observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''
        
        #Batching observation
        observation = torch.Tensor(observation)
        observations = observation.unsqueeze(0) # (1, observation_space)
        probs = self.policy(observations)        # (1, n_actions)
        distribs = Categorical(probs = probs)    
        actions = distribs.sample()
        action = actions.numpy()[0]
        
        #Save metrics
        self.add_metric(mode = 'act')
        
        # Action
        return action


    def learn(self):
        '''Do one step of learning.
        return : metrics, a list of metrics computed during this learning step.
        '''
        values = dict()
        self.step += 1
        
        #Sample trajectories
        observations, actions, rewards, dones = self.memory.sample(
            method = "all",
            )
        actions = actions.to(dtype = torch.int64)
        
        #Learn only at end of episode
        if not dones[-1]:
            return
            
        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards /= self.reward_scaler
        
        #Computing loss = - sum_t( G_t * sum_t'(log(pi(a_t' | S = s_t'))) )   | with G_t being the sum of future rewards at time t (with discount factor gamma) and sum_t' summing for t' >= t
        self.opt.zero_grad()
            #G_t = sum_[t'>=t](gamma ** (t'-t) * r_t')
        ep_lenght = rewards.shape[0]    #T
        rewards = rewards[:,0].numpy().tolist()
        G = [rewards[-1]] 
        for i in range(1, ep_lenght):
            t = ep_lenght - i
            previous_G_t =  rewards[t] + self.gamma * G[0]
            G.insert(0, previous_G_t)
        G = torch.tensor(G)
        
            #log_proba_t = sum_t'(log(pi(a|s)))
        probs = self.policy(observations)   #(T, n_actions)
        probs = torch.gather(probs, dim = 1, index = actions)   #(T, 1)
        log_probs = torch.log(probs)[:,0]     #(T,)
            #sum_t( G_t * log_proba_t )
        loss = torch.multiply(log_probs, G)
        loss = - torch.sum(loss)
        
        #Backpropagate to improve policy
        loss.backward()
        self.opt.step()
        self.memory.__empty__()
        
        #Save metrics
        values["actor_loss"] = loss.detach().numpy()
        self.add_metric(mode = 'learn', **values)



    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        self.memory.remember((observation, action, reward, done, next_observation, info))

        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.add_metric(mode = 'remember', **values)






class REINFORCE_OFFLINE():

    def __init__(self, memory, actor : nn.Module, metrics = [], config = REINFORCE_CONFIG):
        super().__init__(config)
        self.config = config
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation', 'probs'])
        self.step = 0
        self.last_action = None
        self.last_prob = 1.
        
        self.policy = actor
        self.opt = optim.Adam(lr = self.learning_rate, params=self.policy.parameters())
        self.metrics = list(Metric(self) for Metric in metrics)
        
        
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an actioné
        '''

        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
        probs = self.policy(observations)        # (1, n_actions)
        distribs = Categorical(probs = probs)    
        actions = distribs.sample()
        action = actions.numpy()[0]
        
        # Action
        self.last_action = action
        self.last_prob = probs[0][action]
        return action


    def learn(self):
        '''Do one step of learning.
        return : metrics, a list of metrics computed during this learning step.
        '''
        metrics = list()
        self.step += 1
        
        #Sample trajectories
        observations, actions, rewards, dones, next_observations, probs = self.memory.sample(
            method = "all",
            func = lambda arr : torch.Tensor(arr),
        )
        actions = actions.to(dtype = torch.int64)
        
        #Learn only at end of episode
        if not dones[-1]:
            return metrics
            
        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards /= self.reward_scaler
        
        #Computing loss = - sum_t( G_t * sum_t'(log(pi(a_t' | S = s_t'))) )   | with G_t being the sum of future rewards at time t (with discount factor gamma) and sum_t' summing for t' >= t
        self.opt.zero_grad()
            #G_t = sum_[t'>=t](gamma ** (t'-t) * r_t')
        ep_lenght = rewards.shape[0]    #T
        G = [rewards[-1].numpy()] 
        for i in range(1, ep_lenght):
            t = ep_lenght - i
            previous_G_t =  rewards[t] + self.gamma * G[0]
            G.insert(0, previous_G_t)
        G = torch.tensor(G)
            #log_proba_t = sum_t'(log(pi(a|s)))
        probs = self.policy(observations)   #(T, n_actions)
        probs = torch.gather(probs, dim = 1, index = actions)   #(T, 1)
        log_probs = torch.log(probs)[:,0]     #(T,)
            #sum_t( G_t * log_proba_t )
        loss = torch.multiply(log_probs, G)
        loss = - torch.sum(loss)
        
        #Backpropagate to improve policy
        loss.backward()
        self.opt.step()
        self.memory.__empty__()
        
        #Metrics
        values = {"actor_loss" : loss.detach().numpy()}
        self.add_metric(mode = 'learn', values = values)

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        prob = self.last_prob
        self.memory.remember((observation, action, reward, done, next_observation, info, prob))
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)

    