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
from torch.distributions.categorical import Categorical

from MEMORY import Memory
from CONFIGS import PPO_CONFIG
from METRICS import *
from rl_algos.AGENT import AGENT

class PPO(AGENT):
    '''PPO updates its networks without changing too much the policy, which increases stability.
    NN trained : Actor Critic
    Policy used : On-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous (discrete not implemented)
    '''

    def __init__(self, actor : nn.Module, state_value : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Reward, Metric_Total_Reward, Metric_Performances]
        super().__init__(config = PPO_CONFIG, metrics = metrics)
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation'])
        self.last_action = None
        
        self.state_value = state_value
        self.state_value_target = deepcopy(state_value)
        self.opt_critic = optim.Adam(lr = 1e-4, params=self.state_value.parameters())
        
        self.policy = actor
                
        
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped numpy observation.
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
        self.last_action = action
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1
        criterion = nn.MSELoss()
        
        #We learn once we got enought transitions
        if self.step % self.timesteps != 0:
            return
        
        #Sample trajectories
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            method = "all_shuffled",
            )
        actions = actions.to(dtype = torch.int64)
        rewards = rewards.float()
        # print(observations.shape, actions, rewards, dones, sep = '\n\n')
        # raise

        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards = rewards / self.reward_scaler
        
        #We perform gradient descent on K epochs on T datas with minibatch of size M <= T.
        policy_new = deepcopy(self.policy)
        opt_policy = optim.Adam(lr = 1e-4, params=policy_new.parameters())
        
        #Compute probability of old policy
        pi_theta_old_s_a = self.policy(observations)  
        pi_theta_old_s   = torch.gather(pi_theta_old_s_a, dim = 1, index = actions)
        
        n_batch = self.timesteps // self.batch_size
        for _ in range(self.epochs):
            for i in range(n_batch):
                #Batching data
                observations_batch = observations[i * self.batch_size : (i+1) * self.batch_size]
                actions_batch = actions[i * self.batch_size : (i+1) * self.batch_size]
                rewards_batch = rewards[i * self.batch_size : (i+1) * self.batch_size]
                dones_batch = dones[i * self.batch_size : (i+1) * self.batch_size]
                next_observations_batch = next_observations[i * self.batch_size : (i+1) * self.batch_size]
                pi_theta_old_s_batch = pi_theta_old_s[i * self.batch_size : (i+1) * self.batch_size]
                
                #Advantage function A, using a V value and will be usefull for later
                with torch.no_grad():
                    V_s_target = rewards_batch + (1 - dones_batch) * self.gamma * self.state_value_target(next_observations_batch)
                    A_s = V_s_target - self.state_value(observations_batch)
                    
                #Objective function : J_clip = min(r*A, clip(r,1-e,1+e)A)  where r = pi_theta_new/pi_theta_old and A advantage function
                pi_theta_new_s_a = policy_new(observations_batch)
                pi_theta_new_s   = torch.gather(pi_theta_new_s_a, dim = 1, index = actions_batch)
                ratio_s = pi_theta_new_s / pi_theta_old_s_batch
                ratio_s_clipped = torch.clamp(ratio_s, 1 - self.epsilon_clipper, 1 + self.epsilon_clipper)
                J_clip = torch.minimum(ratio_s * A_s, ratio_s_clipped * A_s).mean()

                #Error on critic : L = L(V(s), V_target)   with V_target = r + gamma * (1-d) * V_target(s_next)
                V_s = self.state_value(next_observations_batch)
                critic_loss = criterion(V_s, V_s_target).mean()
                
                #Entropy : H = sum_a(- log(p) * p)      where p = pi_theta(a|s)
                log_pi_theta_s_a = torch.log(pi_theta_new_s_a)
                pmlogp_s_a = - log_pi_theta_s_a * pi_theta_new_s_a
                H = torch.mean(pmlogp_s_a, dim = 1).mean()

                #Total objective function
                J = J_clip - self.c_critic * critic_loss + self.c_entropy * H
                loss = - J
                
                #Gradient descend
                opt_policy.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward(retain_graph = True)
                opt_policy.step()
                self.opt_critic.step()
        
        #Update policy
        self.policy = deepcopy(policy_new)
        self.memory.__empty__()
        
        #Update target network
        if self.update_method == "periodic":
            if self.step % self.target_update_interval == 0:
                self.state_value_target = deepcopy(self.state_value)
        elif self.update_method == "soft":
            for phi, phi_target in zip(self.state_value.parameters(), self.state_value_target.parameters()):
                phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data    
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Save metrics
        values["critic_loss"] = critic_loss.detach().numpy()
        values["actor_reward"] = J_clip.detach().numpy()
        values["value"] = V_s.mean().detach().numpy()
        self.add_metric(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        self.memory.remember((observation, action, reward, done, next_observation))
        
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done, "next_obs" : next_observation}
        self.add_metric(mode = 'remember', **values)
    