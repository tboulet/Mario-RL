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

from div.utils import *
from MEMORY import Memory
from CONFIGS import PPO_CONFIG
from METRICS import *
from rl_algos.AGENT import AGENT

class PPO(AGENT):
    '''PPO updates its networks without changing too much the policy, which increases stability.
    NN trained : Actor Critic
    Policy used : Off-policy
    Online : Yes
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous (discrete not implemented)
    '''

    def __init__(self, actor : nn.Module, state_value : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Count_Episodes]
        super().__init__(config = PPO_CONFIG, metrics = metrics)
        self.memory_transition = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'])
        self.memory_episodes = Memory(MEMORY_KEYS = ['episode'])
        
        self.state_value = state_value
        self.state_value_target = deepcopy(state_value)
        self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.state_value.parameters())
        
        self.policy = actor
        
        self.last_prob = None
        self.episode_ended = False
                
        
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
        self.last_prob = probs[0, action].detach()
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1
        
        #Learn every n end of episodes
        if not self.episode_ended:
            return
        self.episode += 1
        self.episode_ended = False
        if self.episode % self.train_freq_episode != 0:
            return
        
        #Sample trajectories
        episodes = self.memory_episodes.sample(
            method = "last",
            sample_size=self.n_episodes,
            as_tensor=False,
            )
        episodes = episodes[0]
        
        #Compute A_s and V_s estimates and concatenate trajectories. 
        advantages = list()
        V_targets = list()
        for observations, actions, rewards, dones, probs in episodes:
            #Scaling the rewards
            if self.reward_scaler is not None:
                rewards = rewards / self.reward_scaler
            #Compute V and A
            advantages.append(self.compute_critic(self.compute_advantage_method, observations = observations, rewards = rewards, dones = dones))
            V_targets.append(self.compute_critic(self.compute_value_method, observations = observations, rewards = rewards, dones = dones))
        advantages = torch.concat(advantages, axis = 0).detach()
        V_targets = torch.concat(V_targets, axis = 0).detach()
        observations, actions, rewards, dones, probs = [torch.concat([episode[elem] for episode in episodes], axis = 0) for elem in range(len(episodes[0]))]
        
        #Shuffling data
        indexes = torch.randperm(len(rewards))
        observations, actions, rewards, dones, probs, advantages, V_targets = \
            [element[indexes] for element in [observations, 
                                            actions, 
                                            rewards, 
                                            dones, 
                                            probs, 
                                            advantages,
                                            V_targets,
                                            ]]
        
        #Type bug fixes
        actions = actions.to(dtype = torch.int64)
        rewards = rewards.float()
        
        #We perform gradient descent on K epochs on T datas with minibatch of size M <= T.
        policy_new = deepcopy(self.policy)
        opt_policy = optim.Adam(lr = self.learning_rate_actor, params=policy_new.parameters())           
        n_batch = math.ceil(len(observations) / self.batch_size)
    
        for _ in range(self.epochs):
            for i in range(n_batch):
                #Batching data
                observations_batch = observations[i * self.batch_size : (i+1) * self.batch_size]
                actions_batch = actions[i * self.batch_size : (i+1) * self.batch_size]
                probs_batch = probs[i * self.batch_size : (i+1) * self.batch_size]
                advantages_batch = advantages[i * self.batch_size : (i+1) * self.batch_size]
                V_targets_batch = V_targets[i * self.batch_size : (i+1) * self.batch_size]

                #Objective function : J_clip = min(r*A, clip(r,1-e,1+e)A)  where r = pi_theta_new/pi_theta_old and A advantage function
                pi_theta_new_s_a = policy_new(observations_batch)
                pi_theta_new_s   = torch.gather(pi_theta_new_s_a, dim = 1, index = actions_batch)
                ratio_s = pi_theta_new_s / probs_batch
                ratio_s_clipped = torch.clamp(ratio_s, 1 - self.epsilon_clipper, 1 + self.epsilon_clipper)
                J_clip = torch.minimum(ratio_s * advantages_batch, ratio_s_clipped * advantages_batch).mean()

                #Error on critic : L = L(V(s), V_target)   with V_target = r + gamma * (1-d) * V_target(s_next)
                V_s = self.state_value(observations_batch)
                critic_loss = F.smooth_l1_loss(V_s, V_targets_batch).mean()
                
                #Entropy : H = sum_a(- log(p) * p)      where p = pi_theta(a|s)
                log_pi_theta_s_a = torch.log(pi_theta_new_s_a)
                pmlogp_s_a = - log_pi_theta_s_a * pi_theta_new_s_a
                H_s = torch.sum(pmlogp_s_a, dim = 1)
                H = H_s.mean()
                            
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
        values["J_clip"] = J_clip.detach().numpy()
        values["value"] = V_s.mean().detach().numpy()
        values["entropy"] = H.mean().detach().numpy()
        self.add_metric(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        prob = self.last_prob.detach()
        self.memory_transition.remember((observation, action, reward, done, prob, info))
        if done:
            self.episode_ended = True
            episode = self.memory_transition.sample(method = 'all', as_tensor=True)
            self.memory_transition.__empty__()
            self.memory_episodes.remember((episode,))
            
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.add_metric(mode = 'remember', **values)