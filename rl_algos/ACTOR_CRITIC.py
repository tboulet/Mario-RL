from copy import copy, deepcopy
from operator import index
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
from CONFIGS import ACTOR_CRITIC_CONFIG
from METRICS import *
from rl_algos.AGENT import AGENT

class ACTOR_CRITIC(AGENT):

    def __init__(self, 
                actor : nn.Module, 
                action_value : nn.Module = None, 
                state_value : nn.Module = None, 
                advantage_value : nn.Module = None, 
                ):
        '''A general Actor Critic algorithm, using a policy network that learns to act and a critic network that learns the model. 
        The parameter compute_gain_method defines the method for defining the weight of the policy gradients which can be (or not be) offline, centered, causal and using a critic.
        
        memory : a Memory object
        actor : a nn.Module neural network, pi_theta : s --> [p(a|s) for a]
        action_value : a nn.Module neural network, Q_omega : s --> [Q(s,a) for a]
        state_value : a nn.Module neural network, V_phi  : s --> V(s)
        metrics : a list of Metrics objects
        **config : a config dictionnary for Actor_Critic
        '''
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Count_Episodes]
        super().__init__(config = ACTOR_CRITIC_CONFIG, metrics = metrics)
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation'])
        self.step = 0
                
        self.setup_critic(action_value, state_value, advantage_value)
        self.policy = actor
        self.opt_policy = optim.Adam(lr = self.learning_rate_actor, params=self.policy.parameters())
        
        
    
    def setup_critic(self, Q, V, A):
        '''Method for preparing the ACTOR_CRITIC object to use and train its critic depending of the method used
        for computing gain.
        '''
        if self.compute_gain_method in ("total_reward", "total_future_reward", "total_reward_minus_leaky_mean", "total_reward_minus_MC_mean"):
            self.use_Q, self.use_V, self.use_A = False, False, False
            if self.compute_gain_method == "total_reward_minus_leaky_mean":
                self.alpha_0 = self.config["alpha_0"] if "alpha_0" in self.config else 1e-2
                self.V_0 = 0
        elif self.compute_gain_method in ("state_value", "state_value_centered", "total_future_reward_minus_state_value", "GAE"):
            if V is None:
                raise Exception(f"Using method {self.compute_gain_method} requires to use state value V.")
            self.use_Q, self.use_V, self.use_A = False, True, False
            self.state_value = V
            self.state_value_target = deepcopy(V)
            self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params = self.state_value.parameters())
        elif self.compute_gain_method in ("action_value", "action_value_centered", "total_future_reward_minus_action_value"):
            if Q is None:
                raise Exception(f"Using method {self.compute_gain_method} requires to use action value Q.")
            self.use_Q, self.use_V, self.use_A = True, False, False
            self.action_value = Q
            self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.action_value.parameters())
        elif self.compute_gain_method == "advantage_value":
            if A is None:
                raise Exception(f"Using method {self.compute_gain_method} requires to use advantage value A.")
            self.use_Q, self.use_V, self.use_A = False, False, True
            self.advantage_value = A
            self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.action_value.parameters())
            print("NOT IMPLEMENTED")
            raise
        else:
            raise NotImplementedError(f"Method {self.compute_gain_method} is not implemented.")
        
        
        
    
    
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''
        
        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
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
        self.step += 1
        values = dict()
        
        #Sample trajectories
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            method = "all",
            )
        actions = actions.to(dtype = torch.int64)
        
        #Learn only at end of episode
        if not dones[-1]:
            return
        
        #Scaling the rewards
        if self.reward_scaler is not None:          
            rewards /= self.reward_scaler
        
        #Updating the policy 
        if self.step % self.batch_size == 0:
            #Loss = - sum_t(G_t * ln(pi(a_t|s_t)))
            #G_t can be estimated by various methods
            with torch.no_grad():
                G = self.compute_gain(observations, actions, rewards, dones, next_observations, method = self.compute_gain_method)
            for _ in range(self.gradient_steps_policy):
                self.opt_policy.zero_grad()     
                probs = self.policy(observations)   #(T, n_actions)
                probs = torch.gather(probs, dim = 1, index = actions)   #(T, 1)
                log_probs = torch.log(probs)[:,0]     #(T,)
                loss_pi = torch.multiply(log_probs, G)
                loss_pi = - torch.sum(loss_pi)
                #Backpropagate to improve policy
                loss_pi.backward(retain_graph = True)
                self.opt_policy.step()
            #Empty memory of previous episode
            self.memory.__empty__()
            values["actor_loss"] = loss_pi.detach().numpy().mean()
        
        #Updating the action value
        if self.use_Q:
            #Bootsrapping : Q(s,a) = r + gamma * max_a'(Q(s_next, a')) * (1-d)
            criterion = nn.MSELoss()
            with torch.no_grad():
                Q_s_a_next = self.action_value(next_observations)
                Q_s_next, bests_a = torch.max(Q_s_a_next, dim = 1, keepdim=True)
                Q_s_estimated = rewards + (1-dones) * self.gamma * Q_s_next
            for _ in range(self.gradient_steps_critic):
                self.opt_critic.zero_grad()
                Q_s_a = self.action_value(observations)
                Q_s = Q_s_a.gather(dim = 1, index = actions)
                loss_Q = criterion(Q_s_estimated, Q_s)
                loss_Q.backward(retain_graph = True)
                if self.clipping is not None:
                    for param in self.action_value.parameters():
                        param.grad.data.clamp_(-self.clipping, self.clipping)
                self.opt_critic.step()
            values["critic_loss"] = loss_Q.detach().numpy()
            values["value"] = Q_s.detach().numpy().mean()
            
        #Updating the state value
        if self.use_V:
            #Bootstrapping : V(s) = r + gamma * V(s_next) * (1-d) or MC estimation
            criterion = nn.MSELoss()
            with torch.no_grad():
                V_s_estimated = rewards + (1-dones) * self.gamma * self.state_value_target(next_observations)
                V_s_estimated = V_s_estimated.to(torch.float32)  
            for _ in range(self.gradient_steps_critic):
                self.opt_critic.zero_grad()
                V_s = self.state_value(observations)
                loss_V = criterion(V_s, V_s_estimated)         
                loss_V.backward()
                if self.clipping is not None:
                    for param in self.state_value.parameters():
                        param.grad.data.clamp_(-self.clipping, self.clipping)
                self.opt_critic.step()
            values["critic_loss"] = loss_V.detach().numpy()
            values["value"] = V_s.detach().numpy().mean()
        
        #Updating the advantage value
        if self.use_A and self.step == 0:
            #to implement if possible
            raise
            
        #Updating V_0
        if self.compute_gain_method == "total_reward_minus_leaky_mean":
            ep_lenght = rewards.shape[0]
            weigths_gamma = torch.Tensor([self.gamma ** t for t in range(ep_lenght)])
            rewards_weighted = torch.multiply(rewards, weigths_gamma)
            total_reward = torch.sum(rewards_weighted)
            self.V_0 += self.alpha_0 * (total_reward - self.V_0)
            
        #Metrics
        self.add_metric(mode = 'learn', **values)






    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        self.memory.remember((observation, action, reward, done, next_observation))
        
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done, "next_obs" : next_observation}
        self.add_metric(mode = 'remember', **values)
        
        
        
        
    def compute_gain(self, observations, actions, rewards, dones, next_observations, method):
        '''Compute the "gain" or the "advantage function" that will be applied as weights to the gradients of ln(pi).
        *args : the elements of a trajectory during one episode
        method : the method used for computing the gain
        return : a tensor of shape the lenght of the previous episode, containing the gain/advantage at each step t.
        '''
        ep_lenght = rewards.shape[0]        
        if method == "total_reward":
            weigths_gamma = torch.Tensor([self.gamma ** t for t in range(ep_lenght)])
            rewards_weighted = torch.multiply(rewards, weigths_gamma)
            total_reward = torch.sum(rewards_weighted)
            G = total_reward.repeat(repeats = (ep_lenght,))
        elif method == "total_future_reward":           
            G = self.compute_future_total_rewards(rewards)
        elif method == "total_reward_minus_MC_mean":
            total_reward_MC_mean = None #to implement
            G = self.compute_future_total_rewards(rewards) - total_reward_MC_mean
        elif method == "total_reward_minus_leaky_mean":
            G = self.compute_future_total_rewards(rewards) - self.V_0
        elif method == "total_future_reward_minus_state_value":
            G = self.compute_future_total_rewards(rewards) - self.state_value(observations)
        elif method == "state_value":
            G = self.state_value(observations)[:,0]
        elif method == "state_value_centered":
            V_s = self.state_value(observations)
            G = rewards + (1-dones) * self.gamma * self.state_value(next_observations) - V_s
            G = G[:, 0]
        elif method == "GAE":
            delta = (rewards + self.state_value(next_observations) - self.state_value(observations)).detach().numpy()[:, 0]
            A_GAE = [None for _ in range(ep_lenght - 1)] + [delta[-1]]
            for u in range(1, ep_lenght):
                t = ep_lenght - 1 - u
                A_GAE[t] = self.gamma * self.lmbda * A_GAE[t+1] + delta[t]     
            G = torch.tensor(A_GAE, dtype=torch.float)
                       
        elif method == "action_value":
            Q_s_a = self.action_value(observations)
            G = torch.gather(Q_s_a, dim = 1, index=actions)[:, 0]
        elif method == "action_value_centered":
            Q_s_a = self.action_value(observations)
            Q_s = torch.gather(Q_s_a, dim = 1, index=actions)[:, 0]
            PI_s_a = self.policy(observations)
            Q_s_a_weighted = Q_s_a * PI_s_a
            Q_s_mean = torch.sum(Q_s_a_weighted, dim = 1)
            G = Q_s - Q_s_mean
        elif method == "total_future_reward_minus_action_value":
            total_future_rewards = self.compute_future_total_rewards(rewards)
            Q_s_a = self.action_value(observations)
            PI_s_a = self.policy(observations)
            Q_s_a_weighted = Q_s_a * PI_s_a
            Q_s_mean = torch.sum(Q_s_a_weighted, dim = 1)
            G = total_future_rewards - Q_s_mean
        
            
        else:
            raise NotImplementedError("Method for computing gain is not recognized.")   
         
        return G.detach()
    

    def compute_future_total_rewards(self, rewards):
        '''Compute [G_t for t] where G_t is the sum of future reward weighted by discount factor.
        rewards : a tensor of shape (duration_last_episode, 1)
        return : a tensor of same shape, [G_t for t] where G_t is the sum for t' >= t of r_t * gamma^(t' - t)
        '''
        ep_lenght = rewards.shape[0]
        weigths_gamma = torch.Tensor([self.gamma ** t for t in range(ep_lenght)])
        rewards_weighted = torch.multiply(rewards, weigths_gamma)
        future_total_rewards = list(torch.sum(rewards_weighted[t:]) for t in range(ep_lenght))
        return torch.Tensor(future_total_rewards)


