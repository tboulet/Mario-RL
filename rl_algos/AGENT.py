from abc import ABC, abstractmethod
import torch
import wandb
from random import randint
from METRICS import *
from div.utils import pr_and_raise, pr_shape

class AGENT(ABC):
    
    def __init__(self, config = dict(), metrics = list()):
        self.step = 0
        self.episode = 0
        self.metrics = [Metric(self) for Metric in metrics]
        self.config = config
        for name, value in config.items():
            setattr(self, name, value)
        self.metrics_saved = list()
        
    @abstractmethod
    def act(self, obs):
        pass
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, **kwargs):
        pass
    
    def add_metric(self, mode, **values):
        if mode == 'act':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_act(**values))
        if mode == 'remember':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_remember(**values))
        if mode == 'learn':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_learn(**values))
    
    def log_metrics(self):
        for metric in self.metrics_saved:
            wandb.log(metric, step = self.step)
        self.metrics_saved = list()
        
    def compute_critic(self, method, rewards, dones = None, observations = None, next_observations = None):
        '''Estimate some critic values such as advantage function A_pi_st_at of each transitions (s, a, r)t, noted A_s,  on an episode.
        method : string, method used for advantage estimation, in (TD, MC, n_step, GAE) or (total_reward)
        *args : elements for computing A_s, torch tensor.
        return : A_s as a torch tensor of shape (T, 1)
        '''
        if method == 'V_MC':
            rewards = rewards[:, 0] #(T,)
            values = list()
            t = len(rewards) - 1
            next_reward = 0
            while t >= 0:
                next_reward = rewards[t] + self.gamma * next_reward
                values.insert(0, next_reward)
                t -= 1
            res = torch.Tensor(values).unsqueeze(-1)           
                    
        elif method == 'A_MC':
            rewards = rewards[:, 0] #(T,)
            advantages = list()
            t = len(rewards) - 1
            next_reward = 0
            while t >= 0:
                next_reward = rewards[t] + self.gamma * next_reward
                advantages.insert(0, next_reward)
                t -= 1
            res = torch.Tensor(advantages).unsqueeze(-1) - self.state_value(observations)            
        
        elif method == 'V_TD':
            res = rewards + (1 - dones) * self.gamma * self.state_value(next_observations)

        elif method == 'A_TD':
            res = rewards + (1 - dones) * self.gamma * self.state_value(next_observations) - self.state_value(observations)
        

        
        else:
            raise Exception(f"Method '{method}' for computing advantage estimate is not implemented.")

        return res

#Use the following agent as a model for minimum restrictions on AGENT subclasses :
class RANDOM_AGENT(AGENT):
    '''A random agent evolving in a discrete environment.
    n_actions : int, n of action space
    '''
    def __init__(self, n_actions):
        super().__init__(metrics=[MetricS_On_Learn_Numerical, Metric_Performances]) #Choose metrics here
        self.n_actions = n_actions  #For RandomAgent only
    
    def act(self, obs):
        #Choose action here
        ...
        action = randint(0, self.n_actions - 1)
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'act', **values)
        
        return action
    
    def learn(self):
        #Learn here
        ...
        #Save metrics
        self.step += 1
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'learn', **values)
    
    def remember(self, *args):
        #Save kwargs in memory here
        ... 
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'remember', **values)