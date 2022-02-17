from abc import ABC, abstractmethod
import wandb
from random import randint
from METRICS import *

class AGENT(ABC):
    
    def __init__(self, config = dict(), metrics = list()):
        self.step = 0
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