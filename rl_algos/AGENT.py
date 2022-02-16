from abc import ABC, abstractmethod
from random import randint


class AGENT(ABC):
    
    def __init__(self, config = dict(), metrics = list()):
        self.step = 0
        self.metrics = [Metric(self) for Metric in metrics]
        self.config = config
        for name, value in config.items():
            setattr(self, name, value)
        self.metrics_saved = list()
        
    @abstractmethod
    def act(self, action, values):
        return action
    
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
        

class RANDOM_AGENT(AGENT):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        super().__init__()
    
    def act(self, obs):
        return randint(0, self.n_actions - 1)
    
    def learn(self):
        return dict()
    
    def remember(self, **kwargs):
        return dict()