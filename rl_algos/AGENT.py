from abc import ABC, abstractmethod
from random import randint


class AGENT(ABC):
    
    def __init__(self, config = dict()):
        self.step = 0
        for name, value in config.items():
            setattr(self, name, value)
    
    @abstractmethod
    def act(self, obs):
        pass
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, **kwargs):
        pass
    

class RANDOM_AGENT(AGENT):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        super().__init__()
    
    def act(self, obs):
        return randint(0, self.n_actions - 1)
    
    def learn():
        return dict()
    
    def remember(self, **kwargs):
        return dict()