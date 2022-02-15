from random import randint

class AGENT:
    def __init__(self, config = dict()):
        self.step = 0
        for name, value in config.items():
            setattr(self, name, value)
    def act(self, obs):
        pass
    def learn(self):
        self.step += 1
        return list()
    def remember(self, *kwargs):
        return list()


class RandomAgent(AGENT):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
    def act(self, obs):
        return randint(0, self.n_actions - 1)