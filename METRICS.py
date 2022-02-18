from numbers import Number
from time import time


class Metric():
    def __init__(self):
        pass

    def on_learn(self, **kwargs):
        return dict()
    
    def on_remember(self, **kwargs):
        return dict()
    
    def on_act(self, **kwargs):
        return dict()
    

class MetricS_On_Learn(Metric):
    '''Log every metrics whose name match classical RL important values such as Q_value, actor_loss ...'''
    metric_names = ["value", "Q_value", "V_value", "actor_loss", "critic_loss", ]
    def __init__(self, agent):
        super().__init__()
        self.agent = agent  
    
    def on_learn(self, **kwargs):
        return {metric_name : kwargs[metric_name] for metric_name in self.metric_names if metric_name in kwargs}


class MetricS_On_Learn_Numerical(Metric):
    '''Log every numerical metrics.'''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent  
    
    def on_learn(self, **kwargs):
        return {metric_name : kwargs[metric_name] for metric_name, value in kwargs.items() if isinstance(value, Number)}


class Metric_Total_Reward(Metric):
    '''Log total reward (sum of reward over an episode) at every episode.'''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent    
        self.total_reward = 0
        self.new_episode = False

    def on_remember(self, **kwargs):
        try:
            if self.new_episode: 
                self.total_reward = 0
                self.new_episode = False
            self.total_reward += kwargs["reward"]

            if kwargs["done"]:
                self.new_episode = True
                return {"total_reward" : self.total_reward}
            else:
                return dict()
        except KeyError:
            return dict()


class Metric_Reward(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
    def on_remember(self, **kwargs):
        try:
            return {"reward" : kwargs["reward"]}
        except:
            return dict()
        

class Metric_Epsilon(Metric):
    '''Log exploration factor.'''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def on_learn(self, **kwargs):
        try:
            return {"epsilon" : self.agent.f_eps(self.agent)}
        except:
            return dict()


class Metric_Critic_Value_Unnormalized(Metric):
    '''Log value not scaled.'''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.is_normalized = not( hasattr(agent, "reward_scaler") or agent.reward_scaler is None )
        
    def on_learn(self, **kwargs):
        try:
            if self.is_normalized:
                return {"value_unnormalized" : self.agent.reward_scaler * kwargs["value"]}
            else:
                return {"value_unnormalized" : kwargs["value"]}
        except KeyError:
            return dict()


class Metric_Action_Frequencies(Metric):
    '''Log action frequency in one episode for each action possible.'''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.frequencies = dict()
        self.new_episode = False

    def on_remember(self, **kwargs):
        try:
            if self.new_episode: 
                self.frequencies = dict()
                self.ep_lenght = 0
                self.new_episode = False
            action = kwargs["action"]
            if action not in self.frequencies:
                self.frequencies[action] = 0
            self.frequencies[action] += 1

            if kwargs["done"]:
                self.new_episode = True
                ep_lenght = sum(self.frequencies.values())
                return {f"action_{a}_freq" : n_actions / ep_lenght for a, n_actions in self.frequencies.items()}
            else:
                return dict()
        except KeyError:
            return dict()
        

class Metric_Count_Episodes(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.n_episodes = 0
        
    def on_remember(self, **kwargs):
        try:
            if kwargs["done"]:
                self.n_episodes += 1
                return {"n_episodes" : self.n_episodes}
            else:
                return dict()
        except KeyError:
            return dict()
        
        
class Metric_Time_Count(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.t0 = time()
    
    def on_learn(self, **kwargs):
        return {"time" : round((time() - self.t0) / 60, 2)}
    
        
class Metric_Performances(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.t0 = time()
    def on_x(self, step_of_training : str):
        dur = time() - self.t0
        self.t0 = time()
        if self.agent.step < 10:
            return dict()
        return {step_of_training: dur}
    def on_act(self, **kwargs):
        return self.on_x("time : ACTING + LOGGING (+ RENDERING)")
    def on_remember(self, **kwargs):
        return self.on_x("time : ENV REACTING + REMEMBERING")
    def on_learn(self, **kwargs):
        return self.on_x("time : SAMPLING + LEARNING")
    
    
