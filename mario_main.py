import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym.wrappers import Monitor
import wandb

from div.render import render_agent
from div.run import run 
from div.utils import *

from MEMORY import Memory
from METRICS import *

from rl_algos.DQN import DQN
from rl_algos.REINFORCE import REINFORCE

class ObservationMarioWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        obs = np.swapaxes(obs, 0, 2)
        obs = np.array(obs)
        return obs



def run(agent, env, episodes, wandb_cb = True, 
        n_render = 20
        ):
    
    print("Run starts.")
################### FEEDBACK #####################
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        config=agent.config,
        )
##################### END FEEDBACK ###################

        
    for episode in range(1, episodes+1):
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()
            
            ###### Feedback ######
            print(f"Episode n°{episode} - Total step n°{agent.step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
            ######  End Feedback ######  
            
            #If episode ended.
            if done:
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()
    print("End of run.")
    
    
    

if __name__ == "__main__":
    #ENV
    env = load_smb_env(obs_complexity=1, action_complexity=1)
    env = ObservationMarioWrapper(env)

    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Action_Frequencies]
    
    #ACTOR PI
    n_channels = 3
    n_actions = env.action_space.n
    actor = nn.Sequential(
        nn.Conv2d(n_channels, 8, 3),
        nn.Tanh(),
        nn.Conv2d(8, 8, 3),
        nn.Flatten(),
        nn.Linear(475776, n_actions),
        nn.Softmax(),
    )
    
    #CRITIC Q/V
    action_value = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, out_features=env.action_space.n),
    )
    n_actions = env.action_space.n
    height, width, n_channels = env.observation_space.shape
    
    action_value = nn.Sequential(
        nn.Conv2d(n_channels, 8, 3),
        nn.Tanh(),
        nn.Conv2d(8, 8, 3),
        nn.Flatten(),
        nn.Linear(475776, n_actions),
    )

    #AGENT
    dqn = DQN(action_value=action_value, metrics = metrics)
    reinforce = REINFORCE(actor=actor, metrics=metrics)

    #RUN
    run(dqn, env, episodes=1000, wandb_cb = False)    







