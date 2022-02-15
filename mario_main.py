r"""
File for lauching implemented agents on Mario env.
"""

#Torch library for Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#Utils library
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
#Gym (for creating environments) and WandB (for feedback)
import gym
from gym.wrappers import Monitor
import wandb

#Import utils
from div.render import render_agent 
from div.utils import *
#Import metrics that will be used for extracting information during the training process, log them on WandB
from METRICS import *
#Import RL agents
from rl_algos.ALL_AGENTS import RandomAgent, DQN, REINFORCE, ACTOR_CRITIC



#Wrapper for observation of mario env
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
    '''Train an agent on an env.
    agent : AGENT object (with methods act, learn and remember)
    env : gym environment (with methods reset, step and render)
    episodes : number of episodes of training
    wandb_cb : boolean, True if log metrics on WandB
    n_render : int, one episode on n_render will be rendered
    '''
    
################### FEEDBACK #####################
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        config=agent.config
                        )
##################### END FEEDBACK ###################

    print("Run starts.")  
    for episode in range(1, episodes+1):
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.act(obs)                                                 #Agent acts.
            next_obs, reward, done, info = env.step(action)                         #Env reacts.
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)    #Agent saves last transition in its memory.
            metrics2 = agent.learn()                                                #Agent learns (or not depending of when .learn() performs a learning step...)
            
            ###### Feedback for WandB ######
            print(f"Episode n°{episode} - Total step n°{agent.step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
            ######  End Feedback ######  
            
            if done:        #Leave episode if done, else change the state.
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()   #Annouce WandB the run has ended.
    print("End of run.")




if __name__ == "__main__":
    #ENV
    env = load_smb_env(obs_complexity=1, action_complexity=1)
    env = ObservationMarioWrapper(env)
    
    #METRICS
    metrics = [Metric_Total_Reward, MetricS_On_Learn, Metric_Time_Count]
    
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
    random_agent = RandomAgent(3)
    
    #RUN
    run(reinforce, 
        env, 
        episodes=1000, 
        wandb_cb = True,
        n_render=1,
        )    







