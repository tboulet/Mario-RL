import sys
import math
import cProfile
import random
import matplotlib.pyplot as plt
import numpy as np
#Torch for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary
#Gym for environments, WandB for feedback
import gym
from gym.wrappers import Monitor
import wandb

#Utils
from div.utils import *
from env_mario import load_smb_env
from METRICS import *
#RL agents
from rl_algos.DQN import DQN
from rl_algos.REINFORCE import REINFORCE
from rl_algos.AGENT import RANDOM_AGENT


def run(agent, env, steps, wandb_cb = True, 
        n_render = 20
        ):
    '''Train an agent on an env.
    agent : an AGENT instance (with methods act, learn and remember implemented)
    env : a gym env (with methods reset, step, render)
    episodes : int, number of episodes of training
    wandb_cb : bool, whether metrics are logged in WandB
    n_render : int, one episode on n_render is rendered
    '''
    
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
    episode = 1
    step = 0
    while step < steps:
        done = False
        obs = env.reset()
        
        
        while not done and step < steps:
            action = agent.act(obs)                                                 #Agent acts
            next_obs, reward, done, info = env.step(action)                         #Env reacts            
            agent.remember(obs, action, reward, done, next_obs, info)    #Agent saves previous transition in its memory
            agent.learn()                                                #Agent learn (eventually)
            
            ###### Feedback ######
            print(f"Episode n°{episode} - Total step n°{step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            if wandb_cb:
                agent.log_metrics()
            ######  End Feedback ######  

            #If episode ended, reset env, else change state
            if done:
                step += 1
                episode += 1
                break
            else:
                step += 1
                obs = next_obs
    
    if wandb_cb: run.finish()   #End wandb run.
    print("End of run.")
    
    
    

if __name__ == "__main__":
    #ENV
    n_side = 84
    n_stack = 4
    env = load_smb_env(obs_complexity=1, n_side = n_side, n_stack = n_stack)
    
    
    n_actions = env.action_space.n

    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Action_Frequencies, MetricS_On_Learn]
    
    #ACTOR PI
    actor =  nn.Sequential(
            nn.Conv2d(in_channels=n_stack, out_channels=32, kernel_size=8, stride=4),   #4,84,84 to 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax(),
        )
    
    #CRITIC Q
    action_value =  nn.Sequential(
            nn.Conv2d(in_channels=n_stack, out_channels=32, kernel_size=8, stride=4),   #4,84,84 to 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    #summary(action_value, (n_stack, n_side, n_side))

    #AGENT
    dqn = DQN(action_value=action_value)
    reinforce = REINFORCE(actor=actor)
    random_agent = RANDOM_AGENT(2)
    agent = reinforce
    
    #RUN
    run(agent, 
        env = env, 
        steps=500000, 
        wandb_cb = True,
        n_render=5,
        )    
    
    







