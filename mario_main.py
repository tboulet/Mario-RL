import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
#Torch for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#Gym for environments, WandB for feedback
import gym
from gym.wrappers import Monitor
import wandb

#Utils
from div.utils import *
from METRICS import *
#RL agents
from rl_algos.DQN import DQN
from rl_algos.REINFORCE import REINFORCE


#Wrapper for SMB env
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

        
    for episode in range(1, episodes+1):
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.act(obs)                                                 #Agent acts
            next_obs, reward, done, info = env.step(action)                         #Env reacts
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)    #Agent saves previous transition in its memory
            metrics2 = agent.learn()                                                #Agent learn (eventually)
            
            ###### Feedback ######
            print(f"Episode n°{episode} - Total step n°{agent.step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
            ######  End Feedback ######  
            
            #If episode ended, reset env, else change state
            if done:
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()   #End wandb run.
    print("End of run.")
    
    
    

if __name__ == "__main__":
    #ENV
    env = load_smb_env(obs_complexity=1, action_complexity=1)
    env = ObservationMarioWrapper(env)
    
    n_actions = env.action_space.n
    height, width, n_channels = env.observation_space.shape 
    
    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Action_Frequencies]
    
    #ACTOR PI
    actor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(height * width * n_channels, n_actions),
        nn.Softmax(),
    )
    
    #CRITIC Q
    action_value = nn.Sequential(
        nn.Flatten(),
        nn.Linear(height * width * n_channels, n_actions),
    )

    #AGENT
    dqn = DQN(action_value=action_value, metrics = metrics)
    reinforce = REINFORCE(actor=actor, metrics=metrics)

    #RUN
    run(dqn, 
        env, 
        episodes=1000, 
        wandb_cb = False,
        n_render=1,
        )    







