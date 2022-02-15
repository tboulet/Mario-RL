from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import math
import random
import os
from moviepy.editor import *
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym.wrappers import Monitor as GymMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import wandb

def run(agent, env, episodes, wandb_cb = True, plt_cb = True, video_cb = True, 
        n_ep_save_video = 100, 
        n_render = 20
        ):
    
    print("Run starts.")
    config = agent.config
################### FEEDBACK #####################
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        config=config
                        )
    if video_cb:
        videos_path = "./div/videos/rl-video/"
        env = GymMonitor(env, videos_path, video_callable=lambda ep: ep % n_ep_save_video == 0, force = True)
    if plt_cb:
        logs = dict()
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
                if plt_cb:
                    for key, value in metric.items():
                        try:
                            logs[key]["steps"].append(agent.step)
                            logs[key]["values"].append(value)    
                        except KeyError:
                            logs[key] = {"steps": [agent.step], "values": [value]}      
                        plt.clf()         
                        plt.plot(logs[key]["steps"][-100:], logs[key]["values"][-100:], '-b')
                        plt.title(key)
                        plt.savefig(f"figures/{key}")
            ######  End Feedback ######  
            
            #If episode ended.
            if done:
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()
    print("End of run.")
    


def run_for_sb3(create_agent, config, env, episodes, wandb_cb = True, video_cb = True):
    print("Run for SB3 agent starts.")
    
    env1 = deepcopy(env)
    def make_env():
        env2 = deepcopy(env1)
        env2.reset()
        env2 = Monitor(env2) 
        return env2
    env = DummyVecEnv([make_env])
    
    #Wandb
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        sync_tensorboard=True,
                        monitor_gym=True,
                        config=config
                        )
    #Save videos of agent
    if video_cb:
        n_save = 5000
        video_path = f"div/videos/rl-videos-sb3/{run.id}"
        env = VecVideoRecorder(env, video_folder= video_path, record_video_trigger=lambda step: step % n_save == 0)

    agent = create_agent(env = env)
    agent.learn(total_timesteps=config["total_timesteps"], 
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"div/models/{run.id}",
                    verbose=2,
            ) if wandb_cb else None,
        )
            
    
    if wandb_cb: run.finish()
    print("End of run.")
    return agent