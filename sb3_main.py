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
from stable_baselines3 import PPO
import wandb

from div.render import render_agent
from div.run import run_for_sb3 
from div.utils import *

from MEMORY import Memory
from METRICS import *

from rl_algos_torch.DQN import DQN


if __name__ == "__main__":
    #ENV
    env = gym.make("Acrobot-v1")
    
    #AGENT
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    }
    create_agent = lambda env: PPO(config["policy_type"], env, verbose=1)
    
    #RUN
    agent = run_for_sb3(create_agent, config, env, episodes=1000, wandb_cb = True, video_cb = True)
    render_agent(agent, env)







