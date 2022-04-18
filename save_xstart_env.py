# Python library
from copy import copy, deepcopy
import pickle
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
# Gym for environments
from env_mario import *
import gym
# Sb3 for agents and callbacks
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#Other
import keyboard
import json

def load_smb_env_blind(obs_complexity=1, n_side=84, n_stack=1, n_skip=4):
    # torch (84, 84) --> torch (4, 84, 84)
    game_id = f"SuperMarioBros-1-1-v{3-obs_complexity}"
    env = gym_super_mario_bros.make(game_id)
    MOVEMENTS = [["right"], ["right", "A"]]
    env = JoypadSpace(env, MOVEMENTS)

    # Apply Wrappers to environment
    # Gray the observation, reducing channels without loosing information.
    env = GrayScaleObservation(env)
    # Resize obs shape to (84, 84) for dimension reduction while keeping sufficient infomation
    env = ResizeObservation(env, shape=n_side)
    # Stack last 4 frames to obtain a (4, 84, 84) array
    env = FrameStack(env, num_stack=n_stack)
    env = NumpyingObservation(env)  # Transform obs from LazyFrame to np array
    env = SkipFrame(env, n_skip)  # Skip some frames
    env = RedefineRewardInfo(env)  # Redefines reward
    # env = DisplayMarioPerspective(env)
    return env


def load_x_to_env():
    try:
        file = open('x_to_env.json', 'r')
        x_to_env = pickle.load(file)
        file.close()
    except FileNotFoundError:
        x_to_env = dict()
    return x_to_env

def save_x_to_env(x_to_env):
    pickle.dump(x_to_env, open( "x_to_env.json", "wb" ) )


def run(env : gym.Env):
    x_to_env = load_x_to_env()
    
    obs = env.reset()
    while True:
        if keyboard.is_pressed("space"):   #Jump
            action  = 1
        elif keyboard.is_pressed("s"):     #Save
            state_copy = deepcopy(obs)
            action = 0
            name_save = input("Save name:")
            # x_to_env[int(info["x_pos"].item())] = {"name" : name_save, "env" : obs, "state" : obs}
            x_to_env[info["x_pos"].item()] = {"name" : name_save, "env" : None, "state" : state_copy}
        elif keyboard.is_pressed("q"):     #Quit
            save_x_to_env(x_to_env)
            sys.exit()
        else:                               #Move forward
            action = 0
        
        print("Key:", keyboard.read_key())
        print("Action : ", action)
        
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(.01666)
        if dones:
            obs = env.reset()
            


if __name__ == "__main__":

    n_side = 84
    n_stack = 4
    env1 = load_smb_env_blind(
        obs_complexity=0, n_side=n_side, n_stack=n_stack, n_skip=4)
    n_actions = env1.action_space.n

    MOVEMENTS = [["right"], ["right", "A"]]
    env2 = gym_super_mario_bros.make('SuperMarioBros-v3')
    env2 = JoypadSpace(env2, MOVEMENTS)

    run(env1)
