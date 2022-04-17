# Python library
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
    env = DisplayMarioPerspective(env)
    return env


def train(env, dir, timesteps, verbose=1, policy="CnnPolicy", lr=.003, ns=8, bs=128, ne=12, gamma=.99, gae_lambda=.95):
    model = PPO("CnnPolicy", env=env, learning_rate=lr, n_steps=ns, batch_size=bs,
                n_epochs=ne, gamma=gamma, gae_lambda=gae_lambda, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    model.save(dir)


def run(dir, env):
    model = PPO.load(dir)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(.01666)


if __name__ == "__main__":

    n_side = 84
    n_stack = 4
    env1 = load_smb_env_blind(
        obs_complexity=0, n_side=n_side, n_stack=n_stack, n_skip=4)
    n_actions = env1.action_space.n

    MOVEMENTS = [["right"], ["right", "A"]]
    env2 = gym_super_mario_bros.make('SuperMarioBros-v3')
    env2 = JoypadSpace(env2, MOVEMENTS)

    train(env1, "ppo_test", 60)
    run("ppo_test", env1)
