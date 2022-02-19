import math
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T
import numpy as np
import cv2
# Gym
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


class GrayScaleObservation(gym.ObservationWrapper):
    #numpy (h, w, 3) --> torch (1, h, w) 
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    # torch (h, w) --> torch (84, 84)
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class NumpyingObservation(gym.ObservationWrapper):
    # torch (4, 84, 84) --> numpy (4, 84, 84)        
    def observation(self, observation):
        obs = np.array(observation)
        return obs


class RedefineRewardInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_obs, reward, done, info = super(RedefineRewardInfo, self).step(action)
        x = info["x_pos"]
        if info["life"] <= 1:
            done = True             #One episode is 1 live not 3
            self.reset()
        if done:    
            reward -= 15             #Additional death punishment
        # if done:                  #For a non-terminal game. Mario just try to not die and not getting blocked.
        #     self.reset()  
        #     done = False
        try:                
            if self.position_mario_x == x:
                reward -= 2         #Standing still punishment
            # elif self.position_mario_x > x:
            #     reward += math.log(x)   #Bonus for going far in the level
        except AttributeError:
            self.position_mario_x = info["x_pos"]
        self.position_mario_x = x
        return next_obs, reward, done, info


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
class DisplayMarioPerspective(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.t = 0
        cv2.namedWindow('Mario Perspective', cv2.WINDOW_NORMAL)
        
    def observation(self, observation):
        self.t += 1
        if self.t % 4 == 0:
            img = observation[0]
            cv2.imshow('Mario Perspective', img)
            
        return observation

def load_smb_env(obs_complexity = 1, n_side = 84, n_stack = 1, n_skip = 4):
    # torch (84, 84) --> torch (4, 84, 84)
    game_id = f"SuperMarioBros-1-1-v{3-obs_complexity}"
    env = gym_super_mario_bros.make(game_id)
    MOVEMENTS = [["right"], ["right", "A"]]
    env = JoypadSpace(env, MOVEMENTS)
    
    # Apply Wrappers to environment
    env = GrayScaleObservation(env)             #Gray the observation, reducing channels without loosing information.
    env = ResizeObservation(env, shape=n_side)  #Resize obs shape to (84, 84) for dimension reduction while keeping sufficient infomation
    env = FrameStack(env, num_stack=n_stack)    #Stack last 4 frames to obtain a (4, 84, 84) array
    env = NumpyingObservation(env)              #Transform obs from LazyFrame to np array
    env = SkipFrame(env, n_skip)                #Skip some frames
    env = RedefineRewardInfo(env)               #Redefines reward
    env = DisplayMarioPerspective(env)          #Display 84x84 observation
    return env