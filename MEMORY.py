from typing import Tuple
import torch
import numpy as np
import random as rd
from collections import deque, namedtuple 


class Memory():
    '''Memory class for keeping observations, actions, rewards, ... in memory.
    MEMORY_KEYS : a list of string, each string being the name of a kind of element to remember.
    max_memory_len : maximum memory lenght, no limit by default.
    '''

    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=None):
        self.max_memory_len = max_memory_len
        self.MEMORY_KEYS = MEMORY_KEYS
        self.trajectory = {}

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the buffer.
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        for val, key in zip(transition, self.MEMORY_KEYS):
            if type(val) == bool:
                val = int(val)
            val = torch.tensor(val)
            val = torch.unsqueeze(val, 0)
            if len(val.shape) == 1:
                val = torch.unsqueeze(val, 0)
            try:
                self.trajectory[key] = torch.concat((val, self.trajectory[key]), axis = 0)  #(memory_lenght, n_?)
            except KeyError:
                self.trajectory[key] = val
        self.memory_len = len(self.trajectory[self.MEMORY_KEYS[0]])
        
                
    
    def sample(self, sample_size=None, pos_start=None, method='last', func = None):
        '''Samples several transitions from memory, using different methods.
        sample_size : the number of transitions to sample, default all.
        pos_start : the position in the memory of the first transition sampled, default 0.
        method : the method of sampling in "all", "last", "random", "all_shuffled", "batch_shuffled".
        func : a function applied to each elements of the transitions, usually converting a np array into a pytorch/tf/jax tensor.
        return : a list containing a list of size sample_size for each kind of element stored.
        '''
        if method == 'all':
            trajectory = [self.trajectory[key] for key in self.MEMORY_KEYS]

        elif method == 'last':
            trajectory = [self.trajectory[key][-sample_size:]
                          for key in self.MEMORY_KEYS]

        elif method == 'random':
            indexes = np.random.permutation(self.memory_len)[:sample_size]
            trajectory = [self.trajectory[key][indexes]
                          for key in self.MEMORY_KEYS]

        elif method == 'all_shuffled':
            trajectory = [self.trajectory[key][self.max_memory_len]
                          for key in self.MEMORY_KEYS]

        elif method == 'batch_shuffled':
            trajectory = [self.trajectory[key][pos_start: pos_start +
                                                         sample_size] 
                        for key in self.MEMORY_KEYS]

        else:
            raise NotImplementedError('Not implemented sample')
        
        if func is not None:
            trajectory = [func(elem) for elem in trajectory]
        return trajectory

    def __len__(self):
        return self.memory_len

    def __empty__(self):
        self.trajectory = {}
        

        
class Memory():
    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=None):
        self.max_memory_len = max_memory_len
        self.memory_len = 0
        self.MEMORY_KEYS = MEMORY_KEYS        
        self.trajectory = {key : list() for key in MEMORY_KEYS}

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the buffer.
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        for val, key in zip(transition, self.MEMORY_KEYS):
            self.trajectory[key].append(val)

    def sample(self, sample_size=None, pos_start=None, method='last', func = None):
        '''Samples several transitions from memory, using different methods.
        sample_size : the number of transitions to sample, default all.
        pos_start : the position in the memory of the first transition sampled, default 0.
        method : the method of sampling in "all", "last", "random", "all_shuffled", "batch_shuffled", "batch".
        func : a function applied to each elements of the transitions.
        return : a list containing a list of size sample_size for each kind of element stored.
        '''
        sample_size = min(sample_size, len(self))
                    
        if method == 'all':
            indexes = np.arange(len(self))

        elif method == 'last':
            indexes = np.arange(len(self))[-sample_size:]

        elif method == 'random':
            indexes = np.random.permutation(len(self))[:sample_size]

        elif method == 'all_shuffled':
            indexes = np.random.permutation(len(self))

        elif method == "batch":
            indexes = np.arange(pos_start, pos_start + sample_size)
            
        elif method == 'batch_shuffled':
            indexes = np.arange(pos_start, pos_start + sample_size)
            np.random.shuffle(indexes)

        else:
            raise NotImplementedError('Not implemented sample')
        
        trajectory = list()
        for elements in self.trajectory.values():
            sampled_elements = list()
            for idx in indexes:
                elem = elements[idx]
                if type(elem) == bool:
                    elem = int(elem)
                sampled_elements.append(elem)
                
            sampled_elements = np.array(sampled_elements)
            sampled_elements = torch.tensor(sampled_elements)
            if len(sampled_elements.shape) == 1:
                sampled_elements = torch.unsqueeze(sampled_elements, -1)
            trajectory.append(sampled_elements)

        if func is not None:
            trajectory = [func(elem) for elem in trajectory]
            
        return trajectory

    def __len__(self):
        return len(self.trajectory[self.MEMORY_KEYS[0]])

    def __empty__(self):
        self.trajectory = dict()
        
