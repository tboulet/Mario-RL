from typing import Tuple
import torch
import numpy as np
import random as rd

#Memory using tensor
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
        

#Memory using lists    
class Memory():
    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=float('inf')):
        self.max_memory_len = max_memory_len
        self.MEMORY_KEYS = MEMORY_KEYS        
        self.trajectory = {key : list() for key in MEMORY_KEYS}

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the buffer. Complexity = O(size_transition)
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        for val, key in zip(transition, self.MEMORY_KEYS):
            if type(val) == bool: val = int(val)
            self.trajectory[key].append(val)
        if len(self) > self.max_memory_len:
            for val, key in zip(transition, self.MEMORY_KEYS):
                self.trajectory[key].pop()

    def sample(self, sample_size=None, pos_start=None, method='last'):
        '''Samples several transitions from memory, using different methods. Complexity = O(sample_size x transition_size)
        sample_size : the number of transitions to sample, default all.
        pos_start : the position in the memory of the first transition sampled, default 0.
        method : the method of sampling in "all", "last", "random", "all_shuffled", "batch_shuffled", "batch".
        return : a list containing a list of size sample_size for each kind of element stored.
        '''
        if sample_size is None:
            sample_size = len(self)
        else:
            sample_size = min(sample_size, len(self))
                    
        if method == 'all':
            #Each elements in order.
            indexes = [n for n in range(len(self))]

        elif method == 'last':
            #sample_size last elements in order.
            indexes = [n for n in range(len(self) - sample_size, len(self))]

        elif method == 'random':
            #sample_size elements sampled.
            indexes = [rd.randint(0, len(self) - 1) for _ in range(sample_size)]

        elif method == 'all_shuffled':
            #Each elements suffled.
            indexes = [n for n in range(len(self))]
            rd.shuffle(indexes)

        elif method == "batch":
            #Element n° pos_start and sample_size next elements, in order.
            indexes = [pos_start + n for n in range(sample_size)]
            
        elif method == 'batch_shuffled':
            #Element n° pos_start and sample_size next elements, shuffled.
            indexes = [pos_start + n for n in range(sample_size)]
            rd.shuffle(indexes)

        else:
            raise NotImplementedError('Not implemented sample')

        trajectory = list()
        for elements in self.trajectory.values():
            sampled_elements = torch.tensor(np.array([elements[idx] for idx in indexes]))
            if len(sampled_elements.shape) == 1:
                sampled_elements = torch.unsqueeze(sampled_elements, -1)
            trajectory.append(sampled_elements)

        return trajectory

    def __len__(self):
        return len(self.trajectory[self.MEMORY_KEYS[0]])

    def __empty__(self):
        self.trajectory = {key : list() for key in self.MEMORY_KEYS}
        
