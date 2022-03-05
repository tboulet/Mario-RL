import numpy as np
import random as rd

import torch
        

#Memory using lists    
class Memory():
    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=float('inf')):
        self.max_memory_len = max_memory_len
        self.MEMORY_KEYS = MEMORY_KEYS        
        self.trajectory = {key : list() for key in MEMORY_KEYS}

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the buffer.
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        for val, key in zip(transition, self.MEMORY_KEYS):
            if type(val) == bool: val = int(val)
            self.trajectory[key].append(val)
        if len(self) > self.max_memory_len:
            for val, key in zip(transition, self.MEMORY_KEYS):
                self.trajectory[key].pop()

    def sample(self, sample_size=None, pos_start=None, method='last', as_tensor = True):
        '''Samples several transitions from memory, using different methods.
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
        
        elif method == 'episodic_batches':
            #Return a list L of list E of element-tensor, each list E being a batch
            indexes_dones = [0] + [idx for idx, d in enumerate(self.trajectory['done']) if d == 1]
            res_batches = list()
            for i in range(len(indexes_dones) - 1):
                idx, idx_next = indexes_dones[i:i+2]
                res_batches.append(
                    self.sample(pos_start = idx, sample_size = idx_next - idx, method = 'batch')
                )
            return res_batches

        else:
            raise NotImplementedError('Not implemented sample')

        trajectory = list()
        for elements in self.trajectory.values():
            sampled_elements = np.array([elements[idx] for idx in indexes])
            if len(sampled_elements.shape) == 1:
                sampled_elements = np.expand_dims(sampled_elements, -1)
            if as_tensor:
                sampled_elements = torch.Tensor(sampled_elements)                
            trajectory.append(sampled_elements)

        return trajectory

    def __len__(self):
        return len(self.trajectory[self.MEMORY_KEYS[0]])

    def __empty__(self):
        self.trajectory = {key : list() for key in self.MEMORY_KEYS}
        
