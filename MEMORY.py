import numpy as np

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
            val = np.array(val)
            batched_val = val.reshape((1, -1))      # (1, n_?)      
            #Add it to the already existing elements. If no elements, create the value for it.      
            try:
                self.trajectory[key] = np.concatenate((self.trajectory[key], batched_val), axis = 0)  #(memory_lenght, n_?)
            except KeyError:
                self.trajectory[key] = batched_val
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