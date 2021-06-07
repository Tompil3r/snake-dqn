from collections import deque, namedtuple
import random
import numpy as np


Sample = namedtuple('Sample', ['state', 'action', 'reward', 'next_state'])


class Memory():
    def __init__(self, limit):
        self.limit = limit
        self.data = deque(maxlen=limit)

    
    def save_experience(self, state, action, reward, next_state):
        # store an experience
        self.data.append(Sample(state, action, reward, next_state))
    

    def sample(self, batch_size, indices=None):
        if indices is not None:
            # if indices given return the samples pointed by the indices
            return self.data[indices]
        
        memory_size = len(self.data)


        if memory_size >= batch_size:
            # if samples are not given and batch size is smaller or equal to the amount of samples in memory
            # then return random samples from memory (no duplicates)
            return self.data[random.sample(range(memory_size), batch_size)]
        
        # if samples are not given and the batch size is larger the amount of samples in memory
        # avoid situation - use warmup to initialize enought samples at start
        return self.data[np.random.randint(0, memory_size, batch_size)]
