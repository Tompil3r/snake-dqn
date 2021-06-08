from collections import deque, namedtuple
import random
import numpy as np


Experience = namedtuple('Sample', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemory():
    def __init__(self, limit):
        self.limit = limit
        self.states = deque(maxlen=limit)
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.next_states = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)

        # self.memory = deque(maxlen=limit)

    
    def store_experience(self, state, action, reward, next_state, done):
        # store a memory
        # self.memory.append(Experience(state, action, reward, next_state, done))
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)
    

    def sample(self, batch_size):
        states = deque()
        actions = deque()
        rewards = deque()
        next_states = deque()
        terminals = deque()

        memory_size = len(self.states)

        if memory_size >= batch_size:
            # enough experiences
            indices = random.sample(range(memory_size), batch_size)
        
        else:
            # not enough experiences
            # avoid situation - use warmup to produce experience
            indices = np.random.randint(0, memory_size, batch_size)
        
        for idx in indices:
            states.append(self.states[idx])
            actions.append(self.actions[idx])
            rewards.append(self.rewards[idx])
            next_states.append(self.next_states[idx])
            terminals.append(self.terminals[idx])

        return states, actions, rewards, next_states, terminals

'''
    def sample(self, batch_size, indices=None):
        if indices is not None:
            # if indices given return the samples pointed by the indices
            return self.memory[indices]
        
        memory_size = len(self.memory)

        if memory_size >= batch_size:
            # if samples are not given and batch size is smaller or equal to the amount of samples in memory
            # then return random samples from memory (no duplicates)
            return random.sample(self.memory, batch_size)
        
        # if samples are not given and the batch size is larger the amount of samples in memory
        # avoid situation - use warmup to initialize enought samples at start
        experience = []
        indices = np.random.randint(0, memory_size, batch_size)
        
        for idx in indices:
            experience.append(self.memory[idx])

        return experience
'''