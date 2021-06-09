import numpy as np


class ActionSpace():
    def __init__(self, nb_actions):
        
        assert type(nb_actions) is int, 'nb_actions must to be integer'
        self.nb_actions = nb_actions

    
    def sample(self):
        return np.random.randint(0, self.nb_actions)

    
    def contains(self, x):
        assert type(x) is int, 'x must be integer'
        return x >= 0 and x < self.nb_actions