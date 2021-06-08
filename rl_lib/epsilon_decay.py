import numpy as np


class EpsilonDecay():
    ''' Abstract class - each specific epsilon decay is implemented in its own class
    '''
    def __call__(self, **kwargs):
        raise NotImplementedError()


class ExpDecay(EpsilonDecay):
    
    def __call__(self, **kwargs):
        eps = kwargs['eps']
        decay_rate = kwargs['decay_rate']
        min_eps = kwargs.get('min_eps', .01)
        max_eps = kwargs.get('max_eps', 1.)

        # epsilon changes with respect to itself
        return min(max(eps*decay_rate, min_eps), max_eps)


class LinearDecay(EpsilonDecay):
    
    def __call__(self, **kwargs):
        eps = kwargs['eps']
        decay_rate = kwargs['decay_rate']
        min_eps = kwargs.get('min_eps', .01)
        max_eps = kwargs.get('max_eps', 1.)

        # epsilon changes by a constant number
        return min(max(eps+decay_rate, min_eps), max_eps)


class EpisodicDecay(EpsilonDecay):

    def __call__(self, **kwargs):
        episode = kwargs['episode']
        decay_rate = kwargs['decay_rate']
        min_eps = kwargs.get('min_eps', .01)
        max_eps = kwargs.get('max_eps', 1.)

        # epsilon is calculated by the number of episode
        return (min_eps + (max_eps - min_eps) * np.exp(-decay_rate*episode))

