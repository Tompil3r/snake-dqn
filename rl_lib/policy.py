import numpy as np


class Policy():
    ''' Abstract class for Policy, each class will implement the necessary functions
    '''

    def select_action(self, **kwargs):
        # Each policy implementation will implement this function
        raise NotImplementedError()
    

    def on_episode_end(self, **kwargs):
        # Each policy implementation will implement this function
        raise NotImplementedError()


class GreedyQPolicy(Policy):
    
    def select_action(self, q_values):
        # q values array must be 1 dimensional
        assert q_values.ndim == 1
        
        # return the index of the highest q value
        return np.argmax(q_values)
    

    def on_episode_end(self, episode):
        pass


class EpsilonGreedyQPolicy(Policy):
    def __init__(self, eps=1., min_eps=.001, max_eps=1., decay_rate=0, decay=None):
        super().__init__()

        self.eps = eps
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.decay_rate = decay_rate
        self.decay = decay


    def select_action(self, q_values):
        # q values array must be 1 dimensional
        assert q_values.ndim == 1

        if np.random.uniform() < self.eps:
            nb_actions = q_values.shape[0]
            
            # return index of random action
            return np.random.randint(0, nb_actions)
        
        else:
            # return the index of the highest q value
            return np.argmax(q_values)
    

    def on_episode_end(self, episode):
        if self.decay is not None:
            # decay epsilon using the decay_func attribute if available, else skip
            self.decay(eps=self.eps, decay_rate=self.decay_rate, min_eps=self.min_eps, max_eps=self.max_eps, episode=episode)

        
    class BoltzmannQPolicy(Policy):
        def __init__(self, tau=1., clip=(-500., 500.)):
            super().__init__()

            self.tau = tau
            self.clip = clip


        def select_action(self, q_values):
            assert q_values.ndim == 1

            # necessary?
            q_values = q_values.astype('float64')
            
            nb_actions = q_values.shape[0]
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))

            # probability distribution for each action
            probs = exp_values / np.sum(exp_values)

            # choose action index with respect to the probability distribution
            return np.random.choice(range(nb_actions), p=probs)
