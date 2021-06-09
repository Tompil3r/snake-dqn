import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory



def build_model(state_shape, nb_actions, optimizer, name='model'):
    model = Sequential(layers=[
        Flatten(input_shape=state_shape),
        Dense(units=32, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=nb_actions, activation='linear'),

    ], name=name)


def build_agent(model, nb_actions, memory_limit, eps_max_value, eps_min_value, nb_steps, eps_test_value=.0):
    policy = EpsGreedyQPolicy()
    decay_policy = LinearAnnealedPolicy(policy, 'eps', eps_max_value, eps_min_value, eps_test_value, nb_steps)
    test_policy = GreedyQPolicy()
    memory = SequentialMemory(limit=memory_limit, window_length=1)
    
    return DQNAgent(model, memory=memory, policy=decay_policy, test_policy=test_policy, enable_double_dqn=True, nb_actions=nb_actions)
