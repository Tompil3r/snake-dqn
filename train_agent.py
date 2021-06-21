from env import SnakeEnv
from agent import DQNAgent
import numpy as np
import os


env = SnakeEnv()

weights_path = 'model_weights.h5'

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions
training_steps = 4_000_000
decay_steps = int(training_steps * 0.7)

agent = DQNAgent(state_shape, nb_actions, eps_decay_steps=decay_steps)

agent.create_experiences(env, 1000)

if os.path.exists(weights_path):
    agent.load_weights(weights_path)

history = agent.fit(env, training_steps, batch_size=64)

agent.save_weights(weights_path)
