from env import SnakeEnv
from agent import DQNAgent
import numpy as np

env = SnakeEnv()

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions
training_steps = 10000

agent = DQNAgent(state_shape, nb_actions, eps_decay_steps=training_steps)

agent.create_experiences(env, 1000)

history = agent.fit(env, training_steps, batch_size=64)

agent.save_weights('model_weights.h5')
