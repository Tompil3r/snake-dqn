from env import SnakeEnv
from agent import DQNAgent
import numpy as np
import os
import models

env = SnakeEnv()

weights_path = 'model_weights.h5'

state_shape = env.observation_space.shape + (1,)
nb_actions = env.action_space.nb_actions
learning_rate = .0001
training_steps = 4_000_000

# model = models.build_conv_model(state_shape, nb_actions, learning_rate)
# target_model = models.build_conv_model(state_shape, nb_actions, learning_rate)

# agent = DQNAgent(state_shape, nb_actions, model=model, target_model=target_model, eps_decay_steps=training_steps)
agent = DQNAgent(state_shape, nb_actions, eps=.1)

agent.create_experiences(env, 1000)

if os.path.exists(weights_path):
    agent.load_weights(weights_path)

history = agent.fit(env, training_steps, batch_size=64)

agent.save_weights(weights_path)
