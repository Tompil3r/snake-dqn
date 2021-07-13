from env import SnakeEnv
from agent import DQNAgent
import numpy as np
import os
import models
import pickle


env = SnakeEnv()

weights_path = 'model_weights.h5'
log_path = 'training_log.pkl'

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions
training_steps = 18_000_000

model = models.build_model_8(state_shape, nb_actions)
target_model = models.build_model_8(state_shape, nb_actions)

agent = DQNAgent(state_shape, nb_actions, model=model, target_model=target_model, memory_limit=40_000, eps_decay_steps=training_steps)

agent.create_experiences(env, 1000)

if os.path.exists(weights_path):
    agent.load_weights(weights_path)

history = agent.fit(env, training_steps, batch_size=64, validation_steps=100_000, validation_episodes=10)

agent.save_weights(weights_path)

with open(log_path, 'wb+') as file:
    pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)