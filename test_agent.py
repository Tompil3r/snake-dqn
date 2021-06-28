from env import SnakeEnv
from agent import DQNAgent
import numpy as np
import models


env = SnakeEnv()

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions

model = models.build_model_1(state_shape, nb_actions)
target_model = models.build_model_1(state_shape, nb_actions)
agent = DQNAgent(state_shape, nb_actions, model=model, target_model=target_model)

agent.load_weights('model_weights.h5')
scores = agent.test(env, 10, visualize=True)
env.close()

print(np.mean(scores['rewards']))