from env import SnakeEnv
from agent import DQNAgent
import numpy as np


env = SnakeEnv()

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions

agent = DQNAgent(state_shape, nb_actions)

agent.load_weights('model_weights.h5')
scores = agent.test(env, 10, visualize=True)
env.close()

print(np.mean(scores['rewards']))