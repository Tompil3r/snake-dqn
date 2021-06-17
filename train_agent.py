import gym
from agent import DQNAgent
import numpy as np


env = gym.make('CartPole-v0')

nb_actions = env.action_space.n
state_shape = env.observation_space.shape

agent = DQNAgent(state_shape, nb_actions)

agent.create_experiences(env, 100)
history = agent.fit(env, 1000)

agent.save_weights('model_weights.h5')

history = agent.test(env, 100, visualize=False)
print(np.mean(history['rewards']))
