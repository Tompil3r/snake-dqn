from env import SnakeEnv
from agent import DQNAgent
import numpy as np
import models


env = SnakeEnv()
# env.termination_step = 50

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions

model = models.build_model_8(state_shape, nb_actions)
target_model = models.build_model_8(state_shape, nb_actions)
agent = DQNAgent(state_shape, nb_actions, model=model, target_model=target_model)

agent.load_weights('model_weights.h5')
history = agent.test(env, 100, visualize=False)
env.close()

print('mean rewards:', np.mean(history['rewards']))
print('mean scores:', np.mean(history['scores']))