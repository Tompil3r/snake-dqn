import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from env.env import SnakeEnv
from nn.model import Model
from nn.agent import Agent


env = SnakeEnv()

state_shape = env.observation_space.shape
nb_actions = env.action_space.nb_actions
memory_limit = 10000
eps_max_value = 1.
eps_min_value = 0.01
training_nb_steps = 1000000
eps_decay_nb_steps = training_nb_steps // 5


model = Model.build_model(state_shape, nb_actions, name='Snake-Model')
agent = Agent.build_agent(model, nb_actions, memory_limit, eps_max_value, eps_min_value, eps_decay_nb_steps)
agent.compile('adam', metrics=['mae'])

training_history = agent.fit(env, training_nb_steps, visualize=False, verbose=1)
agent.save_weights('snake_model_weights.h5', overwrite=True)