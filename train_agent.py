import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from env.env import SnakeEnv
from nn.model import Model
from nn.agent import Agent


env = SnakeEnv()

state_shape = env.observation_space.shape
print(state_shape)
nb_actions = env.action_space.nb_actions
memory_limit = 10000
# eps_max_value = 1.
# eps_min_value = 0.1
training_nb_steps = 100_000
eps_decay_nb_steps = training_nb_steps // 5

weights_path = 'agent_misc/snake_model_weights.h5'

model = Model.build_model2(state_shape, nb_actions, name='Snake-Model')
agent = Agent.build_agent2(model, nb_actions, memory_limit)
agent.compile('adam', metrics=['mae'])
# agent.load_weights(weights_path)

training_history = agent.fit(env, training_nb_steps, visualize=False, verbose=1)
agent.save_weights(weights_path, overwrite=True)