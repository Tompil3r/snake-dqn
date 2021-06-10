from env.env import SnakeEnv
import time
import numpy as np

env = SnakeEnv()


state_0 = env.reset()
state_1, reward, done, info = env.step(env.action_right)
state_2, reward, done, info = env.step(env.action_right)
state_3, reward, done, info = env.step(env.action_right)
state_4, reward, done, info = env.step(env.action_right)
state_5, reward, done, info = env.step(env.action_right)


print(state_0, end='\n\n')
print(state_1, end='\n\n')
print(state_2, end='\n\n')
print(state_3, end='\n\n')
print(state_4, end='\n\n')
print(state_5, end='\n\n')
