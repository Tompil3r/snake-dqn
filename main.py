from env import SnakeEnv

env = SnakeEnv()


done = False
state = env.reset()

while not done:
    env.render()
    new_state, reward, done, info = env.step(env.action_space.sample())
    print(reward, done, info, end='\r')

env.close()