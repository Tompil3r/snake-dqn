from env import SnakeEnv
import time

env = SnakeEnv()


# state = env.reset()
for i in range(5):
    done = False
    state = env.reset()
    while not done:
        actions = env.render(user_mode=True)
        # new_state, reward, done, info = env.step(env.action_space.sample())
        for action in actions:
            new_state, reward, done, info = env.step(action)
            print(reward, done, info, end='\r')
        
        if len(actions) == 0:
            new_state, reward, done, info = env.step(None)
            print(reward, done, info, end='\r')

env.close()