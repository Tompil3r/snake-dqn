import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from env import SnakeEnv
from agent import DQNAgent


def main():
    env = SnakeEnv()
    user_control = True

    state_shape = env.observation_space.shape
    nb_actions = env.action_space.nb_actions

    agent = DQNAgent(state_shape, nb_actions)
    weights_path = 'model_weights.h5'
    weights_loading_try = 0


    while True:
        try:
            agent.load_weights(weights_path)
            break

        except Exception:
            weights_loading_try += 1
            
            if weights_loading_try > 1:
                print('Unable to load model weights')
                break


    while True:
        game_started = False
        state = env.reset()
        done = False

        while user_control and not game_started:
            user_actions, switch_mode = env.render(user_control=user_control)

            if switch_mode:
                user_control = not user_control

            if user_actions is not None and len(user_actions) > 0:
                for action in user_actions:
                    new_state, reward, done, info = env.step(action)
                    state = new_state
            
                game_started = True


        while not done:
            user_actions, switch_mode = env.render(user_control=user_control)

            if switch_mode:
                user_control = not user_control

            if user_control:
                if user_actions is not None and len(user_actions) > 0:
                    for action in user_actions:
                        new_state, reward, done, info = env.step(action)
                        state = new_state
                
                else:
                    new_state, reward, done, info = env.step(None)
                    state = new_state
            
            else:
                state = agent.preprocess_state(state)
                action = agent.select_action(state)
                new_state, reward, done, info = env.step(action)

                state = new_state

    
if __name__ == '__main__':
    main()
