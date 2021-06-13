import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from collections import namedtuple, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
from tensorflow.python.keras.backend_config import epsilon


Memory = namedtuple('Memory', (['state', 'action', 'reward', 'next_state', 'done']))


class DQNAgent():
    def __init__(self, state_shape, nb_actions, model=None, target_model=None, memory_limit=50_000, gamma=.99,
    eps=1., min_eps=.1, eps_decay=.97, learning_rate=.002):
        self.state_shape = state_shape
        self.state_sample_shape = (1,) + self.state_shape
        self.nb_actions = nb_actions

        self.memory = deque(maxlen=memory_limit)

        self.gamma = gamma
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate

        self.model = model if model is not None else self.build_default_model(name='model')
        self.target_model = target_model if target_model is not None else self.build_default_model(name='target-model')
        self.update_target_weights()


    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())
    

    def build_default_model(self, name='model'):
        model = Sequential(layers=[
            Flatten(input_shape=self.state_shape),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.nb_actions, activation='linear'),
        ], name=name)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        
        return model
    

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append(Memory(state, action, reward, next_state, done))


    def create_experiences(self, env, nb_steps, nb_max_episode_steps=-1):
        done = False
        state = env.reset()

        for step in range(nb_steps):
            if done:
                state = env.reset()
            
            state = np.reshape(state, self.state_sample_shape)
            action = self.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            next_state = np.reshape(next_state, self.state_sample_shape)

            self.store_experience(state, action, reward, next_state, done)
            state = next_state

            if step == nb_max_episode_steps:
                done = True


    def get_batch(self, batch_size):
        try:
            return random.sample(self.memory, batch_size)
        except:
            return self.memory

    
    def act(self, state, training=False):
        if not training:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)
        
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.nb_actions)
        
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)
    

    def replay_experience(self, batch_size):
        minibatch = self.get_batch(batch_size)

        for memory in minibatch:
            target = self.model.predict(memory.state)

            if memory.done:
                target[0][memory.action] = memory.reward

            else:
                future_q_values = self.target_model.predict(memory.next_state)[0]
                target[0][memory.action] = memory.reward + self.gamma * np.amax(future_q_values)

            self.model.fit(memory.state, target, epochs=1, verbose=0)

            self.eps = max(self.min_eps, self.eps*self.eps_decay)
    

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    

    def load_weights(self, filepath, update_target_weights=True):
        self.model.load_weights(filepath)

        if update_target_weights:
            self.update_target_weights()
    

    def fit(self, env, nb_steps, batch_size=32, target_weights_update=10_000, nb_max_episode_steps=-1, verbose=1, visualize=False):
        episodes = []
        rewards = []
        steps = []

        episode_nb = 0
        episode_step = 0
        episode_reward = 0
        done = False

        state = env.reset()

        for step in range(nb_steps):
            if visualize:
                env.render()

            if done:
                episodes.append(episode_nb)
                rewards.append(episode_reward)
                steps.append(episode_step)

                episode_step = 0
                episode_reward = 0
                done = False

                state = env.reset()

                if verbose == 1:
                    print(f'step {step}/{nb_steps} - episode reward {episode_reward} - {step*100/nb_steps}%', end='\r')


            state = np.reshape(state, self.state_sample_shape)
            action = self.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            next_state = np.reshape(next_state, self.state_sample_shape)

            self.store_experience(state, action, reward, next_state, done)
            state = next_state

            if step % target_weights_update == 0:
                self.update_target_weights()
            
            self.replay_experience(batch_size)

            if nb_max_episode_steps == episode_step:
                done = True

            if verbose == 2:
                print(f'step {step+1}/{nb_steps} - reward {reward} - {(step+1)*100/nb_steps}%', end='\r')

        return {'episodes':episodes, 'rewards':rewards, 'steps':steps}


    def test(self, env, nb_episodes, nb_max_episode_steps=-1, verbose=1, visualize=True):
        episodes = []
        rewards = []
        steps = []

        for episode in range(nb_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                if visualize:
                    env.render()
                
                state = np.reshape(state, self.state_sample_shape)
                action = self.act(state)

                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                state = next_state

                episode_step += 1

                if episode_step == nb_max_episode_steps:
                    done = True

                if verbose == 2:
                    print(f'step {episode_step} - episode {episode+1}/{nb_episodes} - reward {reward} - {(episode+1)*100/nb_episodes}%', end='\r')
            
            episodes.append(episode)
            rewards.append(episode_reward)
            steps.append(episode_step)

            if verbose == 1:
                print(f'episode {episode+1}/{nb_episodes} - episode reward {episode_reward} - {(episode+1)*100/nb_episodes}%')
        
        return {'episodes':episodes, 'rewards':rewards, 'steps':steps}