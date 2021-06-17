import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from collections import namedtuple, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import time


Memory = namedtuple('Memory', ['states', 'actions', 'rewards', 'next_states', 'terminals'])


class DQNAgent():
    def __init__(self, state_shape, nb_actions, model=None, target_model=None, memory_limit=50_000, gamma=.99,
    eps=1., min_eps=.1, eps_decay=.995, learning_rate=.0001):
        self.state_shape = state_shape
        self.state_batch_shape = (1,) + self.state_shape
        self.nb_actions = nb_actions

        self.memory = Memory(deque(maxlen=memory_limit), deque(maxlen=memory_limit), deque(maxlen=memory_limit),
        deque(maxlen=memory_limit), deque(maxlen=memory_limit))

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
    

    def store_experience(self, state, action, reward, next_state, terminal):
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.next_states.append(next_state)
        self.memory.terminals.append(terminal)


    def create_experiences(self, env, nb_steps, nb_max_episode_steps=-1):
        done = False
        state = env.reset()

        for step in range(nb_steps):
            if done:
                state = env.reset()
            
            state = self.preprocess_state(state)
            action = self.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            next_state = self.preprocess_state(next_state)

            self.store_experience(state, action, reward, next_state, done)
            state = next_state

            if step == nb_max_episode_steps:
                done = True


    def get_batch(self, batch_size):
        memory_size = len(self.memory.states)
        states = np.empty(shape=(batch_size,) + self.state_shape, dtype=np.float64)
        actions = np.empty(shape=(batch_size,), dtype=np.int32)
        rewards = np.empty(shape=(batch_size,), dtype=np.float64)
        next_states = np.empty(shape=(batch_size,) + self.state_shape, dtype=np.float64)
        terminals = np.empty(shape=(batch_size,), dtype=np.int8)

        if batch_size <= memory_size:
            indices = random.sample(range(memory_size), batch_size)
            
        else:
            indices = np.random.randint(0, memory_size, batch_size)

        for sample_idx, idx in enumerate(indices):
            states[sample_idx] = self.memory.states[idx]
            actions[sample_idx] = self.memory.actions[idx]
            rewards[sample_idx] = self.memory.rewards[idx]
            next_states[sample_idx] = self.memory.next_states[idx]
            terminals[sample_idx] = not self.memory.terminals[idx] # invert terminals value

        return states, actions, rewards, next_states, terminals

    
    def act(self, state, training=False):
        if not training:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)
        
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.nb_actions)
        
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)
    

    def replay_experience(self, batch_size):
        states, actions, rewards, next_states, terminals = self.get_batch(batch_size)

        target = self.model.predict(states)
        future_q_values = self.target_model.predict(next_states)

        for idx in range(batch_size):
            # if current state is terminal -> terminals[idx] = 0 -> target = rewards[idx] + terminals[idx] * self.gamma 
            #  * np.amax(future_q_values[idx]) = reward only

            # if current state is NOT terminal -> terminals[idx] = 1 -> target = rewards[idx] + terminals[idx] * self.gamma
            # * np.amax(future_q_values[idx]) = reward + expected return from next state

            target[idx, actions[idx]] = rewards[idx] + terminals[idx] * self.gamma * np.amax(future_q_values[idx])

        self.model.fit(states, target, epochs=1, verbose=0)
        self.eps = max(self.min_eps, self.eps*self.eps_decay)
    

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    

    def load_weights(self, filepath, update_target_weights=True):
        self.model.load_weights(filepath)

        if update_target_weights:
            self.update_target_weights()
    

    def preprocess_state(self, state):
        return np.reshape(state, self.state_batch_shape)
    

    def fit(self, env, nb_steps, batch_size=32, target_weights_update=10_000, nb_max_episode_steps=-1, verbose=1, visualize=False):
        start_time = time.perf_counter()

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

                if verbose == 1:
                    print((f'step {step}/{nb_steps} - episode reward {episode_reward} - {step*100/nb_steps}% - time '
                    f'{time.perf_counter() - start_time:.3f}s'), end='\r')

                episode_nb += 1
                episode_step = 0
                episode_reward = 0
                done = False

                state = env.reset()


            state = self.preprocess_state(state)
            action = self.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            next_state = self.preprocess_state(next_state)

            self.store_experience(state, action, reward, next_state, done)
            state = next_state

            if step % target_weights_update == 0:
                self.update_target_weights()
            
            self.replay_experience(batch_size)

            if nb_max_episode_steps == episode_step:
                done = True

            if verbose == 2:
                print(f'step {step+1}/{nb_steps} - reward {reward} - {(step+1)*100/nb_steps}% - time {time.pert_counter() - start_time:.3f}s',
                end='\r')

            episode_step += 1

        return {'episodes':episodes, 'rewards':rewards, 'steps':steps}


    def test(self, env, nb_episodes, nb_max_episode_steps=-1, verbose=1, visualize=True):
        start_time = time.perf_counter()

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
                
                state = self.preprocess_state(state)
                action = self.act(state)

                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                state = next_state

                if episode_step == nb_max_episode_steps:
                    done = True

                if verbose == 2:
                    print((f'step {episode_step} - episode {episode+1}/{nb_episodes} - reward {reward} - {(episode+1)*100/nb_episodes}% - '
                    f'time {time.perf_counter() - start_time:.3f}s'), end='\r')
                
                episode_step += 1
            
            episodes.append(episode)
            rewards.append(episode_reward)
            steps.append(episode_step)

            if verbose == 1:
                print((f'episode {episode+1}/{nb_episodes} - episode reward {episode_reward} - {(episode+1)*100/nb_episodes}% - '
                f'time {time.perf_counter() - start_time:.3f}s'), end='\r')
        
        return {'episodes':episodes, 'rewards':rewards, 'steps':steps}