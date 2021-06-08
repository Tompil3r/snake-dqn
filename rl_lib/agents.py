
import numpy as np


class DQNAgent():
    def __init__(self, nb_actions, model, memory, policy, test_policy=None, target_model=None, gamma=.99, target_model_update_steps=10000):
        self.nb_actions = nb_actions
        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.policy = policy
        self.test_policy = test_policy
        self.target_model = None

        self.gamma = gamma
        self.target_model_update_steps = target_model_update_steps

        # self.training = False
        # self.step = 0


    def update_target_weights(self):
        # copy weights from main model to target model
        self.target_model.set_weights(self.model.get_weights())


    def act(self, state, training=False):
        # get predicted q-values by the model
        state = np.reshape(state, (1, len(state)))
        q_values = self.model.predict(state)[0]

        if training or self.test_policy is None:
            # if model is training or agent has no test policy - return the output of normal policy
            return self.policy.select_action(q_values)
        
        # if model is not training and has test policy - retyrb the output of test policy
        return self.test_policy.select_action(q_values)
    

    def warmup(self, env, nb_warmup_steps):
        done = False
        episode = 0
        state = env.reset()


        for step in range(nb_warmup_steps):
            if done:
                episode += 1
                state = env.reset()
                self.policy.on_episode_end(episode)
                done = False
            
            # get action from agent
            action = self.act(state, training=True)

            # execute agent's action
            next_state, reward, done, info = env.step(action)
                
            # store experience
            self.memory.store_experience(state, action, reward, next_state, done)
                
            # update current state
            state = next_state
            

    def replay(self, batch_size):
        # get mini batch
        states, actions, rewards, next_states, terminals = self.memory.sample(batch_size)
        
        # get current q values
        curr_q_values = self.model.predict(np.array(states))
        # get next q values
        next_q_values = self.model.predict(np.array(next_states))
        # curr_q_values are assign to target_q_values to avoid loss on actions the agent did not choose
        target_q_values = curr_q_values

        # set target q value for each state sample
        for idx in range(batch_size):
            if terminals[idx]:
                target_q_values[idx, actions[idx]] = rewards[idx]
            
            else:
                target_q_values[idx, actions[idx]] = self.gamma * np.amax(next_q_values[idx]) + rewards[idx]
        
        # train the network on the states and target q values
        self.model.fit(np.array(states), target_q_values, epochs=1, verbose=0)



    def fit(self, env, nb_steps, batch_size=32, verbose=1, visualize=False, nb_max_episode_steps=-1):
        episode_steps = []
        episode_rewards = []

        episode_reward = 0
        episode = 0
        episode_step = 0

        done = False

        state = env.reset()

        for step in range(nb_steps):
            # render environment if specified
            if visualize:
                env.render()

            # update variables after each episode
            if done:
                episode_rewards.append(episode_reward)
                episode_steps.append(episode_step+1)
                episode += 1
                episode_step = 0
                episode_reward = 0

                env.reset()
                self.policy.on_episode_end(episode)
                done = False
            

            # if needed, update the target model weights
            if self.target_model is not None and step % self.target_model_update_steps == 0:
                self.update_target_weights()

            # get action from agent
            action = self.act(state, training=True)

            # execute agent's action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # store experience
            self.memory.store_experience(state, action, reward, next_state, done)

            # update current state
            state = next_state

            # train model
            self.replay(batch_size)

            # advance episode step
            episode_step += 1

            # if episode step equals max episode steps allowed, done = True -> environment reset
            if episode_step == nb_max_episode_steps:
                done = True
            
            # verbose if specified
            if verbose == 1:
                print(f'{step+1}/{nb_steps} Completed - {((step+1)*100)/nb_steps:.2f}%', end='\r')
        
        
        env.close()
        return {'episode_steps':episode_steps, 'episode_rewards':episode_rewards}
    

    def test(self, env, nb_episodes, verbose=1, visualize=False, nb_max_episode_steps=-1):
        episode_steps = []
        episode_rewards = []

        for episode in range(nb_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done:
                if visualize:
                    env.render()

                # get action from agent
                action = self.act(state, training=False)

                # execute agent's action
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # store experience
                self.memory.store_experience(state, action, reward, next_state, done)

                # update current state
                state = next_state

                step += 1

                if done or step == nb_max_episode_steps:
                    break


            episode_steps.append(step)
            episode_rewards.append(episode_reward)

        env.close()
        return {'episode_steps':episode_steps, 'episode_rewards':episode_rewards}
    

    def save_weights(self, filename):
        self.model.save_weights(filename)

            