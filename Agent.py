import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Flatten, Input, Convolution2D
from keras.layers.convolutional import Conv2D
from keras import backend as K

from utils import preprocess, show_video

class Agent(threading.Thread):
    ''' individual advantage actor critic training agent '''
    def __init__(self, action_size, state_size, model, lstm, sess, optimizer, discount_factor, no_op_max, summary_ops):
        threading.Thread.__init__(self)

        # actions are (left, stay, right)
        self.action_size = action_size

        # env settings (4 stacked frames of 84 x 84)
        self.state_size = state_size

        # global model
        self.actor, self.critic = model

        # whether to use lstm
        self.lstm = lstm

        # pass in tf session
        self.sess = sess

        self.optimizer = optimizer

        self.discount_factor = discount_factor

        self.no_op_max = no_op_max

        # pass in tensorboard summary
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        # for calculating discounted rewards
        self.states, self.actions, self.rewards = [],[],[]

        # instantiate local version of model
        self.local_actor, self.local_critic = self.build_localmodel()

        # metrics
        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.update_freq = 20

    # Thread interactive with environment
    def run(self, total_episodes = 50000):
        ''' trains individual agent'''

        global episode

        env = gym.make(env_name)

        env = gym.wrappers.Monitor(env, 'training', force=True)

        global_step = 0

        with self.sess.as_default(), self.sess.graph.as_default():
            while episode < MAX_EPISODES:
                done = False
                dead = False

                # reset game stats
                step, score = 0, 0

                # 5 lives per game
                start_life = 5

                observe = env.reset()

                # do nothing at the start of episode to avoid sub-optimal performance
                for _ in range(random.randint(1, self.no_op_max)):
                    observe, _, _, _ = env.step(1)

                # no preceding frame, copy initial state 4 times
                state = preprocess(observe)

                # chronological order of state frames is newest to oldest: (4,3,2,1)
                state = np.stack((state, state, state, state), axis=2)
                state = np.reshape([state], (1, 84, 84, 4))

                while not done:
                    global_step += 1
                    step += 1

                    # action for the current state
                    action = self.get_action(state)

                    # map (0, 1, 2) to (1, 2, 3) corresponding to gym actions
                    action += 1

                    # take a step
                    observe, reward, done, info = env.step(action)

                    # process observation into state
                    next_state = preprocess(observe)
                    next_state = np.reshape([next_state], (1, 84, 84, 1))

                    # add previous 3 frames
                    next_state = np.append(next_state, state[:, :, :, :3], axis=3)

                    # update average max probability
                    self.avg_p_max += np.amax(self.actor.predict(np.float32(state / 255.)))

                    # missed ball, subtract lives
                    if start_life > info['ale.lives']:
                        dead = True
                        start_life = info['ale.lives']

                    # add to score
                    score += reward

                    # save <s, a, r> into replay memory
                    self.replay_memory(state, action, reward)

                    # dead, we reset the history, since previous states don't matter anymore
                    if dead:
                        state = np.stack((next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0]), axis=2)
                        state = np.reshape([state], (1, 84, 84, 4))
                        dead = False
                    else:
                        state = next_state

                    # train every n steps
                    if global_step % self.update_freq == 0:
                        self.train_step(done)
                        self.update_localmodel()

                    # if done, plot the score over episodes
                    if done:
                        print("episode:", episode, "  score:", score, "global step:", global_step)

                        stats = [score, self.avg_p_max / float(step), step]

                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])})

                        summary_str = self.sess.run(self.summary_op)

                        self.summary_writer.add_summary(summary_str, episode + 1)

                        self.avg_p_max, self.avg_loss = 0, 0

                        episode += 1

    def discount_rewards(self, rewards, done):
        ''' for the advantage estimate, we go backwards in time to accumulate the discounted reward component '''

        # nice ref explanation: https://danieltakeshi.github.io/2018/06/28/a2c-a3c/

        # accumulated discounted rewards
        discounted_rewards = np.zeros_like(rewards)

        # we use predicted value, otherwise if done, we use 0
        cumsum = 0 if done else self.critic.predict(np.float32(self.states[-1] / 255.))[0]

        # cumulate the discounted rewards backwardsbased on recurrence relationship
        for t in reversed(range(0, len(rewards))):
            cumsum = cumsum * self.discount_factor + rewards[t]
            discounted_rewards[t] = cumsum

        return discounted_rewards

    def train_step(self, done):
        ''' trains actor and critic '''

        # same shape as rewards (game_steps x 1)
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))

        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.critic.predict(states)

        # (game_steps x 1)
        values = np.reshape(values, len(values))

        # advantage function based on paper
        advantages = discounted_rewards - values

        # optimize
        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_rewards])

        # reset
        self.states, self.actions, self.rewards = [], [], []

    def build_localmodel(self, summary = False):
        ''' same as public model'''

        if not self.lstm:
            # actor and critic share the convolutional layers
            input = Input(shape=self.state_size)
            conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
            conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
            conv = Flatten()(conv)

            # fully connected layer
            fc = Dense(256, activation='relu')(conv)

            # policy: softmax output with one entry per action representing the probability of selecting the action
            policy = Dense(self.action_size, activation='softmax')(fc)

            # value: single linear output representing the value function
            value = Dense(1, activation='linear')(fc)

            # actor determines policy
            actor = Model(inputs=input, outputs=policy)

            # critic chooses value
            critic = Model(inputs=input, outputs=value)

            actor._make_predict_function()
            critic._make_predict_function()

            if summary:
                print("Actor model:")
                actor.summary()

                print("\nCritic model:")
                critic.summary()

        else: # adds lstm layer after fc layer

            input = Input(shape=(self.state_size)) # 4x64x64x3
            x = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'))(input)
            x = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
            x = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(x)
            x = TimeDistributed(Flatten())(x)

            x = LSTM(256, activation='tanh')(x)

            # actor
            policy = Dense(self.action_size, activation='softmax')(x)

            # critic
            value = Dense(1, activation='linear')(x)

            # actor determines policy
            actor = Model(inputs=input, outputs=policy)

            # critic chooses value
            critic = Model(inputs=input, outputs=value)

            actor._make_predict_function()
            critic._make_predict_function()

            if summary:
                print("Actor model:")
                actor.summary()

                print("\nCritic model:")
                critic.summary()

        return actor, critic

    def get_action(self, state):
        ''' chooses action based on epsilon-greedy policy '''

        # normalizes input from [0, 255] to [0, 1]
        state = np.float32(state / 255.)

        # get policy distribution
        policy = self.local_actor.predict(state)[0]

        # sample action based on probability
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]

        return action_index

    def update_localmodel(self):
        ''' updates local model with global model weights'''

        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def replay_memory(self, state, action, reward):
        ''' saves current sample as [s, a, r] for calculating discounted rewards '''
        self.states.append(state)

        # one hot
        act = np.zeros(self.action_size)
        act[action - 1] = 1

        self.actions.append(act)
        self.rewards.append(reward)
