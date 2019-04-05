import gym
import random
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

env_name = "BreakoutDeterministic-v4"

class A3CEvaluator:
    def __init__(self, action_size, lstm = False, file = 'a3c'):

        self.state_size = (84, 84, 4)
        self.action_size = action_size

        # balances between immediate and future rewards
        self.discount_factor = 0.99

        # whether to use lstm
        self.lstm = lstm

        self.file = file

        # max do-nothing actions at start of episode
        self.no_op_max = 30

        self.actor, self.critic = self.build_actor_critic()

    def build_actor_critic(self, summary = False):
        ''' builds actor critic network using architecture outlined in paper'''

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
        ''' samples action from sample distribution '''

        # normalizes input from [0, 255] to [0, 1]
        state = np.float32(state / 255.)

        # get policy distribution
        policy = self.actor.predict(state)[0]

        # get highest prob action (instead of sampling as in training)
        action_index = np.argmax(policy)

        return action_index

    def load_model(self, name):
        ''' loads trained model '''
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
        print("Actor Critic Models loaded")

    def play(self, games = 1, render = True):
        env = gym.make(env_name)
        agent.load_model(self.file)

        if render:
            env = gym.wrappers.Monitor(env, 'test', force=True)

        for game in range(games):
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
                step += 1

                if render:
                    env.render()

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

                # missed ball, subtract lives
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                # add to score
                score += reward

                # dead, we reset the history, since previous states don't matter anymore
                if dead:
                    state = np.stack((next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0]), axis=2)
                    state = np.reshape([state], (1, 84, 84, 4))
                    dead = False
                else:
                    state = next_state

                # if done, plot the score over episodes
                if done:
                    print("game:", game, "  score:", score, "  step:", step)
                    step = 0

        env.close()

        if render:
            show_video(training = False)

if __name__ == '__main__':
    ai = A3CEvaluator(action_size=3, lstm= False, file = 'a3c')
    ai.play()
