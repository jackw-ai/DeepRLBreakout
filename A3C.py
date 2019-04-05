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
from Agent import Agent

# global variables for A3C
global episode
episode = 0
MAX_EPISODES = 5000000

env_name = "BreakoutDeterministic-v4"

class A3C:
    ''' Asynchronous Advantage Actor Critic Model for Breakout game with optional lstm memory layer'''
    def __init__(self, action_size, load_model = False, lstm = False):

        # for saving and backups
        self.name = 'a3c'

        self.load_model = load_model

        # whether to add lstm layer
        self.lstm = lstm

        # actions are (left, stay, right)
        self.action_size = action_size

        # env settings (4 stacked frames of 84 x 84)
        self.state_size = (84, 84, 4)

        # balances between immediate and future rewards
        self.discount_factor = 0.99

        # max do-nothing actions at start of episode
        self.no_op_max = 30

        # learning rate
        self.actor_lr = self.critic_lr = 0.00025

        # number of individual asynchronous agents
        self.threads = 8

        # create model for actor and critic network
        self.actor, self.critic = self.build_actor_critic()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        ## tensorflow setup:

        # this is a thing only in tf...
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

         # tensorboard setup:
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        self.summary_writer = tf.summary.FileWriter('summary/' + self.name, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.load_model(self.name + ".h5")

    def train(self):
        ''' trains agents in multithreaded environment '''
        global episodes

        episode = 0

        with self.sess.as_default(), self.sess.graph.as_default():
            # instantiate self.threads agents
            agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic], self.lstm, self.sess, self.optimizer,
                            self.discount_factor, self.no_op_max, [self.summary_op, self.summary_placeholders,
                            self.update_ops, self.summary_writer]) for _ in range(self.threads)]

            # starts training agents
            for agent in agents:
                time.sleep(1)
                agent.start()

            # saves target model every 10 minutes
            while episode < MAX_EPISODES:
                time.sleep(60*10)
                self.save_model(name)

            self.save_model(name)

        self.sess.close()


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

    def actor_optimizer(self):
        ''' optimizer for actor '''
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01*entropy
        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        ''' optimizer for critic for value approximation '''
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self, name):
        ''' loads trained model '''
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        ''' saves trained model'''
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

    def setup_summary(self):
        ''' summary for tensorboard '''

        episode_total_reward = tf.Variable(0.0)
        episode_avg_max_q = tf.Variable(0.0)
        episode_duration = tf.Variable(0.0)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
