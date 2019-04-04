import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, Lambda, add
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 10000


class DQN:
    ''' main DQN class for training atari breakout ai model for various DQN architectures: vanilla, double, dueling '''
    
    def __init__(self, action_size = 3, double = False, dueling = False, load_model = False):
        
        self.load_model = load_model
        
        # actions are (left, stay, right)
        self.action_size = action_size

        # whether to use DDQN architecture
        self.double = double
        
        # whether to use Dueling architecture
        self.dueling = dueling
        
        # env settings (4 stacked frames of 84 x 84)
        self.state_size = (84, 84, 4)
        
        # epsilon (exploration rate)
        # ε annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.0
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        
        # training batch size
        self.batch_size = 32
        
        # random policy run for n frames before learning starts
        self.replay_start_size = 50000
        
        # balances between immediate and future rewards
        self.discount_factor = 0.99
        
        # replay memory queue
        self.memory = deque(maxlen=1000000)
        
        # max do-nothing actions at start of episode
        self.no_op_max = 30
        
        # number of steps for each Q-learning update
        self.update_freq = 4
        
        # RMSProp optimizer parameters
        self.lr = 0.00025
        self.grad_momentum = 0.95
        self.min_sq_grad = 0.01
        
        # build model
        self.model = self.build_model()
        
        ''' have target model for better stability '''
        # target model for improved stability
        self.target_model = self.build_model(summary = False)
        self.update_target_model()
        
        # updates target model every 10000 steps
        self.update_target_rate = 10000

        # optimizer with huber loss
        self.optimizer = self.optimizer()

        self.avg_q_max, self.avg_loss = 0, 0
        
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

    def build_model(self, summary = True):
        ''' deep convolutional q network based on deepmind papers'''
        
        if self.dueling: # dueling architecture
            
            # initial layers are the same
            input = Input(shape = self.state_size)
            cnnout = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
            cnnout = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnnout)
            cnnout = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(cnnout)
            out = Flatten()(cnnout)

            # splits into dueling: advantage and value
            ## reference: https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
            advantage = Dense(512, activation='relu')(out)
            advantage = Dense(self.action_size)(advantage)
           
            value = Dense(512, activation='relu')(out)
            value =  Dense(1)(value)
            
            # before aggregating, we subtract average advantage to acc elerate training
            advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                               output_shape=(self.action_size,))(advantage)
                
            value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                           output_shape=(self.action_size,))(value)

            # sums advantage and value to estimate q-value
            q_value = add([value, advantage])
            model = Model(inputs=input, outputs= q_value)
            
            if summary:
                model.summary()

            return model
            
        else: # DQN architecture based on DeepMind
            
            # input: state output: q-value
            model = Sequential()
            model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                             input_shape=self.state_size))
            model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
            model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(self.action_size))
            
            if summary:
                model.summary()
                
            return model

    def update_target_model(self):
        ''' 
        clones model to target model
        
        This improves model stability since each update also increases target Q
        as generating targets using older set of paramets adds delay
        making divergence less likely
        '''
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        ''' chooses action based on epsilon-greedy policy '''
        
        # normalizes input from [0, 255] to [0, 1]
        state = np.float32(state / 255.0)
        
        if np.random.rand() <= self.epsilon:
            # exploration: get random action
            return random.randrange(self.action_size)
        else:
            # exploitation: get current known best action
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, dead):
        ''' saves current sample as [s, a, r, s'] to replay memory '''
      
        # normalizes states from [0, 255] to [0, 1]
        state = np.float32(state / 255.0)
        next_state = np.float32(next_state / 255.0)
        
        self.memory.append((state, action, reward, next_state, dead))

    def optimizer(self):
        ''' 
        optimizer with huber loss: 
        MSE for error [-1, 1], MAE for errors outside this range  
        '''
        
        # action: (0, 1, 2)
        a = K.placeholder(shape=(None,), dtype='int32')
        
        # target Q-value
        y = K.placeholder(shape=(None,), dtype='float32')

        # prediction for Q-value for subsequent state
        py_x = self.model.output

        # get action's predicted q-value
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        
        # error is just Q - Q^hat
        error = K.abs(y - q_value)

        # quadratic for error in range [-1, 1], since abs, just [0, 1]
        quadratic_error = K.clip(error, 0.0, 1.0)
        
        # linear outside the range
        linear_error = error - quadratic_error
        
        # huber loss
        loss = K.mean(0.5 * K.square(quadratic_error) + linear_error)

        # learning rate, gradient momentum, min_squared gradient according to paper
        optimizer = RMSprop(lr=self.lr, rho = self.grad_momentum, epsilon=self.min_sq_grad)
        
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        
        # inputs: state, action, target q-value
        train = K.function([self.model.input, a, y], [loss], updates=updates)
        
        return train
    
    def train_replay(self):
        ''' 
        training step: picks random sample from replay memory to train on
        
        '''
        
        # prefill with random policy up to train_start
        if len(self.memory) < self.replay_start_size:
            return
        
        # decay epsilon linearly for first 1000000 frames
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # sample batch from memory
        mini_batch = random.sample(self.memory, self.batch_size)

        # dims (batch x image dim (84 x 84 x 4))
        state = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_state = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        
        # get action, reward, dead from mini_batch
        action = [mini_batch[i][1] for i in range(self.batch_size)]
        reward = [mini_batch[i][2] for i in range(self.batch_size)]
        dead = [mini_batch[i][3] for i in range(self.batch_size)]

        # get state
        for i in range(self.batch_size):          
            state[i] = mini_batch[i][0] 
            next_state[i] = mini_batch[i][3]

        # predict target q value
        target_value = self.target_model.predict(next_state)

        # need this for Double DQN
        if self.double:
            model_val = self.model.predict(next_state)
            
        target = np.zeros((self.batch_size,))
        
        for i in range(self.batch_size):
            if dead[i]: # died, reward always 0
                target[i] = reward[i]
            else:
                if self.double:
                    ''' key DDQN feature: let original model select action but update reward with target model '''
                    target[i] = reward[i] + self.discount_factor * target_value[i][np.argmax(model_val[i])]

                else: # update Q from target model
                    target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizer([state, action, target])
    
        self.avg_loss += loss[0]

    def train(self, total_episodes = 50000):
        ''' trains model '''
        
        # Deterministic-v4 skips 4 frames as in Deepmind paper
        env = gym.make('BreakoutDeterministic-v4')

        env = gym.wrappers.Monitor(env, 'training', force=True)

        scores, episodes, global_step = [], [], 0

        # each episode is one game
        for e in range(total_episodes):
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

            # start training
            while not done:
                global_step += 1
                step += 1

                # get action for the current state and go one step in environment
                action = self.get_action(state)

                # map (0, 1, 2) to (1, 2, 3) corresponding to gym actions
                action += 1

                # take step
                observe, reward, done, info = env.step(action)

                # process observation into state
                next_state = preprocess(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))

                # add previous 3 frames
                next_state = np.append(next_state, state[:, :, :, :3], axis=3)

                self.avg_q_max += np.amax(self.model.predict(np.float32(state / 255.))[0])

                # missed ball, subtract lives
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                # initial paper says to trim reward, probs don't need to with huber loss?
                #reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                self.replay_memory(state, action, reward, next_state, dead)

                # train every n steps
                if step % self.update_freq == 0:
                    self.train_replay()

                # DDQN: update the target model with model
                if global_step % self.update_target_rate == 0:
                    self.update_target_model()

                score += reward

                # if dead, we ignore previous states the history
                if dead:
                    dead = False
                else:
                    state = next_state

                # plot stats over episodes
                if done:
                    if global_step > self.replay_start_size:
                        stats = [score, self.avg_q_max / float(step), step,
                                 self.avg_loss / float(step)]

                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])})

                        summary_str = self.sess.run(self.summary_op)

                        self.summary_writer.add_summary(summary_str, e + 1)

                    print("episode:", e, "  score:", score, "  memory length:",
                          len(self.memory), "  epsilon:", self.epsilon,
                          "  global_step:", global_step, "  average q:",
                          self.avg_q_max / float(step), "  average loss:",
                          self.avg_loss / float(step))

                    # reset stats after every game
                    self.avg_q_max, self.avg_loss = 0, 0

            # we backup the model every 100 episodes        
            if e % 1 == 0:
                self.save_model(self.name + ".h5")

        # save fully trained model
        self.save_model(self.name + ".h5")
        
        # closes tf session and gym env
        self.sess.close()
        env.close()
        
    def save_model(self, name = "dqn.h5"):
        ''' saves model locally'''
        self.model.save_weights(name)
        
    def load_model(self, name = "dqn.h5"):
        ''' load model saved locally'''
        self.model.load_weights(name)

    def setup_summary(self):
        ''' summary operators for tensorboard '''
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
    
    @property
    def name(self):
        ''' model name '''
        name = ['dqn']
        
        if self.double:
            name += ['d']
        
        if self.dueling:
            name += ['dueling']
        
        return ''.join(name[::-1])

if __name__ == '__main__':
    model = DQN(action_size=3, double = True, dueling = True)
    model.train(total_episodes = EPISODES)
