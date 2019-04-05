from DQN import *

env_name = "BreakoutDeterministic-v4"

class DQNEvaluator:
    ''' testing AI to evaluate model '''
    def __init__(self, action_size = 3, dueling = False, file = 'dqn.h5'):

        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.no_op_steps = 20

        self.file = file

        # slightly different arcitecture for dueling network
        self.dueling = dueling

        self.model = self.build_model()

        # tensorflow
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.sess.run(tf.global_variables_initializer())

    def build_model(self, summary = False):
        ''' build the same model again '''

        if self.dueling:
            # initial layers are the same
            input = Input(shape = self.state_size)
            cnnout = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
            cnnout = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnnout)
            cnnout = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(cnnout)
            out = Flatten()(cnnout)

            # splits into dueling: advantage and value
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
        else:
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

    def get_action(self, history, epsilon = 0.0):
        ''' greedy actions only '''

        if np.random.random() < epsilon:
            return random.randrange(3)

        history = np.float32(history / 255.0)

        q_value = self.model.predict(history)

        return np.argmax(q_value[0])

    def load_model(self, filename):
        ''' load trained weights '''
        self.model.load_weights(filename)
        print("Model Loaded")

    def play(self, games = 1, render = True):
        ''' plays breakout games and renders them '''

        env = gym.make(env_name)

         # testing metric: average score across games
        self.avg_score = 0

        # file for model
        self.load_model(self.file)

        if render:
            env = gym.wrappers.Monitor(env, 'test', force=True)

        for e in range(games):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5
            observe = env.reset()

            for _ in range(random.randint(1, self.no_op_steps)):
                observe, _, _, _ = env.step(1)

            state = preprocess(observe)
            state = np.stack((state, state, state, state), axis=2)
            state = np.reshape([state], (1, 84, 84, 4))

            while not done:

                if render:
                    env.render()

                step += 1

                action = self.get_action(state)

                # convert from (0,1,2) to (1,2,3)
                action += 1

                observe, reward, done, info = env.step(action)

                next_state = preprocess(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_state = np.append(next_state, state[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward

                # if dead, we reset the history, since previous states don't matter anymore
                if dead:
                    state = np.stack((next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0], next_state[:, :, :, 0]), axis=2)
                    state = np.reshape([state], (1, 84, 84, 4))
                    dead = False
                else:
                    state = next_state

                if done:
                    self.avg_score += score
                    print("game:", e, "  score:", score)

            self.avg_score = self.avg_score / games

            if games > 1:
                print("average score across %d games was %f" %(games, self.avg_score))

        env.close()
        self.sess.close()

        if render:
            show_video()

if __name__ == '__main__':
    ai = DQNEvaluator(action_size=3, dueling = False, file = 'ddqn.h5')
    ai.play()
