from DQN import *

class GameAI:
    ''' testing AI to evaluate model '''
    def __init__(self, action_size = 3, file = 'breakout_model.h5'):
        
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.no_op_steps = 20

        self.model = self.build_model()
        
        # file for model
        self.file = file
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, summary = True):
        ''' build the same model again '''
        
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
        ''' plays games and can render them '''
        self.load_model(self.file)
        
        env = gym.make('BreakoutDeterministic-v4')
        
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
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                
                if render:
                    env.render()
                
                step += 1

                action = self.get_action(history)

                # convert from (0,1,2) to (1,2,3)
                action += 1

                if dead:
                    action = 1
                    dead = False

                observe, reward, done, info = env.step(action)

                next_state = preprocess(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward

                history = next_history

                if done:
                    print("episode:", e, "  score:", score)

            env.close()
            self.sess.close()
            
            if render:
                show_video()


if __name__ == '__main__':
    ai = GameAI(action_size=3)
    ai.play()
