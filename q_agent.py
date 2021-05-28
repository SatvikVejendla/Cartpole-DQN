import abc
from collections import deque
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from helpers.create import create_model
from helpers.pretrain import save_state, load_state

from config.train import Config
class Agent:
    def __init__(self, train=True):
        self.state_size = 4
        self.action_size = 2

        self.gamma = Config.gamma.value
        self.epsilon = Config.epsilon.value
        self.epsilon_decay = Config.decay_rate.value
        self.epsilon_min = Config.epsilon_min.value
        self.batch_size = Config.batch_size.value
        

        self.model = self.build_model(train)
        self.memory = deque(maxlen=Config.mem_max.value)

    def save(self):
        save_state(self.model)

    def build_model(self, train):
        if train:
            return create_model()
        else:
            return load_state()

    def get_action(self, state, do_train=True):
        if do_train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_memory(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save()