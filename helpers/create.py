from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from config.train import Config
from config.model import Structure

def create_model():
    model = Sequential()
    model.add(Dense(Structure.h1.value, activation=Structure.a1.value, input_dim=Structure.i_dim.value))
    model.add(Dense(Structure.h2.value, activation=Structure.a2.value))
    model.add(Dense(Structure.o.value))
    model.compile(Adam(lr=Config.alpha.value), Structure.loss.value)
    return model
