from keras.models import load_model
from config.files import Paths

def load_state():
    return load_model(Paths.load_path.value)

def save_state(model):
    model.save(Paths.save_path.value)