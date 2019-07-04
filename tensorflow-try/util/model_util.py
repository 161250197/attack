from tensorflow import keras
import numpy as np

_model = 'model.h5'
_shadow_array = 'shadow.txt'
_prob_array = 'prob.txt'


def load_model():
    print("[INFO] loading Model...")
    return keras.models.load_model(_model)


def save_model(model):
    print("[INFO] saving Model...")
    model.save(_model, include_optimizer=True)


def load_shadow_arrays():
    print("[INFO] loading Shadow...")
    return np.loadtxt(_shadow_array).reshape((10, 28, 28, 1))


def save_shadow_arrays(arr):
    print("[INFO] saving Shadow...")
    np.savetxt(_shadow_array, arr.flatten())


def load_prob():
    print("[INFO] loading Prob...")
    return np.loadtxt(_prob_array).reshape((10, 10))


def save_prob(arr):
    print("[INFO] saving Prob...")
    np.savetxt(_prob_array, arr.flatten())