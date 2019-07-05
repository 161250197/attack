from tensorflow import keras
import numpy as np

__model = 'model.h5'
__shadow_array = 'shadow.txt'


def load_model():
    """
    加载本地模型
    :return: 模型
    """
    print("[INFO] loading Model...")
    return keras.models.load_model(__model)


def save_model(model):
    """
    保存本地模型
    :param model: 模型
    """
    print("[INFO] saving Model...")
    model.save(__model, include_optimizer=True)


def load_shadow_arrays():
    """
    加载本地遮罩
    :return: 遮罩
    """
    print("[INFO] loading Shadow...")
    return np.loadtxt(__shadow_array).reshape((10, 28, 28, 1))


def save_shadow_arrays(arr):
    """
    保存本地遮罩
    :param arr: 遮罩
    """
    print("[INFO] saving Shadow...")
    np.savetxt(__shadow_array, arr.flatten())
