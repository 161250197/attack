import matplotlib
from keras.optimizers import SGD
import numpy as np

from util.fashion_model.pyimagesearch.minivggnet import MiniVGGNet
from util.fashion_mnist_util import load_data, images_random_test
from util.model_util import save_model, load_model, save_shadow_arrays, load_shadow_arrays

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

# initialize the number of epochs to train for, base learning rate,
# and batch size
__NUM_EPOCHS = 25
__INIT_LR = 1e-2
__BS = 32


def init_model():
    """
    初始化
    训练并保存模型
    """
    # load data
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    # train the network
    ((model), (H)) = train_model(train_images, train_labels, test_images, test_labels)

    # save the model
    save_model(model)

    # initialize our list of output images
    images_random_test(model, images=test_images, labels=test_labels)


def test_model():
    """
    加载本地模型并测试
    """
    model = load_model()

    # load data
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    # initialize our list of output images
    images_random_test(model, images=test_images, labels=test_labels)


def create_shadow():
    """
    创建对抗阴影
    :return:
    """
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    shadow_array = np.zeros(tuple((10, 28, 28, 1)), dtype=np.float)
    img_count_array = np.zeros(10, dtype=np.int)

    for i in np.arange(len(train_labels)):
        label = train_labels[i].argmax()

        # 4 5 6 不准确
        if i < 100 or label < 4 or label > 6 :
            img_count_array[label] += 1
            shadow_array[label] += train_images[i]

    for i in np.arange(10):
        shadow_array[i] /= img_count_array[i]

    save_shadow_arrays(shadow_array)


def test_shadow():
    shadow = load_shadow_arrays()
    model = load_model()

    for i in np.arange(10):
        prob = model.predict(shadow[np.newaxis, i])
        pred = prob.argsort()
        print(pred)


def train_model(train_images, train_labels, test_images, test_labels):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=__INIT_LR, momentum=0.9, decay=__INIT_LR / __NUM_EPOCHS)
    model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training model...")
    H = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=__BS, epochs=__NUM_EPOCHS)

    return model, H
