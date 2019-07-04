# import the necessary packages TODO 移除 sklearn
import matplotlib
from test.pyimagesearch.minivggnet import MiniVGGNet
from util.fashion_mnist_util import plot_model_history, images_random_test, print_prediction, show_images
from util.model_util import save_model, load_model, save_shadow_arrays, load_shadow_arrays, save_prob
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import numpy as np

# TODO
from ssim import SSIM

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

# initialize the number of epochs to train for, base learning rate,
# and batch size
_NUM_EPOCHS = 25
_INIT_LR = 1e-2
_BS = 32


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

    # plot the training loss and accuracy
    plot_model_history(N=_NUM_EPOCHS, H=H)

    show_model_effect(model)


def test_model():
    """
    加载本地模型并测试
    """

    model = load_model()

    show_model_effect(model)


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
        print(model.predict(shadow[np.newaxis, i]).argsort())


def create_prob():
    """
    创建判定矩阵
    """
    ((train_images, train_labels), (test_images, test_labels)) = load_data()
    model = load_model()

    preds = model.predict(train_images)

    prob_array = np.zeros(tuple((10, 10)), dtype=np.int)
    for i in np.arange(len(preds)):
        label = train_labels[i].argmax()

        pred_label = preds[i].argmax()
        prob_array[label][pred_label] += 1

    save_prob(prob_array)


def load_data():
    # grab the Fashion MNIST dataset (if this is your first time running
    # this the dataset will be automatically downloaded)
    print("[INFO] loading Fashion MNIST...")
    ((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()

    # # "channels_first" ordering
    # train_images = train_images.reshape((train_images.shape[0], 1, 28, 28))
    # test_images = test_images.reshape((test_images.shape[0], 1, 28, 28))

    # "channels_last" ordering
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # one-hot encode the training and testing labels
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)


def train_model(train_images, train_labels, test_images, test_labels):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=_INIT_LR, momentum=0.9, decay=_INIT_LR / _NUM_EPOCHS)
    model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training model...")
    H = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=_BS, epochs=_NUM_EPOCHS)

    return model, H


def show_model_effect(model):
    # load data
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    # # make predictions on the test set
    print_prediction(model, images=test_images, labels=test_labels)

    # initialize our list of output images
    # images_random_test(model, images=test_images, labels=test_labels)
