from keras.utils import np_utils
from keras.datasets import fashion_mnist
# from imutils import build_montages
import numpy as np
# import cv2
import math

# initialize the label names
__labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

__random_test_count = 16


def get_row_col(count):
    """
    获取展示的行和列
    :param count: 图片总数
    :return: 行，列
    """
    row = math.floor(math.sqrt(count) / 2) * 2
    if row > 6:
        row = 6
    col = math.ceil(count / row)
    return row, col


def images_random_test(model, images, labels):
    """
    随机测试模型
    :param model: 模型
    :param images: 测试集
    :param labels: 测试标签集
    """
    # TODO
    print('[INFO] please import imutils and cv2')

    # # initialize our list of output images
    # output_images = []
    #
    # # randomly select a few testing fashion items
    # for i in np.random.choice(np.arange(0, len(labels)), size=(__random_test_count,)):
    #     # classify the clothing
    #     image = images[np.newaxis, i]
    #     probs = model.predict(image)
    #     prediction = probs.argmax(axis=1)
    #     label = __labelNames[prediction[0]]
    #
    #     # # "channels_first" ordering
    #     # image = (images[i][0] * 255).astype("uint8")
    #
    #     # "channels_last" ordering
    #     image = (images[i] * 255).astype("uint8")
    #
    #     # initialize the text label color as green (correct)
    #     color = (0, 255, 0)
    #
    #     # otherwise, the class label prediction is incorrect
    #     if prediction[0] != np.argmax(labels[i]):
    #         color = (0, 0, 255)
    #
    #     # merge the channels into one image and resize the image from
    #     # 28x28 to 96x96 so we can better see it and then draw the
    #     # predicted label on the image
    #     image = cv2.merge([image] * 3)
    #     image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    #     cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
    #                 color, 2)
    #
    #     # add the image to our list of output images
    #     output_images.append(image)
    #
    # # construct the montage for the images
    # (row, col) = get_row_col(len(images))
    # montage = build_montages(output_images, (96, 96), (col, row))[0]
    #
    # # show the output montage
    # cv2.imshow("Fashion MNIST", montage)
    # cv2.waitKey(0)


def predict_image_label(model, img):
    """
    预测图片标签
    :param model: 模型
    :param img: 图片
    :return: 预测标签，概率分布
    """
    probs = model.predict(np.expand_dims(img, axis=0))
    prediction = np.argmax(probs)
    return prediction, probs[0]


def image_label_prob(model, img, label):
    """
    获取特定标签的概率
    :param model: 模型
    :param img: 图片
    :param label: 标签
    :return:
    """
    return model.predict(np.expand_dims(img, axis=0))[0][label]


def show_images(images):
    """
    显示图片
    :param images: 图片
    """
    # TODO
    print('[INFO] please import imutils and cv2')

    # # initialize our list of output images
    # output_images = []
    #
    # # randomly select a few testing fashion items
    # for i in np.arange(len(images)):
    #     # "channels_last" ordering
    #     image = (images[i] * 255).astype("uint8")
    #
    #     # merge the channels into one image and resize the image from
    #     # 28x28 to 96x96 so we can better see it and then draw the
    #     # predicted label on the image
    #     image = cv2.merge([image] * 3)
    #     image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    #
    #     # add the image to our list of output images
    #     output_images.append(image)
    #
    # # construct the montage for the images
    # (row, col) = get_row_col(len(images))
    # montage = build_montages(output_images, (96, 96), (col, row))[0]
    #
    # # show the output montage
    # cv2.imshow("Fashion MNIST", montage)
    # cv2.waitKey(0)


def create_test_data():
    """
    创建测试数据
    :return: 测试数据
    """
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    shape = tuple((__random_test_count, 28, 28, 1))
    images = np.zeros(shape, dtype=np.float)

    # randomly select a few testing fashion items
    idx = 0
    for i in np.random.choice(np.arange(0, len(train_images)), size=(__random_test_count,)):
        image = train_images[i]
        images[idx] += image
        idx += 1

    return images, shape


def load_data():
    """
    加载数据集
    :return: 数据集
    """
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
