import matplotlib.pyplot as plt
import numpy as np
from imutils import build_montages
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.datasets import fashion_mnist
import cv2
import math

_history_plot = 'plot.png'

# initialize the label names
_labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

_random_test_count = 16


def print_prediction(model, images, labels):
    # make predictions on the fashion_model set
    preds = model.predict(images)

    # show a nicely formatted classification report
    print("[INFO] evaluating network...")
    print(classification_report(labels.argmax(axis=1), preds.argmax(axis=1),
                                target_names=_labelNames))


def plot_model_history(N, H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(_history_plot)


def get_row_col(count):
    row = math.floor(math.sqrt(count) / 2) * 2
    if row > 6:
        row = 6
    col = math.ceil(count / row)
    return row, col


def images_random_test(model, images, labels):
    # initialize our list of output images
    output_images = []

    # randomly select a few testing fashion items
    for i in np.random.choice(np.arange(0, len(labels)), size=(_random_test_count,)):
        # classify the clothing
        image = images[np.newaxis, i]
        probs = model.predict(image)
        prediction = probs.argmax(axis=1)
        label = _labelNames[prediction[0]]

        # # "channels_first" ordering
        # image = (images[i][0] * 255).astype("uint8")

        # "channels_last" ordering
        image = (images[i] * 255).astype("uint8")

        # initialize the text label color as green (correct)
        color = (0, 255, 0)

        # otherwise, the class label prediction is incorrect
        if prediction[0] != np.argmax(labels[i]):
            color = (0, 0, 255)

        # merge the channels into one image and resize the image from
        # 28x28 to 96x96 so we can better see it and then draw the
        # predicted label on the image
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2)

        # add the image to our list of output images
        output_images.append(image)

    # construct the montage for the images
    (row, col) = get_row_col(len(images))
    montage = build_montages(output_images, (96, 96), (col, row))[0]

    # show the output montage
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)


def predict_image_label(model, img):
    probs = model.predict(np.expand_dims(img, axis=0))
    prediction = np.argmax(probs)
    return prediction, probs[0]


def image_label_prob(model, img, label):
    return model.predict(np.expand_dims(img, axis=0))[0][label]


def show_images(images):
    # initialize our list of output images
    output_images = []

    # randomly select a few testing fashion items
    for i in np.arange(len(images)):
        # "channels_last" ordering
        image = (images[i] * 255).astype("uint8")

        # merge the channels into one image and resize the image from
        # 28x28 to 96x96 so we can better see it and then draw the
        # predicted label on the image
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

        # add the image to our list of output images
        output_images.append(image)

    # construct the montage for the images
    (row, col) = get_row_col(len(images))
    montage = build_montages(output_images, (96, 96), (col, row))[0]

    # show the output montage
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)


def create_test_data():
    ((train_images, train_labels), (test_images, test_labels)) = load_data()

    shape = tuple((_random_test_count, 28, 28, 1))
    images = np.zeros(shape, dtype=np.float)

    # randomly select a few testing fashion items
    idx = 0
    for i in np.random.choice(np.arange(0, len(train_images)), size=(_random_test_count,)):
        image = train_images[i]
        images[idx] += image
        idx += 1

    return images, shape


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
