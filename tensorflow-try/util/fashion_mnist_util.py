import matplotlib.pyplot as plt
import numpy as np
from imutils import build_montages
from sklearn.metrics import classification_report
import cv2

_history_plot = 'plot.png'

# initialize the label names
_labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]


def print_prediction(model, images, labels):
    # make predictions on the test set
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


def images_random_test(model, images, labels):
    # initialize our list of output images
    output_images = []

    # randomly select a few testing fashion items
    for i in np.random.choice(np.arange(0, len(labels)), size=(16,)):
        # classify the clothing
        image = images[np.newaxis, i]
        probs = model.predict(images[np.newaxis, i])
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
    montage = build_montages(output_images, (96, 96), (4, 4))[0]

    # show the output montage
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)


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
    montage = build_montages(output_images, (96, 96), (4, 4))[0]

    # show the output montage
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)
