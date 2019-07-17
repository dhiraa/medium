import tensorflow as tf
from imutils import paths # https://github.com/jrosebr1/imutils
import os
import cv2
from tqdm import tqdm
from stridednet import StridedNet
import matplotlib.pyplot as plt
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import numpy as np
# For “one-hot” encoding our class labels.
from sklearn.preprocessing import LabelBinarizer
# For splitting our data such that we have training and evaluation sets.
from sklearn.model_selection import train_test_split
# We’ll use this to print statistics from evaluation.
from sklearn.metrics import classification_report

import argparse


def run(args):
    # initialize the set of labels from the CALTECH-101 dataset we are
    # going to train our network on
    LABELS = set(["Faces", "Leopards", "Motorbikes", "airplanes"])

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args['dataset']))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in tqdm(imagePaths):
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # if the label of the current image is not part of of the labels
        # are interested in, then ignore the image
        if label not in LABELS: # Comment this if you wanted to consider the whole dataset
            continue

        # load the image and resize it to be a fixed 96x96 pixels,
        # ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (96, 96))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data into a NumPy array, then preprocess it by scaling
    # all pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    """

        [0, 0, 0, 1]  for “airplane”
        [0, 1, 0, 0]  for “Leopards”
        etc.

    """

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels,
                                                      test_size=0.25,
                                                      stratify=labels,
                                                      random_state=42)

    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / args['epochs'])

    model = StridedNet(width=96, height=96, depth=3,
                       classes=len(lb.classes_),
                       reg=tf.keras.regularizers.l2(0.0005),
                       init="he_normal")
    model = model.build()

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                                zoom_range=0.15,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.15,
                                                                horizontal_flip=True,
                                                                fill_mode="nearest")
    # train the network
    print("[INFO] training network for {} epochs...".format(
        args['epochs']))

    H = model.fit_generator(generator.flow(trainX, trainY, batch_size=32),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
                            epochs=args['epochs'])

    print(H.history)
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy
    N = args['epochs']
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args['plot'])


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-e", "--epochs", type=int, default=50,
                    help="# of epochs to train our network for")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    print(args)

    run(args)
