import numpy as np
from process import loadup
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Create a neural network
features = 6

def data_clean():
    """
    Function for importing and cleaning the data.
    """

    # Load forces and positions and shuffle data
    fx = loadup("data", "force")
    fx = fx[:, :3] *1e12 # pN

    x = loadup("data", "pos")
    x = x[:, :4]*1e6 # Microns um

    data = np.c_[fx, x]
    np.random.shuffle(data)

    # Split into training and testing set.
    data_points = data.shape[0]
    training_split = int(np.ceil(0.9*data_points))

    # Training data
    training_data = data[:training_split, :features]
    training_targets = data[:training_split, features:]

    # Testing data
    testing_data = data[training_split:, :features]
    testing_targets = data[training_split:, features:]

    return training_data, training_targets, testing_data, testing_targets


def modelmk1():
    """ Builder function for the mk1 model. """
    size = 64
    model = keras.Sequential([
    keras.layers.Dense(size, activation="relu", input_shape=(features,)),
    keras.layers.Dense(size, activation="relu"),
    keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    return model


def modelmk2():
    """ Builder function for the mk2 model """