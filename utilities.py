import numpy as np
from process import loadup
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


def data_prep(split, axis):
    """
    Load up the force data and generate the training/testing split
    """
    pass


def data_clean():
    """
    Function for importing and cleaning the data.
    """
    features = 6

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