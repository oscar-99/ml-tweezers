# File mainly for training a model and assessing model performance 
import logging
import os

# Shutup TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model
import seaborn as sns

import keras

from utilities import loadup, store, ts_data_prep, remove_from_file
from network import ResNetTS
from simulation import simulate, generate_data, force_comp
from diag import stat_values, position_plot, hist, history_plot_regression, history_plot_classify, regression_error_plot, data_distribution_plot, error_plot_2d



# Parameters for training.
train_test_ratio = 0.95
axes = [0, 1, 2] # x, y and z axis
sample_size = 10000
epochs = 100


# Process data.
training_data, training_labels, testing_data, testing_labels = ts_data_prep(train_test_ratio, axes, 'cont-data-nr-01-1', sample_size, ['n', 'r'], discrete=False)


# Build and train model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-nr-regression-xyz")
model.build_regression_output(2) # output is n, r
# model.fit(training_data, training_labels, testing_data, testing_labels, epochs)
# model.evaluate_regression(testing_data, testing_labels)
print(model.predict(testing_data[:1,:,:]))
print(testing_labels[0])

# Diagnostics
error_plot_2d(model, training_data, training_labels)
history_plot_regression(model)

