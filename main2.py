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

from utilities import loadup, store, ts_data_prep, remove_from_file, ts_2d_data_prep
from network import ResNetTS
from simulation import simulate, generate_data, force_comp
from diag import stat_values, position_plot, hist, history_plot_regression, history_plot_classify, regression_error_plot, data_distribution_plot, error_plot_2d



# Parameters for training.
axes = [0, 1, 2] # x, y and z axis
epochs = 5

# Process data.
training_data, training_labels = ts_2d_data_prep('cont-data-nr-01-1-train', axes, 1000)
testing_data, testing_labels = ts_2d_data_prep('cont-data-nr-01-1-test', axes, 1000)
validation_testing_data, validation_testing_labels = ts_2d_data_prep('validation-test', axes, 'all')

# Build and train model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-nr-regression-xyz-4")
model.build_regression_output(2) # output is n, r
# model.fit(training_data, training_labels, testing_data, testing_labels, epochs)
model.evaluate_regression(testing_data, testing_labels)
model.evaluate_regression(validation_testing_data, validation_testing_labels)

tiles = 40
# Diagnostics
error_plot_2d(model, validation_testing_data, validation_testing_labels, tiles, tiles, 'Validation')
error_plot_2d(model, training_data, training_labels, tiles, tiles, "Training")
history_plot_regression(model)

