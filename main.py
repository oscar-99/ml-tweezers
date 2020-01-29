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
from diag import stat_values, position_plot, hist, history_plot_regression, history_plot_classify, regression_error_plot

sns.set()

# Parameters for generating data.
t = 1
simulations = 1
sampling_rate = 1
radius_range = (0.2,0.2)
n_range = (1.4, 1.4)
classes = 0

'''
generate_data('radii_test', t, simulations, sampling_rate, radius_range, n_range, classes, only_forces=False, append=False)

# Plot some statistical analysis
position_plot('radii_test', 'radii')
stat_values('radii_test', 'force', 'radii')
stat_values('radii_test', 'pos', 'radii')
'''

# Parameters for training.
train_test_ratio = 0.9
axes = [0, 1, 2] # x, y and z axis
sample_size = 5000
epochs = 85


# Process data.
# training_data, training_labels, testing_data, testing_labels = ts_data_prep(train_test_ratio, axes, 'cont-data-n-01-1', sample_size, 'n', discrete=False)


# Build and train model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-n-regression-xyz")
model.build_regression_output(1)
# model.fit(training_data, training_labels, testing_data, testing_labels, epochs)
model.evaluate_regression(testing_data, testing_labels)


# Diagnostics
history_plot_regression(model)
regression_error_plot(model, testing_data, testing_labels)
