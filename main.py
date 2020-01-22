import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

import keras

from utilities import loadup, store, ts_data_prep
from network import ResNetTS
from simulation import simulate, generate_data, force_comp
from diag import stat_values, position_data_plot, hist, hist_plot_regression, hist_plot_classify, regression_error_plot



# Parameters for generating data.
t = 1
simulations = 1000
sampling_rate = 10
radius_range = (0.6, 0.6)
n_range = (1.5,1.7)
classes = 0

# generate_data('cont_data_n', t, simulations, sampling_rate, radius_range, n_range, classes, only_forces=False, append=True)


# Plot some statistical analysis
position_data_plot('cont_data_n', radii=False)
stat_values('cont_data_n', 'force', 'n')
stat_values('cont_data_n', 'pos', 'n')

# Parameters for training.
train_test_ratio = 0.9
axes = [0, 2] # x and z axis
sample_size = 2000
epochs = 500

'''
# Process data.
training_data, training_labels, testing_data, testing_labels = ts_data_prep(train_test_ratio, axes, 'cont_data_n', sample_size, 'n', discrete=False)


# Build model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-nregression")
model.build_regression_output()
model.load_weights(location='models/resnet3-10classes.h5')
model.evaluate_regression(testing_data, testing_labels)
model.fit(training_data, training_labels, testing_data, testing_labels, epochs)
model.evaluate_regression(testing_data, testing_labels)


regression_error_plot(model, testing_data, testing_labels)



hist_plot_regression('resnet3-nregressionhistory')
'''