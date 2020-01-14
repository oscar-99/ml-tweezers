import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

import keras
import tensorflow as tf 

from utilities import loadup, store, ts_classify_data_prep, ts_regression_data_prep, hist_plot_regression, hist_plot_classify
from network import ResNetTS
from simulation import generate_data, generate_cont_data, simulate, force_comp

# Parameters
train_test_ratio = 0.9
axes = [0, 2] # x and z axis
sample_size = 10000
epochs = 150
out_classes = 10


# Process data 
training_data, training_labels, testing_data, testing_labels = ts_regression_data_prep(train_test_ratio, axes, 'cont_data', sample_size) 


# Build model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-regression-10000")
model.build_regression_output()
model.load_weights(location='models/resnet3-10classes.h5')
model.evaluate_regression(testing_data, testing_labels)
model.fit(training_data, training_labels, testing_data, testing_labels, epochs)
model.evaluate_regression(testing_data, testing_labels)


hist_plot_regression('resnet3-regression-10000history')

