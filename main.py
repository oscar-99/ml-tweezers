import numpy as np 
import keras

from utilities import loadup, store, ts_classify_data_prep
from network import modelmk2

training_data, training_labels, testing_data, testing_labels = ts_classify_data_prep(0.9, 2)

training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], 1))
testing_data = testing_data.reshape((testing_data.shape[0], testing_data.shape[1], 1))

input_shape = training_data.shape[1:]

model = modelmk2(input_shape, 5, "resnet3")

model.fit(training_data, training_labels, testing_data, testing_labels)