import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

import keras
import tensorflow as tf 

from utilities import loadup, store, ts_classify_data_prep
from network import ResNetTS
from simulation import generate_data, generate_cont_data, simulate, force_comp

# Parameters
train_test_ratio = 0.9
axes = [0, 2] # x and z axis
sample_size = 4000
out_classes = 10

'''
# Process data 
training_data, training_labels, testing_data, testing_labels = ts_classify_data_prep(train_test_ratio, axes, 'class10_data', sample_size) 


# Build model
input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, "resnet3-10classes")
model.build_classify_output(10)
model.load_weights(location='models/resnet3-10classes.h5')
model.evaluate_classify(testing_data, testing_labels)
model.fit(training_data, training_labels, testing_data, testing_labels, 20)
'''

hist_data = pd.read_csv('models/resnet3history.csv')
plt.subplot(2, 1, 1)
plt.title("Loss Statistics")
plt.plot(hist_data["val_loss"])
plt.plot(hist_data["loss"])


plt.subplot(2, 1, 2)
plt.title("Accuracy Statistics")
plt.plot(hist_data["val_acc"])
plt.plot(hist_data["acc"])
plt.xlabel("Epochs")
plt.legend(["Validation", "Training"] )

plt.show()
