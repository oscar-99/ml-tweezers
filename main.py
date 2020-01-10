import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

import keras
import tensorflow as tf 

from utilities import loadup, store, ts_classify_data_prep
from network import ResNetTS
from simulation import generate_data, simulate, force_comp


training_data, training_labels, testing_data, testing_labels = ts_classify_data_prep(1, [0, 2], 'discrete_data', sample_size=10000) 

input_shape = training_data.shape[1:] # Time series length
model = ResNetTS(input_shape, 5, "resnet3")
model.load_weights(training_data, training_labels)
# model.fit(training_data, training_labels, testing_data, testing_labels, 1000)


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