import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

import keras

from utilities import loadup, store, ts_classify_data_prep
from network import modelmk2
from simulation import generate_data, simulate, force_comp

'''
training_data, training_labels, testing_data, testing_labels = ts_classify_data_prep(0.9, 2, sample_size=3000)

input_shape = training_data.shape[1:] # Time series length

model = modelmk2(input_shape, 5, "resnet3-fixed", epochs=30)
model.fit(training_data, training_labels, testing_data, testing_labels)
'''

hist_data = pd.read_csv('models/resnet3-fixedhistory.csv')
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
