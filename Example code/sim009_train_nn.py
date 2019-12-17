# Python script to train neural networks

import sys

if len(sys.argv) == 1:
    raise Exception('Usage: ./program.py sidx');
elif int(sys.argv[1]) > 6 or int(sys.argv[1]) < 0:
    raise Exception('Argument must be in range [0, 6]');

sidx = int(sys.argv[1])

from scipy.io import loadmat
import numpy as np
from scipy.constants import c

## Load data

fname = '/data/uqilento/Feb19-1-0/input.mat'

positions = np.transpose(loadmat(fname)['positions']) * 1e+6 # positions [um]
radius = np.transpose(loadmat(fname)['radius']) * 1e+6 # radius [um]
indices = np.transpose(loadmat(fname)['index_particle']) # particle refractive index

forces = np.transpose(loadmat(fname)['forces']) * c # forces [N * c]

## Process the data a bit

forces_divided_by_radius_2_and_index = np.array([force / R**2 / (index - 1.33) for force, R, index in zip(forces, radius, indices)])

## Split validation and training data and shuffle

number_train_samples = 9000000
number_val_samples = 1000000

feature_number = 5
tmp = np.concatenate((positions, radius, indices, forces_divided_by_radius_2_and_index), axis=1)
np.random.shuffle(tmp)

train_data = tmp[:number_train_samples, :feature_number]
train_targets = tmp[:number_train_samples, feature_number:]

val_data = tmp[number_train_samples:number_train_samples + number_val_samples, :feature_number]
val_targets = tmp[number_train_samples:number_train_samples + number_val_samples, feature_number:]

## Setup keras model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras import models
from keras import layers

size = [256, 128, 64, 32, 16, 8, 4];
#sidx = 0;  # 0-6

model = models.Sequential()
model.add(layers.Dense(size[sidx], activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(size[sidx], activation='relu'))
model.add(layers.Dense(size[sidx], activation='relu'))
model.add(layers.Dense(3))

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

model.summary()

## Train the model

mae = []
mape = []
val_mae = []
val_mape = []

for epochs, batch_size in zip([100, 100, 100, 100, 100],
                              [32, 128, 1024, 4096, 16384]):
    print(">>> ")
    print(">>> ", batch_size, " <<<")
    print(">>> ")

    history = model.fit(train_data,
                        train_targets,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(val_data, val_targets))

    mae.extend(history.history['mean_absolute_error'])
    mape.extend(history.history['mean_absolute_percentage_error'])
    val_mae.extend(history.history['val_mean_absolute_error'])
    val_mape.extend(history.history['val_mean_absolute_percentage_error'])

## Save the result

save_file_name = "nn5dof_size_{}.h5".format(size[sidx])
model.save(save_file_name)

import pickle

save_file_name_pkl = "nn5dof_size_{}.pkl".format(size[sidx])
with open(save_file_name_pkl, 'wb') as f:
    pickle.dump([mae, mape, val_mae, val_mape], f)

