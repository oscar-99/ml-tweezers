import numpy as np
from process import loadup
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Create a neural network
features = 6

def data_clean():
    """
    Function for importing and cleaning the data.
    """

    # Load forces and positions and shuffle data
    fx = loadup("data", "force")
    fx = fx[:, :3] *1e12 # pN

    x = loadup("data", "pos")
    x = x[:, :4]*1e6 # Microns um

    data = np.c_[fx, x]
    np.random.shuffle(data)

    # Split into training and testing set.
    data_points = data.shape[0]
    training_split = int(np.ceil(0.9*data_points))

    # Training data
    training_data = data[:training_split, :features]
    training_targets = data[:training_split, features:]

    # Testing data
    testing_data = data[training_split:, :features]
    testing_targets = data[training_split:, features:]

    return training_data, training_targets, testing_data, testing_targets


def build_model():
    """ Builder function for keras model. """
    size = 64
    model = keras.Sequential([
    keras.layers.Dense(size, activation="relu", input_shape=(features,)),
    keras.layers.Dense(size, activation="relu"),
    keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    return model


training_data, training_targets, testing_data, testing_targets = data_clean()


model = build_model()
model.summary()

# Add in early exit?


EPOCHS = 10
model.fit(training_data, training_targets, epochs=EPOCHS, validation_data=(testing_data, testing_targets))


model.save("models/baseE10.h5")
test_predict = model.predict(testing_data)
print(test_predict)
plt.scatter(testing_targets, test_predict)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()