import numpy as np
from process import loadup
import matplotlib.pyplot as plt
import os
import time

import tensorflow as tf
from tensorflow import keras


def modelmk1():
    """ Builder function for the mk1 model. """
    size = 64
    features = 6
    model = keras.Sequential([
    keras.layers.Dense(size, activation="relu", input_shape=(features,)),
    keras.layers.Dense(size, activation="relu"),
    keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    return model


class modelmk2():
    """ 
    Class for the mk2 model a ResNet time series classification model for a single force axis.
    """

    def __init__(self, input_shape, n_classes, name, verbose=True):
        """
        Initialise the mk2 model.

        Parameters:
            input_shape (int): Size of the input expected.
            n_classes (int): The number of output classes.
            name (str): Name of the model used for storing weights.
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.name = name
        self.directory = os.path.join("models", self.name + ".h5")
        self.duration = None

        self.n_filters = 64

        self.model = self.build_model()
         
        if verbose:
            self.model.summary()

    
    def conv_block(self, factor, input):
        """
        A single convolution block. factor is the integer multiple of n_filters for the conv layer.
        """
        conv_x = keras.layers.Conv1D(filters=factor*self.n_filters, kernel_size=8, padding="same")(input)
        conv_x = keras.layers.normalisation.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(filters=factor*self.n_filters, kernel=5, padding="same")(conv_x)
        conv_y = keras.layers.normalisation.BatchNormalization(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(filters=factor*self.n_filters, kernel_size=3, padding="same")(conv_y)
        conv_z = keras.layers.normalisation.BatchNormalization()(conv_z)

        return conv_z


    def build_model(self):
        """
        Method for building the model. Architecture based off of the paper: 'Deep learning for time series classification: a review'.
        """
        # Build model
        input_layer = keras.layers.Input(self.input_shape)

        # Block 1 
        b1 = self.conv_block(1, input_layer)

        # Shortcut
        short_y = keras.layers.Conv1D(filters = self.n_filters, kernel_size=1, padding='same')(input_layer)
        short_y = keras.layers.normalisation.BatchNormalization()(short_y)
        
        out_b1 = keras.layers.add([short_y, b1])
        out_b1 = keras.layers.Activation("relu")(out_b1)


        # Block 2
        b2 = self.conv_block(2, out_b1)

        # Shortcut 
        short_y = keras.layers.Conv1D(filters=2*self.n_filters, kernel_size=1, padding="same")(short_y)
        short_y = keras.layers.normalisation.BatchNormalization()(short_y)

        out_b2 = keras.layers.add([short_y, b2])
        out_b2 = keras.layers.Activation("relu")(out_b2)


        # Block 3
        b3 = self.conv_block(2, out_b2)

        # Shortcut
        short_y = keras.layers.normalisation.BatchNormalization()(out_b2)

        out_b3 = keras.layers.add([short_y, b3])
        out_b3 = keras.layers.Activation('relu')(out_b3)

        
        # Output layer.
        gap_layer = keras.layers.GlobalAveragePooling1D()(out_b3)
        out_layer = keras.layers.Dense(self.n_classes, activation='softmax')(gap_layer)


        # Build model
        model = keras.models.Model(inputs=input_layer, outputs=out_layer)
        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='catergorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5, patience=50, min_lr=0.0001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.directory, monitor='loss', save_best_only = True)

        self.callbacks = [reduce_learning_rate, model_checkpoint]
        
        return model


    def fit(self, x_train, y_train, x_val, y_val, y_true):
        """
        Trains the model.
        """
        mini_batch_size = int(x_train.shape[0]/10)
        n_epoch = 1500

        start_time = time.time()
        
        hist = self.model.fit(x_train,y_train, batch_size=mini_batch_size, epochs=n_epoch, validation_data=(x_val,y_val), callbacks=self.callbacks)

        self.duration = time.time() - start_time
        

    def predict(self):
        """
        Runs prediction given a time series.
        """
        pass


    def save_weights(self):
        """
        Saves the weights in output file
        """   
        self.model.save_weights(self.directory)


model = modelmk2(1000, 10, "resnet3")