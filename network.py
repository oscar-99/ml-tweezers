import numpy as np
import pandas as pd
from utilities import loadup
import matplotlib.pyplot as plt
import os
import time

import tensorflow as tf
import keras


class ResNetTS():
    """ 
    Class for a ResNet time series classification and regression model for multiple force axes. Architecture based off of the paper: 'Deep learning for time series classification: a review'.
    """

    def __init__(self, input_shape, name, verbose=True, mini_batch_size=16):
        """
        Initialise the model. Body of the model is invariant to the time series length and the output layers and is initalised.
        |       |                    Body                 |        |
        | Input | ConvBlock | ConvBlock | ConvBlock | GAP | Output |

        Parameters:
            input_shape (tuple): (ts length, featurs) Size of the input expected.
            name (str): Name of the model used for storing weights.
        """
        self.name = name
        self.verbose = verbose

        self.input_shape = input_shape
        
        self.directory = os.path.join("models", self.name + ".h5")
        self.check_loc = os.path.join("models", self.name + "-checkpoint" + ".h5")
        self.hist_loc = os.path.join('models', self.name + '-history.csv')
        self.duration = None

        # Hyper parameters
        self.n_filters = 64
        self.mini_batch_size = mini_batch_size

        # Build the model body
        self.build_model_body()
             
        
    def conv_block(self, factor, input):
        """
        A single convolution block. factor is the integer multiple of n_filters for the conv layer.
        """
        conv_x = keras.layers.Conv1D(filters=factor*self.n_filters, kernel_size=8, padding="same")(input)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(filters=factor*self.n_filters, kernel_size=5, padding="same")(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(filters=factor*self.n_filters, kernel_size=3, padding="same")(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        return conv_z


    def build_model_body(self):
        """
        Method for building the body of the model. 
        """
        # Build model
        self.input = keras.layers.Input(shape=self.input_shape)

        # Block 1 
        b1 = self.conv_block(1, self.input)

        # Shortcut
        short_y = keras.layers.Conv1D(filters = self.n_filters, kernel_size=1, padding='same')(self.input)
        short_y = keras.layers.normalization.BatchNormalization()(short_y)
        
        out_b1 = keras.layers.add([short_y, b1])
        out_b1 = keras.layers.Activation("relu")(out_b1)


        # Block 2
        b2 = self.conv_block(2, out_b1)

        # Shortcut 
        short_y = keras.layers.Conv1D(filters=2*self.n_filters, kernel_size=1, padding="same")(out_b1)
        short_y = keras.layers.normalization.BatchNormalization()(short_y)

        out_b2 = keras.layers.add([short_y, b2])
        out_b2 = keras.layers.Activation("relu")(out_b2)


        # Block 3
        b3 = self.conv_block(2, out_b2)

        # Shortcut
        short_y = keras.layers.normalization.BatchNormalization()(out_b2)

        out_b3 = keras.layers.add([short_y, b3])
        out_b3 = keras.layers.Activation('relu')(out_b3)

        
        # Output layers
        self.gap_layer = keras.layers.GlobalAveragePooling1D()(out_b3)


    def build_classify_output(self, n_classes):
        """
        Build a classifier output layer.
        """
        out_layer = keras.layers.Dense(n_classes, name='classify', activation='softmax')(self.gap_layer)

        # Build model
        self.model = keras.models.Model(inputs=self.input, outputs=out_layer)

        reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5, patience=50, min_lr=0.0001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.directory, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_learning_rate, model_checkpoint]
        adam = keras.optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        if self.verbose:
            self.model.summary()

        # By default load up weights of previously fitted model.
        try: 
            self.load_weights()
            print("Previous Weights Loaded.")
            
        except FileNotFoundError:
            pass

    
    def build_regression_output(self, outs, weight_load=True):
        """
        Build a regression output layer.

        Parameters:
        -----------
        outs : int
            Number of outputs of the regression. 1 for just radius or refractive index. 2 for both.
        """
        # Naming this layer is important as weights are loaded by name (all layers except final are the same between the classify and regression models.)
        out_layer = keras.layers.Dense(outs, name='reg{}'.format(outs))(self.gap_layer)


        # Build model
        self.model = keras.models.Model(inputs=self.input, outputs=out_layer)

        # Build callbacks
        reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5, patience=50, min_lr=0.0001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.check_loc)
        model_history = keras.callbacks.CSVLogger(self.hist_loc, append=True)

        self.callbacks = [reduce_learning_rate, model_checkpoint, model_history]
        adam = keras.optimizers.Adam()


        self.model.compile(loss='mse', optimizer=adam, metrics=['mae', 'mape'])

        if self.verbose:
            self.model.summary()

        # By default load up weights of previously fitted model.
        if weight_load:
            try: 
                self.load_weights()
                print("Default Previous Weights Loaded.")       
            except OSError:
                print("Default Previous Weights Not Loaded (Not found).")


    def fit(self, x_train, y_train, x_val, y_val, epochs):
        """
        Trains the model.
        """
        self.n_epoch = epochs
        start_time = time.time()
        
        self.hist = self.model.fit(x_train,y_train, batch_size=self.mini_batch_size, epochs=self.n_epoch, validation_data=(x_val,y_val), callbacks=self.callbacks)

        self.duration = time.time() - start_time
        print("Model Fit Complete. {} epochs completed in {:.2f}".format(self.n_epoch, self.duration/60))
        self.save_weights()    
        

    def evaluate_classify(self, x_val, y_val):
        """
        Runs evaluation for classifier given a time series.
        """
        loss, acc = self.model.evaluate(x_val,  y_val)
        print("Model accuracy: {:5.2f}%, loss: {:5.2f}".format(100*acc, loss))


    def evaluate_regression(self, testing_data, testing_labels):
        """
        Runs evaluation for regression.
        """
        ev = self.model.evaluate(testing_data, testing_labels)
        message = "Model Stats:-- "
        for i, met in enumerate(self.model.metrics_names):
            message = message + "{}: {:.4f}, ".format(met, ev[i])

        print(message)


    def predict(self, testing_data):
        """
        Run prediction for a model.
        """
        return self.model.predict(testing_data)
    

    def load_weights(self, location=None):
        """
        Load in the weights stored at file. Full path necessary.
        """
        # by_name will load in weights only in the final layer
        if location == None:
            self.model.load_weights(self.directory, by_name=True)
        else:
            self.model.load_weights(location, by_name=True)
            print("Custom Weights Loaded")
        
    
    def load_full_model(self, file):
        """
        Load in a full model, full path necessary. 
        """
        self.model = keras.models.load_model(file)
        print("Full model loaded in.")


    def save_weights(self):
        """
        Saves the weights in output file
        """   
        self.model.save_weights(self.directory)


