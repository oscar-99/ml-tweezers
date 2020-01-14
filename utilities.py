import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical




def ts_classify_data_prep(split, axes, data, sample_size):
    """
    Function for prepping univariate force data in a time series format to train a classifier. Loads up the force and radius data, performs z normalisation and converts to one hot labels and generate the training/testing split.

    Parameters:
        split (float): The proportion of train vs. test.
        axis (list[int]): 0 - x axis, 1 - y axis, 2 - z axis.
    """
    # Parameters of data
    ts_len = 1000 

    # Load up data
    f = loadup(data, "force")


    # Clean up and format radii and forces
    radii = f[:sample_size*ts_len,3]
    radii = np.reshape(radii, (sample_size, ts_len))
    radii = radii[:,1]*1e7 - 1 # Change of units to be integers from 0-4
    radii = to_categorical(radii) # Convert to a one hot vector
    
    forces = []

    # Process the force axes individually then stack them.
    for axis in axes:
        faxis = f[:sample_size*ts_len, axis]
        faxis = np.reshape(faxis, (sample_size, ts_len))

        # 'z' Normalise forces 
        fmean = np.mean(faxis, axis=1).reshape(sample_size, 1)
        fstd = np.std(faxis, axis=1).reshape(sample_size, 1)
        faxis = (faxis - fmean)/fstd
        forces.append(faxis)

    faxis = np.stack(forces, axis=2)

    split_index = int(np.ceil(split*sample_size))
    
    training_data = faxis[:split_index, :, :]
    testing_data = faxis[split_index:, :, :]

    training_labels = radii[:split_index, :]
    testing_labels = radii[split_index:, :]

    return training_data, training_labels, testing_data, testing_labels


def ts_regression_data_prep(split, axes, data, sample_size):
    """
    Function for prepping univariate force data in a time series format to train a regression. Loads up the force and radius data, performs z normalisation and generate the training/testing split

    Parameters:
        split (float): The proportion of train vs. test.
        axis (list[int]): 0 - x axis, 1 - y axis, 2 - z axis.
    """
    # Parameters of data
    ts_len = 1000 

    # Load up data
    f = loadup(data, "force")


    # Clean up and format radii and forces
    radii = f[:sample_size*ts_len,3]
    radii = np.reshape(radii, (sample_size, ts_len))
    print(radii)
    radii = radii[:,1]*1e6 # Change of units to microns
    print(radii)
    
    forces = []

    # Process the force axes individually then stack them.
    for axis in axes:
        faxis = f[:sample_size*ts_len, axis]
        faxis = np.reshape(faxis, (sample_size, ts_len))

        # 'z' Normalise forces 
        fmean = np.mean(faxis, axis=1).reshape(sample_size, 1)
        fstd = np.std(faxis, axis=1).reshape(sample_size, 1)
        faxis = (faxis - fmean)/fstd
        forces.append(faxis)

    faxis = np.stack(forces, axis=2)

    split_index = int(np.ceil(split*sample_size))
    
    training_data = faxis[:split_index, :, :]
    testing_data = faxis[split_index:, :, :]

    training_labels = radii[:split_index]
    testing_labels = radii[split_index:]

    return training_data, training_labels, testing_data, testing_labels


def store(filename, x):
    """
    Store simulation results. Takes a filename and an array to be stored.
    """
    save_location = "data/{}.npy".format(filename)
    np.save(save_location, x)


def loadup(filename, tag):
    """
    Loads the dataset with filename from the data folder and returns it as an array.

    Tag is "force" or "pos".
    """
    with h5py.File("data/{}.h5".format(filename), "r") as file:
        return np.array(file[tag])

def hist_plot_regression(file):
    """
    Plots the data stored in the history file a regression run.
    """
    hist_data = pd.read_csv('models/' + file + '.csv')
    plt.subplot(3, 1, 1)
    plt.title("Loss Statistics")
    plt.plot(hist_data["val_loss"])
    plt.plot(hist_data["loss"])
    plt.ylabel("Loss")

    plt.subplot(3, 1, 2)
    plt.title("Absolute Error Statistics")
    plt.plot(hist_data["val_mean_absolute_error"])
    plt.plot(hist_data["mean_absolute_error"])
    plt.ylabel("Mean Absolute Percentage Error")
    plt.legend(["Validation", "Training"] )

    plt.subplot(3, 1, 3)
    plt.title("Percentage Error Statistics")
    plt.plot(hist_data["val_mean_absolute_percentage_error"])
    plt.plot(hist_data["mean_absolute_percentage_error"])
    plt.ylabel("Mean Absolute Percentage Error")
    plt.xlabel("Epochs")
    plt.legend(["Validation", "Training"] )

    plt.show()


def hist_plot_classify(file):
    """
    Plots the data stored in the history file for classify run.
    """
    hist_data = pd.read_csv('models/' + file + '.csv')
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
