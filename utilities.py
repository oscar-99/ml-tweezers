import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras



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
    print(radii)
    radii = keras.utils.to_categorical(radii) # Convert to a one hot vector
    
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
    radii = radii[:,1]*1e6 # Change of units to microns
    
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