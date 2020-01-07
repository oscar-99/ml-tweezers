import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras




def ts_classify_data_prep(split, axis, sample_size=1000):
    """
    Function for prepping univariate force data in a time series format to train a classifier. Loads up the force and radius data, performs z normalisation and converts to one hot labels and generate the training/testing split

    Parameters:
        split (float): The proportion of train vs. test.
        axis (int): 0 - x axis, 1 - y axis, 2 - z axis.
    """
    # Parameters of data
    ts_len = 1000 

    # Load up data
    f = loadup("discrete_data", "force")

    # Clean up and format radii and forces
    radii = f[:sample_size*ts_len,3]
    radii = np.reshape(radii, (sample_size, ts_len))
    radii = radii[:,1]*1e7/2 - 1 # Change of units to be integers from 0-4
    radii = keras.utils.to_categorical(radii) # Convert to a one hot vector
    
    faxis = f[:sample_size*ts_len,axis]
    faxis = np.reshape(faxis, (sample_size, ts_len))

    # 'z' Normalise forces 
    fmean = np.mean(faxis, axis=1).reshape(sample_size, 1)
    fstd = np.std(faxis, axis=1).reshape(sample_size, 1)
    faxis = (faxis - fmean)/fstd


    # Split into training and testing sets and reshape data to 3d form expected by tensorflow (n, 1000, 1) i.e. single dimension multivariate time series.
    split_index = int(np.ceil(split*sample_size))
    
    training_data = faxis[:split_index, :]
    training_data = training_data.reshape(training_data.shape[0], training_data.shape[1], 1)

    testing_data = faxis[split_index:, :]
    testing_data = testing_data.reshape(testing_data.shape[0], testing_data.shape[1], 1)
    

    training_labels = radii[:split_index, :]
    testing_labels = radii[split_index:, :]

    return training_data, training_labels, testing_data, testing_labels


def data_clean():
    """
    Function for importing and cleaning the data.
    """
    features = 6

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