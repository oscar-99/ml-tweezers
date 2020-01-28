import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical


def ts_data_prep(split, axes, file, sample_size, target_var, discrete=True):
    """
    Function for prepping univariate force data in a time series format to train a regression. Loads up the force and radius data, performs z normalisation and generate the training/testing split

    Parameters:
    -----------
    split : float
        The proportion of train vs. test.
    axes : list<int>
        0 - x axis, 1 - y axis, 2 - z axis.
    file : str
        The file where the data is located.
    sample_size : int
        The number of simulated time series to train on.
    target : str
        The variable that is changing and to be predicted, 'radii' or 'n'.
    discrete : bool
        If True data is discrete else continuous.
    """
    # Parameters of data
    ts_len = 1000 

    # Load up data
    f = loadup(file, "force")
    tar = loadup(file, target_var)


    # Clean up and format radii and forces
    if target_var == 'radii':
        tar *= 1e6 # Change of units to microns
    
    # If discrete data one hot encode.
    if discrete:
        tar = one_hot(tar)

    forces = []

    # Process the force axes individually then stack them.
    for axis in axes:
        faxis = f[:sample_size, :, axis]

        # 'z' Normalise forces 
        fmean = np.mean(faxis, axis=1).reshape(sample_size, 1)
        fstd = np.std(faxis, axis=1).reshape(sample_size, 1)
        faxis = (faxis - fmean)/fstd
        forces.append(faxis)

    faxis = np.stack(forces, axis=2)
    split_index = int(np.ceil(split*sample_size))
    
    training_data = faxis[:split_index, :, :]
    testing_data = faxis[split_index:, :, :]

    training_labels = tar[:split_index, :]
    testing_labels = tar[split_index:sample_size, :]

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

    Tag is "force", "pos", "radii" and "n".
    """
    with h5py.File("data/{}.h5".format(filename), "r") as file:
        return np.array(file[tag])


def one_hot(x):
    """
    Performs the one hot encoding of a vector.
    """

    class_values = {}

    for k, val in enumerate(x[:,0], start=0):
        if val in class_values.keys():
            class_values[val].append(k)
        else:
            class_values[val] = [k]

    order = np.sort(list(class_values.keys()))
    one_hot = np.zeros((x.size, len(class_values.keys()) )) 
    
    for j, val in enumerate(order):
        ind_list = class_values[val]
        for index in ind_list:
            one_hot[index, j] = 1
          

    return one_hot
    

def remove_from_file(file_loc, var, val_range):
    '''
    Removes values that fall outside of val_range for var.

    Parameters:
    -----------
    file_loc : str
        File location.
    var : str
        Variable 
    val_range : (float, float)
        Range of valid values of var.
    '''

    # Load up the files.
    only_forces = False
    n = loadup(file_loc, 'n')
    radii = loadup(file_loc, 'radii')
    forces = loadup(file_loc, 'force')

    # Check if positions have been stored before loading up.
    try:
        positions = loadup(file_loc, 'pos')
    except KeyError:
        only_forces = True

    # Find indices where values fall in range.
    if var == 'n':
        valid_indices = np.argwhere( np.logical_and(val_range[0] < n, n < val_range[1]) )
    else:
        valid_indices = np.argwhere( np.logical_and(val_range[0] < radii, radii < val_range[1]) )

    # Get the values where constraints hold.
    valid_indices = valid_indices[:,0]

    radii = radii[valid_indices]
    n = n[valid_indices]
    forces = forces[valid_indices, :, :]
    positions = positions[valid_indices, :, :]

    # Write new values over old.
    with h5py.File('data/' + file_loc + '.h5', "a") as file: 
        if not only_forces:
            file['pos'].resize(positions.shape[0], axis=0)
            file['pos'][:,:] = positions

        file['force'].resize(forces.shape[0], axis=0)
        file['force'][:,:,:] = forces

        file['radii'].resize(radii.shape[0], axis=0)
        file['radii'][:,:] = radii

        file['n'].resize(n.shape[0], axis=0)
        file['n'][:,:] = n


