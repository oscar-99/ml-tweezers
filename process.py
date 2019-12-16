from scipy.io import loadmat
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import c


def store(filename, x):
    """
    Store simulation results. Takes a filename and an array to be stored.
    """
    save_location = "data/{}.npy".format(filename)
    np.save(save_location, x)


def loadup1(filename):
    """
    Loads up npy storage files.
    """
    save_location = "data/{}.npy".format(filename)
    return np.load(save_location)


def loadup2(tag):
    """
    Loads the dataset tagged with tag from the data.h5 file in the data folder and returns it as an array.
    """
    with h5py.File("data/data.h5", "r") as file:
        return np.array(file[tag])


def get_components(x):
    """
    Takes array resulting from simulations and returns three arrays corresonding to the xyz components.
    """
    xdata = x[:,0]
    ydata = x[:,1]
    zdata = x[:,2]

    return xdata, ydata, zdata


def plot_output(data):
    """ 
    Function that plots output of the simulation. 
    """

    xdata, ydata, zdata = get_components(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(xdata, ydata, zdata)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def hist(x, axis):
    """
    Generates a histogram of the data x.
    """

    xdata, ydata, zdata = get_components(x)

    if axis == "x":
        counts, bins = np.histogram(xdata, bins=100)
        plt.hist(xdata, bins = 100)
        plt.title("x Axis Force Histogram")
        plt.xlabel("Force (N)")

    if axis == "y":
        counts, bins = np.histogram(ydata, bins=100)
        plt.hist(ydata, bins = 100)
        plt.title("y Axis Force Histogram")
        plt.xlabel("Force (N)")

    if axis == "z":
        counts, bins = np.histogram(zdata, bins=100)
        plt.hist(zdata, bins = 100)
        plt.title("Axial Force Histogram")
        plt.xlabel("Force (N)")

    plt.show()
    return counts, bins

"""
x = loadup1("posdata")
fx = loadup1("forcedata")

plot_output(x)
hist(x, "x")
hist(x, "y")
hist(x, "z")
"""

x = loadup2("pos")
hist(x, "x")
print(x.shape)
fx = loadup2("force")
print(fx.shape)