import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from scipy.constants import c

from utilities import loadup

sns.set()
# Diagnostics for the data e.g. histograms, time series plots and statistical properties.
def get_components(x):
    """
    Takes array resulting from simulations and returns three arrays corresonding to the xyz components.
    """
    xdata = x[:,0]
    ydata = x[:,1]
    zdata = x[:,2]

    return xdata, ydata, zdata


def position_data_plot(file, multiple=True, radii=True):
    """
    Plots the positions of particle(s) in the trap and compute some statistical properties of the data. 

    Parameters:
    -----------
    file : str
        Name of the file where the data is located.    
    multiple : bool
        If True plot the max, min and median on the same plot. If False plot only the median.
    radii : bool
        If True use radius for the measures of multiple. If False use n.
    """
    positions = loadup(file, 'pos')

    if radii:
        dep = loadup(file, 'radii')
    else:
        dep = loadup(file, 'n')

    # Generate indices selected.
    sorted_indices = np.argsort(dep[:,0])

    if multiple:
        simulations = [sorted_indices[-1], sorted_indices[len(dep)//2], sorted_indices[0]]
    else:
        simulations = [sorted_indices[0]]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    colours = ['r', 'g', 'k']
    label = []

    for k, index in enumerate(simulations):
        pos = positions[index,:,:]
        d = dep[index,0] 

        xdata, ydata, zdata = get_components(pos)
        # Generate Plots 
        ax1.plot3D(xdata, ydata, zdata, c=colours[k])

        if radii:
            label.append("Radius: {:.2f} \mu m".format(d*1e6))
        else:
            label.append("Refractive Index: {:.2f} \mu m".format(d))

        # Statistics
        print("Mean positions: ", np.mean(pos, axis=0))
        print("Standard deviation of positions: ", np.std(pos, axis=0))
        print("Max norm of positions: ", np.max(np.linalg.norm(pos, ord=2, axis=1)))


    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(label)

    plt.title("Position Plot")

    plt.show()


def stat_values(file, dependent, independent):
    """
    Plot mean standard deviation, max, median and min norms vs. radius and n.

    Parameters:
    -----------
    file : str
        Name of the file where the data is located.  
    dependent : str
        The dependent variable to plot 'force' or 'pos'
    independent : str
        The independent variable to plot, 'radii' or 'n'.
    """
    dep = loadup(file, dependent)
    ind = loadup(file, independent)

    if ind.size < 1000:
        num_points = ind.size
    else:
        num_points = 1000

    # Take the first num_points points and compute the stats

    sorted_indices = np.argsort(ind[:,0]) # Returns the indices that would sort ind

    # List for the radii and stats
    x = np.zeros(num_points)
    y = np.zeros((num_points, 5))

    for k, sorted_index in enumerate(sorted_indices):
        x[k] = ind[sorted_index]
        dep_row = dep[sorted_index,:,:]
        norms = np.linalg.norm(dep_row, ord=2, axis=1)

        y[k,:] = np.array([np.std(dep_row, axis=0)[0], np.std(dep_row, axis=0)[1], np.std(dep_row, axis=0)[2], np.max(norms), np.median(norms)])

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(x, y[:,0], 'r')
    ax1.plot(x, y[:,1], 'b')
    ax1.plot(x, y[:,2], 'k')
    
    ax2.plot(x, y[:,3])
    ax2.plot(x, y[:,4])

    ax1.legend(['std x', 'std y', 'std z'])
    ax2.legend(['max norm', 'median norm'])

    fig.suptitle("{} vs. {}".format(dependent, independent))

    plt.show()
    

def plot_time_series(x, y, f):
    """
    Plot some of the time series for each radius.
    """
    n_points = 1000
    dt = 1e-4
    t_tot = n_points*dt
    t = np.arange(0, t_tot, dt)

    for i in range(x*y):
        fz = f[i*n_points:int((i+1)*n_points), 2]*1e12
        fzmean = np.mean(fz)
        fzvar = np.var(fz)
        fznorm = (fz - fzmean )/np.sqrt(fzvar)
        radius = f[i*n_points + 1, 3]*1e6
        plt.subplot(y,x,i+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.plot(t, fznorm, "b", lw=0.2)
        # plt.plot(t, fx, "r", lw=0.2)
        plt.title("Force vs Time, Radius {:.3f} um".format(radius))
        plt.ylim([-3,3])
        plt.ylabel("Force (pN)")
        plt.xlabel("Time (s)")

    plt.show()


def stats_analysis(f):
    """
    Pull some statistics of the time series data.
    """

    n_points = 1000
    dt = 1e-4
    t_tot = n_points*dt
    t = np.arange(0, t_tot, dt)

    for i in range(10):
        fz = f[i*n_points:int((i+1)*n_points), 2]*1e12
        fzmean = np.mean(fz)
        fzvar = np.var(fz)
        radius = f[i*n_points + 1, 3]*1e6
        print("Radius: {:.3f} um".format(radius))
        print("Force Mean: {:.3f} pN".format(fzmean))
        print("Force Var: {:.3f} pN".format(fzvar))


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

    ax.set_xlim3d(-10e-8, 10e-8)
    ax.set_ylim3d(-10e-8, 10e-8)
    ax.set_zlim3d(0, 50e-8)

    plt.show()


def hist(file, index, axis):
    """
    Generates a histogram of the data x.
    """
    positions = loadup(file, 'pos')
    positions = positions[index,:,:]
    xdata, ydata, zdata = get_components(positions)

    if axis == "x":
        counts, bins = np.histogram(xdata, bins=100)
        plt.hist(xdata, bins = 50)
        plt.title("x Axis Force Histogram")
        plt.xlabel("Force (N)")

    if axis == "y":
        counts, bins = np.histogram(ydata, bins=100)
        plt.hist(ydata, bins = 50)
        plt.title("y Axis Force Histogram")
        plt.xlabel("Force (N)")

    if axis == "z":
        counts, bins = np.histogram(zdata, bins=100)
        plt.hist(zdata, bins = 50)
        plt.title("Axial Force Histogram")
        plt.xlabel("Force (N)")

    plt.show()
    return counts, bins


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
    plt.ylabel("Mean Absolute Error")
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


def regression_error_plot(model, testing_data, testing_labels):
    '''
    Generates a plot of the error vs. the test labels.

    Parameters:
    -----------
    model : keras.model
        A keras model to make predictions.
    testing_data : np.array
        A numpy array containing testing data.
    testing_labels : np.array
        A numpy array containing the labels for the testing data.
    '''


    y = model.predict(testing_data)
    y_true = testing_labels
    error = (y_true - y)
    error_pct = ( 100*(y_true - y)/y_true)


    plt.subplot(1, 2, 1)
    plt.scatter(testing_labels, error)
    plt.ylabel('Error')
    
    plt.subplot(1, 2, 2)
    plt.scatter(testing_labels, error_pct)
    plt.ylabel('Percentage Error')
    plt.xlabel('Test Label')

    plt.show()


def classify_error_plot():
    '''
    Generates a plot of the percentage correct vs. the test labels.

    Parameters:
    -----------
    model : keras.model
        A keras model to make predictions.
    testing_data : np.array
        A numpy array containing testing data.
    testing_labels : np.array
        A numpy array containing the labels for the testing data.
    '''
    pass