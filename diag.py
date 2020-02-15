import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from scipy.constants import c
from scipy.stats import binned_statistic_2d

from utilities import loadup


# Diagnostics for the data e.g. histograms, time series plots and statistical properties.
def get_components(x):
    """
    Takes array resulting from simulations and returns three arrays corresonding to the xyz components.
    """
    xdata = x[:,0]
    ydata = x[:,1]
    zdata = x[:,2]

    return xdata, ydata, zdata


def position_plot(file, dep, multiple=True):
    """
    Plots the positions of particle(s) in the trap and compute some statistical properties of the data. 

    Parameters:
    -----------
    file : str
        Name of the file where the data is located.   
    radii : str
        If 'n' choose time series based on refractive index. If 'radii' use radius. 
    multiple : bool
        If True plot the max, min and median on the same plot. If False plot only the median.
    """
    positions = loadup(file, 'pos')

    radii = loadup(file, 'radii')
    n = loadup(file, 'n')

    if dep == 'radii':
        dep_var = radii 
        other_var = n
    if dep == 'n':
        dep_var = n
        other_var = radii
        

    # Generate sorted indices.
    sorted_indices = np.argsort(dep_var[:,0])
    if multiple:
        simulations = [sorted_indices[-1], sorted_indices[len(dep_var)//2], sorted_indices[0]]
    else:
        simulations = [sorted_indices[0]]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    colours = ['r', 'g', 'k']
    label = []

    for k, index in enumerate(simulations):
        pos = positions[index,:,:]
        d = dep_var[index,0] 

        xdata, ydata, zdata = get_components(pos)
        # Generate Plots 
        ax1.plot3D(xdata, ydata, zdata, c=colours[k])

        label.append("Radius: {:.2f} um n: {:.2f}".format(radii[index,0]*1e6, n[index, 0]))

        # Statistics
        print("Mean positions: ", np.mean(pos, axis=0))
        print("Standard deviation of positions: ", np.std(pos, axis=0))
        print("Correlation of x-y positions: ", np.correlate(pos[:, 0], pos[:, 1]))
        print("Max norm of positions: ", np.max(np.linalg.norm(pos, ord=2, axis=1)))


    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(label)

    plt.title("Position Plot, Varying {}".format(dep))

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
    sorted_indices = np.argsort(ind[:num_points,0]) # Returns the indices that would sort ind

    # List for the radii and stats
    x = np.zeros(num_points)
    norms  = np.zeros((num_points, 2))
    stddev = np.zeros((num_points, 3))
    corr = np.zeros((num_points, 3))


    for k, sorted_index in enumerate(sorted_indices):
        x[k] = ind[sorted_index]
        dep_row = dep[sorted_index,:,:]

        # Each simulation compute the norm of the position or force and store the median an max norm.
        norm = np.linalg.norm(dep_row, ord=2, axis=1)
        norms[k, :2] = np.array([np.max(norm), np.median(norm)])

        # Each simulation compute the standard deviation and store it.
        stddev[k,:3] = np.array([np.std(dep_row, axis=0)[0], np.std(dep_row, axis=0)[1], np.std(dep_row, axis=0)[2]])

        corr[k,:3] = np.array([np.correlate(dep_row[:, 0], dep_row[:, 1]),np.correlate(dep_row[:, 0], dep_row[:, 2]), 
        np.correlate(dep_row[:, 1], dep_row[:, 2])] )[:, 0]


    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(x, stddev[:,0], 'r')
    ax1.plot(x, stddev[:,1], 'b')
    ax1.plot(x, stddev[:,2], 'k')
    
    ax2.plot(x, norms[:,0])
    ax2.plot(x, norms[:,1])

    ax3.plot(x, corr[:, 0])
    ax3.plot(x, corr[:, 1])
    ax3.plot(x, corr[:, 2])

    ax1.legend(['Std x', 'Std y', 'Std z'])
    ax2.legend(['Max norm', 'Median norm'])
    ax3.legend(['Correlation x-y', 'Correlation x-z', 'Correlation y-z'])

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


def history_plot_regression(model):
    """
    Plots the data stored in the history file a regression run.
    """
    hist_data = pd.read_csv("models/" + model.name +"-history.csv")
    plt.subplot(3, 1, 1)
    plt.yscale("log")
    plt.title("Loss Statistics")
    plt.plot(hist_data["val_loss"])
    plt.plot(hist_data["loss"])
    plt.ylabel("Loss")
    plt.legend(["Validation", "Training"] )
    
    plt.subplot(3, 1, 2)
    plt.title("Absolute Error Statistics")
    plt.yscale("log")
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


def history_plot_classify(file):
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
    plt.xlabel('Test Label')
    
    plt.subplot(1, 2, 2)
    plt.scatter(testing_labels, error_pct)
    plt.ylabel('Percentage Error')
    plt.xlabel('Test Label')

    plt.show()


def classify_error_plot(model, testing_data, testing_label):
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

    results = {}
    for k in range(testing_label.size):
        result = np.argmax(model.predict(testing_data[k, :, : ]))
        true_val = np.argmax(testing_label)
        if result == true_val:
            correct = 1
        else:
            correct = 0
        
        if true_val in results.keys():
            results[true_val].append(correct)
        else:
            results[true_val] = [correct]

    labels = []
    props = []
    for label, correct_list in results.items(): 
        labels.append(label)
        props.append(sum(correct_list/len(correct_list)))       
    
    plt.bar(np.arange(len(labels)), props)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("Proportion Correct")
    plt.title("Test Prediction Breakdown")


def error_plot_2d(model, data, labels, x_tiles, y_tiles, data_type):
    '''
    A function to visualize the error for 2d regression.

    Parameters:
    -----------
    model : ResNetTS
        A keras model that outputs. 
    data : np.array
        Some data for the model.
    labels : np.array
        The labels for the data.
    x_tiles : int
        The tiles in the x direction in the plot.
    y_tiles : int
        The tiles in the y direction in the plot.
    data_type : str
        The dataset type i.e. train, test etc. For the title.
    '''
    y = model.predict(data)
    dy = y - labels
    n_error = dy[:, 0]
    r_error = dy[:, 1]

    n = labels[:, 0]
    r = labels[:, 1]

    n_error_pct = 100*n_error / n 
    r_error_pct = 100*r_error / r
    
    n_bins = binned_statistic_2d(n, r, n_error_pct, statistic='mean', bins=[x_tiles, y_tiles])

    r_bins = binned_statistic_2d(n, r, r_error_pct, statistic='mean', bins=[x_tiles, y_tiles])

    x_labels = ['{:.2f}'.format(x) for x in n_bins.x_edge]
    y_labels = ['{:.2f}'.format(y) for y in n_bins.y_edge]

    bounding_box = [float(x_labels[0]) , float(x_labels[-1]), float(y_labels[0]), float(y_labels[-1])]

    plt.figure()
    plt.suptitle("{} Dataset Error".format(data_type))
    plt.subplot(1, 2, 1)
    plt.title("Percentage Error in Refractive Index.")
    plt.imshow(n_bins.statistic, cmap='viridis', origin='lower', extent=bounding_box)
    plt.xlabel('Refractive Index')
    plt.ylabel("Radius")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.title("Percentage Error in Radius.")
    plt.imshow(r_bins.statistic, cmap='plasma', origin='lower', extent=bounding_box)
    plt.xlabel('Refractive Index')
    plt.ylabel("Radius")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()


def data_distribution_plot(file, n_tiles, r_tiles):
    '''
    Plot the distribution of the simulated data.

    Parameters:
    -----------
    file : str
        The filename of the data.
    n_tiles : int
        The number of tiles for n values.
    r_tiles : int
        The number of tiles for the r values.
    '''
    n = loadup(file, 'n')
    r = loadup(file, 'radii')

    # Univariate data
    if np.all(np.isclose(r, r[0, 0])):
        plt.hist(n, bins=20)
        plt.xlabel("Refractive Index")

    elif np.all(np.isclose(n, n[0,0])):
        plt.hist(r, bins=20)
        plt.xlabel("Radius (metres)")

    else:
        # hist2d expects (m,) shape
        plt.hist2d(n[:,0], r[:,0], bins=[n_tiles, r_tiles], cmap='plasma')
        plt.xlabel("Refractive Index")
        plt.ylabel("Radius (metres)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Count')

    plt.title("Plot of Data Distribution")
    print("Total Point Count: {}".format(n.size))
    plt.show()
    

    