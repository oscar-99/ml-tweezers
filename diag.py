import numpy as np
from utilities import loadup
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import c

# Diagnostics for the data e.g. histograms, time series plots and statistical properties.]\


def fourier(axis, series):
    """
    Generate the fourier transform for the given force axis.
    
    Parameters:
        axis (int): 0 - x axis, 1 - y axis, 2 - z axis.
    """
    sample_size = 10000
    ts_len = 1000
    f = loadup('discrete_data', "force")
    faxis = f[:sample_size*ts_len, axis]
    print(faxis)
    faxis = np.reshape(faxis, (sample_size, ts_len))
    print(faxis)
    faxis = faxis[series, :]

    fourier = np.fft.fft(faxis)
    plt.plot(fourier)
    plt.show()

fourier(2, 0)



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

    ax.set_xlim3d(-10e-8, 10e-8)
    ax.set_ylim3d(-10e-8, 10e-8)
    ax.set_zlim3d(0, 50e-8)

    plt.show()


def hist(x, axis):
    """
    Generates a histogram of the data x.
    """

    xdata, ydata, zdata = get_components(x)

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
