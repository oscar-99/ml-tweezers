from scipy.io import loadmat
import numpy as np 
from scipy.constants import c


def plot_output(data):
    """ 
    Function that plots output of the simulation. 
    """

    xdata = []
    ydata = []
    zdata = []

    for entry in data:
        xdata.append(entry[0,0])
        ydata.append(entry[0,1])
        zdata.append(entry[0,2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(xdata, ydata, zdata)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()