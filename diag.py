import numpy as np
from process import loadup
import matplotlib.pyplot as plt

f = loadup("discrete_data","force")
x = loadup("discrete_data","pos")

def plot_time_series(x):
    """
    Plot some of the time series for each radius.
    """
    n_points = 10000
    t_tot = n_points*1e-4
    t = np.arange(0, t_tot, 1e-4)

    for i in range(10):
        fz = x[i*n_points:int((i+1)*n_points), 2]*1e12
        fx = x[i*n_points:int((i+1)*n_points), 0]*1e12
        radius = x[i*n_points + 1, 3]*1e6
        plt.subplot(2,5,i+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.2)
        plt.plot(t, fz, "b", lw=0.2)
        # plt.plot(t, fx, "r", lw=0.2)
        plt.title("Force vs Time, Radius {:.3f} um".format(radius))
        plt.ylabel("Force (pN)")
        plt.xlabel("Time (s)")

    plt.show()

# Histogram diagnosics

plot_time_series(f)