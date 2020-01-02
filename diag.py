import numpy as np
from process import loadup
import matplotlib.pyplot as plt

f = loadup("discrete_data","force")

def plot_time_series(x, y):
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


def stats_analysis():
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


# Histogram diagnosics

plot_time_series(6,3)
stats_analysis()