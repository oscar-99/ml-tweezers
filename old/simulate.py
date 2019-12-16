import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import c

# Fixes issue with matplotlib 3d plotting on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Uses trained network to dynamically simulate particle

# Particle properties 
radius = 1e-6 # metres
wavelength = 1064e-9
n_particle = 1.59
n_medium = 1.33


# Start location and Time Step
x0 = np.zeros((1,3))
v0 = np.zeros((1,3))
f0 = np.zeros((1,3))
dt = 1e-4
tfin = 1
times = np.arange(0,tfin,dt)


# Constants
kb = 1.38064852e-23 # Boltzmann
eta = 0.001 # Water Viscosity 
temp = 300 # Temp of Water (K)
power = 0.02 # Power of Laser
gamma = 6*np.pi*eta*radius
nPc = 1.33*power/c

# Storage locations of models.
MODEL_FILE_3DOF = "ot-ml-supp-master/networks/3dof-position/net9/nn3dof_size_256.h5"

# 5 dof model includes: Radius (Units: microns, range: 0.1 to 1)
# Refractive index (range: 1.33 to 2)
MODEL_FILE_5DOF = "ot-ml-supp-master/networks/5dof-position-size-ri/nn5dof_size_256.h5"


def force_method(x, net, dof):
    """
    Returns the force by using prediction from neural net.

    Input: array of positions (in microns), a neural network and degrees of freedom

    Output: array of forces in newtons
    """
    # 3 dof outputs in units 1000Q (Q is trapping efficiency).
    if dof == 3:
        f = net.predict(x)*nPc
        
    # 5 dof unit conversion (radius_particle_mu).^2*(index_particle-1.33)*Q/index_medium.
    if dof == 5:
        x = np.array([np.append(x[0], [radius*1e6, n_particle])])
        # Radius is in microns.
        f = net.predict(x)*((radius*1e6)**2*(n_particle-1.33))*0.02/3e8
    
    # Unit conversion from trap efficiency to force.
    return f


def simulation(dof):
    """
    Simulates the motion of the particle, given an initial position.
    """
    # Load the model
    if dof == 3:
        nn = load_model(MODEL_FILE_3DOF)

    if dof == 5:
        nn = load_model(MODEL_FILE_5DOF)
    # List of positions, velocities and forces.
    x = [x0] 
    v = [v0]
    fx = [f0]
   
    for k in range(times.size):
        # Compute the force.
        x1 = x[k]
        # Pass distance in microns.
        f = force_method(x1*1e6, nn, dof)

        # Deterministic motion
        dx = dt*f/gamma

        # Brownian motion
        dx += np.sqrt(2*kb*temp*dt/gamma)*np.random.normal(0, 1, (1,3))
        x1 = np.add(x1[0], dx)

        # Update Position, Velocity and Force
        x.append(x1)
        v.append(dx/dt)
        fx.append(f)
     
        if (k + 1) % 1000 == 0:
            print("{}/{} points computed".format((k+1), times.size))
    
    print("Simulation complete.")
    return (x, v, fx)


def store(filename, x):
    """
    Store simulation results. Takes a filename and an array to be stored.
    """
    save_location = "data/{}.npy".format(filename)
    np.save(save_location, x)


def loadup(filename):
    """
    Loads the storage file.
    """
    save_location = "data/{}.npy".format(filename)
    return np.load(save_location)


dof = 5
(x, v, fx) = simulation(dof)


store("5dof256x", x)
store("5dof256v", v)
store("5dof256fx", fx)

