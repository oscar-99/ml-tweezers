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
radius = 1.0e-6
wavelength = 1064e-9
n_particle = 1.59
n_medium = 1.33

# Start location and Time Step
x0 = np.zeros((1,3))
v0 = np.zeros((1,3))
f0 = np.zeros((1,3))
dt = 0.5e-3
tfin = 10
times = np.arange(0,tfin,dt)


# Constants
kb = 1.38064852e-23 # Boltzmann
eta = 0.001 # Water Viscosity 
temp = 300 # Temp of Water (K)
gamma = 6*np.pi*eta*radius
nPc = 1.33*0.003/c

# Storage locations of models.
MODEL_FILE_3DOF = "ot-ml-supp-master/networks/3dof-position/net0/nn3dof_size_256.h5"
MODEL_FILE_5DOF = "ot-ml-supp-master/networks/5dof-position-size-ri/nn5dof_size_256.h5"
dof = 3


def force_method(x, net):
    """
    Returns the force by using prediction from neural net.

    Takes in standard array and converts to array in array used by keras.

    Output is array in array of force
    """
    if dof == 3:
        return net.predict(x)

    if dof == 5:
        x = np.array([np.append(x[0], [n_particle, radius])])
        return net.predict(x)/20
        

def simulation():
    """
    Simulates the motion of the particle, given an initial position.
    """
    # Load the model
    nn = load_model(MODEL_FILE_3DOF)
    # List of positions, velocities and forces
    x = [x0] 
    v = [v0]
    fx = [f0]
   
    for k in range(times.size):
        # Compute the force
        x1 = x[k]
        f = force_method(x1*1e6, nn)*nPc
        print(f)

        # Deterministic motion
        dx = dt*f/gamma

        # Brownian motion
        dx += np.sqrt(2*kb*temp*dt/gamma)*np.random.normal(0, 1, (1,3))
        x1 = np.add(x1[0], dx)

        # Update Position, Velocity and Force
        x.append(x1)
        v.append(dx/dt)
        fx.append(f)
    
    return (x, v, fx)


def store(filename, x):
    """
    Store simulation results. Takes a filename and an array to be stored.
    """
    save_location = "data/{}".format(filename)
    np.savetxt(save_location, x)

def loadup(filename):
    """
    Loads the storage file.
    """
    save_location = "data/{}".format(filename)
    return np.loadtxt(save_location)


#(x, v, fx) = simulation()

