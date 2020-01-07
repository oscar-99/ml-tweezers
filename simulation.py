import numpy as np
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import c

# Runs simulation using the 5 dof neural network

import os
# Fixes issue with matplotlib 3d plotting on my mac
# os.environ['KMP_DUPLICATE_LIB_OK']='True

# Fixes problem with numpy dlls on windows (for my pc)
os.environ['PATH']

# Particle Properties
n_part = 1.59 # range: 1.33 - 2.0
radius = 1.0e-6 # [m] range: 0.1 to 1 microns


# Setup Properties
wavelength = 1064e-9 # [m]
power = 0.02 #[W]
kb = 1.38064852e-23
eta = 0.001 #[Ns/m^2]
gamma = 6*np.pi*eta*radius # Drag for sphere
temperature = 300 # [K]


# Initial conditions
# Initial z pos different?
x0 = np.array([[0, 0, 0]])
v0 = np.zeros((1,3))
f0 = np.zeros((1,3))


# Simulation parameters
dt = 1e-4
tfin = .11 # Generate 100 more points than needed so first 100 can be discarded.


def simulate(radius, n, net):
    """
    Simulates motion of a particle given a neural network.
    
    Inputs: position, radius of particle, refractive index and a network. All distances in units of metres.

    Outputs: Tuple of position, velocity and forces.
    """

    x = [x0]
    fx = [f0]
    nsteps = int(np.ceil(tfin/dt)) 

    for k in range(nsteps):
        x1 = x[k]

        # Compute deterministic forces.
        f = force_comp(x1, radius, n, net)
        dx = f*dt/gamma

        # Brownian motion term
        dx += np.sqrt(2*kb*temperature*dt/gamma)*np.random.normal(0, 1, (1,3))
        x1 = np.add(x1, dx)

        # Store position, velocity and forces
        x.append(x1)
        # v.append(dx/dt)
        fx.append(f)

        if (k + 1) % 1000 == 0:
            print("{}/{} points computed".format((k+1), nsteps))

    print("Simulation Complete")
    return x, fx


def force_comp(x, radius, n, net):
    """
    Computes deterministic forces on a particle given a neural network.

    Inputs: Position, radius of particle, refractive index and a network. All distances in units of metres.

    Outputs: Deterministic Forces [N].
    """
    # Neural network takes x, y, z, radius, refractive index, in units of microns
    input = np.array([np.append(x[0]*1e6, [radius*1e6, n])])

    # Outputs in units (radius_particle_mu)^2*(index_particle-1.33)*Q/index_medium so perform unit conversion
    f = net.predict(input)*((radius*1e6)**2)*(n-1.33)*power/c
    
    return f


def store(filename, x):
    """
    Store simulation results. Takes a filename and an array to be stored.
    """
    save_location = "data/{}.npy".format(filename)
    np.save(save_location, x)


def generate_data(append=False, simulations=1000):
    """
    Function to run the simulation code and save results using .h5 file.
    If append is True will append to the given file rather than write over.
    """
    MODEL_FILE_5DOF = "ot-ml-supp-master/networks/5dof-position-size-ri/nn5dof_size_256.h5"
    nn = load_model(MODEL_FILE_5DOF)
    
    SAVE_LOC = "data/discrete_data.h5"

    # Initialise datasets
    if append == False:
        with h5py.File(SAVE_LOC, "w") as file:    
            file.create_dataset("pos", shape=(0,5), maxshape=(None,5))
            file.create_dataset("force", shape=(0,5),  maxshape=(None,5))

    for i in range(simulations):
        # Run a simulation for radius
        radius = 2*np.random.randint(1,6)*1e-7
        print("Beginning Simulation {}/{} For Radius: {}m".format(i+1, simulations, radius))
        x, fx = simulate(radius, n_part, nn)

        # Transform data into a n x 5 matrix
        x = np.array(x)
        x = x[:,0,:]
        x = x[101:, :]
        y = np.c_[x, radius*np.ones((x.shape[0],1)), n_part*np.ones((x.shape[0], 1))]

        fx = np.array(fx)
        fx = fx[:,0,:]
        fx = fx[101:,:]
        fy = np.c_[fx, radius*np.ones((fx.shape[0],1)), n_part*np.ones((fx.shape[0], 1))]

        with h5py.File(SAVE_LOC, "a") as file:
            # Store results
            file["pos"].resize((file["pos"].shape[0] + y.shape[0], 5))
            file["pos"][-y.shape[0]:] = y

            file["force"].resize((file["force"].shape[0] + fy.shape[0], 5))
            file["force"][-fy.shape[0]:] = fy

    print("All Simulations Complete")