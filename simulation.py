import numpy as np
import h5py
from keras.models import load_model
from scipy.constants import c

# Runs simulation using the 5 dof neural network

import os
# Fixes issue with matplotlib 3d plotting on my mac
# os.environ['KMP_DUPLICATE_LIB_OK']='True

# Fixes problem with numpy dlls on windows (for my pc)
# os.environ['PATH']

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
x0 = np.array([[0, 0, 0]])
v0 = np.zeros((1,3))
f0 = np.zeros((1,3))


def simulate(r, n, dt, t, net):
    """
    Simulates motion of a particle given a neural network.
    
    Parameters:
    -----------
    rs : float
        Radius of particle in metres.
    n : float 
        Refractive index.
    dt : float
        Time step.
    t : float
        Total time to run simulation and a network. 
    net : keras.model
        A keras model which gives forces from positions.

    Returns
    -------
        Tuple of position and forces.
    """

    x = [x0]
    fx = [f0]
    nsteps = int(np.ceil(t/dt)) 
    completion_stages = [0.25, 0.50, 0.75, 1]

    for k in range(nsteps):
        x1 = x[k]

        # Compute deterministic forces.
        f = force_comp(x1, r, n, net)
        dx = f*dt/gamma

        # Brownian motion term
        dx += np.sqrt(2*kb*temperature*dt/gamma)*np.random.normal(0, 1, (1,3))
        x1 = np.add(x1, dx)

        # Store position forces
        x.append(x1)
        fx.append(f)

        completion = ((k+1)/nsteps)
        
        if completion >= completion_stages[0]:
            completion_stages.pop(0)
            print("Simulation {:.2%} complete".format(completion))

    print("Simulation Complete")
    x = np.array(x)
    fx = np.array(fx)

    return x, fx


def force_comp(x, r, n, net):
    """
    Computes deterministic forces on a particle given a neural network.

    Parameters:
    -----------
    x : array
        Array of positions in metres.
    r : float
        Radius of particle in metres.
    n : float 
        Refractive index. 
    net : keras model
        A 5 dof keras model

    Outputs: Deterministic Forces [N].
    """
    # Neural network takes x, y, z, radius, refractive index, in units of microns
    input = np.array([np.append(x[0]*1e6, [r*1e6, n])])

    # Outputs in units (radius_particle_mu)^2*(index_particle-1.33)*Q/index_medium so perform unit conversion
    f = net.predict(input)*((r*1e6)**2)*(n-1.33)*power/c
    
    return f


def generate_data(file, t, simulations, sampling_rate, radii_range, n_range, classes, append=True, only_forces=False):
    """
    Function to run the simulation code and save results using .h5 file.
    If append is True will append to the given file rather than write over.

    Parameters:
    ----------
    file : str
        File name where generated data will be saved.
    t : float
        Time to run each simulation for.
    simulations : int
        Number of times to run the simulation.
    sampling_rate : int
        The rate at which points are to be sampled from the simulated data. e.g with sampling rate of 10, 1 in 10 points will be saved.
    radii_range : (float, float)
        Tuple of the range of radii in microns. 
    n_range : (float, float)
        Tuple of the range of the refractive index values.
    classes : int
        If classes == 0 use a continuous distribution otherwise generate classes number of discrete values.
    append : bool, optional
        If true append the generated data to the file. 
    only_forces : bool, optional
        If true will only store force values.
    """
    # Simulation parameters
    dt = 1e-4 # Simulation time step
    buffer = dt*100 # Generate 100 more points than needed so first 100 can be discarded.
    t += buffer

    t_length = int((t-buffer)/(dt*sampling_rate) ) # Number of points
    print('Time Series Length: ', t_length)

    # Save locations
    MODEL_FILE_5DOF = "simulation_model/nn5dof_size_256.h5"
    nn = load_model(MODEL_FILE_5DOF)
    
    SAVE_LOC = "data/" + file + ".h5"


    # Initialise datasets
    if append == False:
        with h5py.File(SAVE_LOC, "w") as file:    
            if not only_forces:
                file.create_dataset("pos", shape=(0,t_length,3), maxshape=(None, t_length, 3))

            file.create_dataset("force", shape=(0,t_length,3),  maxshape=(None, t_length, 3))
            file.create_dataset("radii", shape=(0,1), maxshape=(None,1))
            file.create_dataset("n", shape=(0,1), maxshape=(None,1))

    # Initialise storage lists before running simulations.
    radii = []
    n_part_list = []
    positions = []
    forces = []

    for i in range(simulations):
        # Randomly generate the radii and refractive indices.
        if classes == 0:    
            radius = (1e-6)*np.random.uniform(radii_range[0],radii_range[1])
            n_part = np.random.uniform(n_range[0], n_range[1])
        else: 
            radius = 1e-6*(radii_range[0] + np.random.randint(0,classes)*(radii_range[1] - radii_range[0])/(classes-1))
            n_part = n_range[0] + np.random.randint(0, classes)*(n_range[1] - n_range[0])/(classes-1)
        

        print("Beginning Simulation {}/{} For Radius: {:.3f} um, n: {:.3f} ".format(i+1, simulations, radius*1e6, n_part))
        x, fx = simulate(radius, n_part, dt, t, nn)

        # Transform data into a n x 5 matrix
        x = x[:,0,:]
        x = x[101:, :]
        x = x[::sampling_rate, :]

        fx = fx[:,0,:]
        fx = fx[101:,:]
        fx = fx[::sampling_rate, :]


        if not only_forces:
            positions.append(x)

        forces.append(fx)
        radii.append(radius)
        n_part_list.append(n_part)

        # If 100 Simulations have been run save results and obtain  
        if (i+1) % 100 == 0:
            print("Saving Progress")
            forces = np.stack(forces)
            positions = np.stack(positions)
            radii = np.array(radii)
            radii = np.reshape(radii, (radii.shape[0], 1))
            n_part_list = np.array(n_part_list)
            n_part_list = np.reshape(n_part_list, (n_part_list.shape[0], 1))

            write_to_dataset(SAVE_LOC, forces, positions, radii, n_part_list, only_forces)

            radii = []
            n_part_list = []
            positions = []
            forces = []


    # Format lists into numpy arrays and store them
    if len(forces) > 1:
        forces = np.stack(forces)
        if not only_forces:
            positions = np.stack(positions)
        
    elif len(forces) == 1:
        forces = np.array(forces)
        if not only_forces:
            positions = np.array(positions)
        
    
    radii = np.array(radii)
    radii = np.reshape(radii, (radii.shape[0], 1))
    n_part_list = np.array(n_part_list)
    n_part_list = np.reshape(n_part_list, (n_part_list.shape[0], 1))

    if len(forces) >= 1:
        write_to_dataset(SAVE_LOC, forces, positions, radii, n_part_list, only_forces)

    print("All Simulations Complete")


def write_to_dataset(save_file, forces, positions, radii, n_part_list, only_forces):
    """
    Helper function for the generate data function.
    """

    with h5py.File(save_file, 'a') as file:
        if not only_forces:
            file['pos'].resize(file['pos'].shape[0] + positions.shape[0], axis=0)
            file['pos'][-positions.shape[0]:] = positions

        file['force'].resize(file['force'].shape[0] + forces.shape[0], axis=0)
        file['force'][-forces.shape[0]:] = forces

        file['radii'].resize(file['radii'].shape[0] + radii.shape[0], axis=0)
        file['radii'][-radii.shape[0]:] = radii

        file['n'].resize(file['n'].shape[0] + n_part_list.shape[0], axis=0)
        file['n'][-n_part_list.shape[0]:] = n_part_list





