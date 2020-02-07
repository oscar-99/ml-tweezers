import numpy as np
import h5py
from keras.models import load_model
from scipy.constants import c
import random

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


def simulate(r, n, dt, t, net, verbose):
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
    verbose: bool
        Verbose output. 

    Returns
    -------
        Tuple of position and forces.
    """

    x = [x0]
    fx = [f0]
    nsteps = int(np.ceil(t/dt)) 
    completion_stages = [0.50, 1]

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
    if not os.path.exists(SAVE_LOC):
        create_dataset(SAVE_LOC, t_length)

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
        

        print("Beginning Simulation {}/{} For Radius: {:.3f} um, n: {:.3f} ".format(i+1, int(simulations), radius*1e6, n_part))
        x, fx = simulate(radius, n_part, dt, t, nn, True)

        x = x[:,0,:]
        x = x[101:, :] # Cut out first hundred points
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
            write_to_dataset(SAVE_LOC, forces, positions, radii, n_part_list)

            radii = []
            n_part_list = []
            positions = []
            forces = []

    # Save remaining lists.
    if len(forces) >= 1:
        write_to_dataset(SAVE_LOC, forces, positions, radii, n_part_list)

    print("All Simulations Complete")


def write_to_dataset(save_file, forces_list, positions_list, r_list, n_list):
    """
    Helper function for the generate data function which writes the values in the lists to the dataset in the correct format.
    """
    # Stack up into array format.
    if len(forces_list) > 1:
        forces = np.stack(forces_list)
        positions = np.stack(positions_list)
    else: 
        forces = np.array(forces_list)
        positions = np.array(positions_list)

    radii = np.array(r_list)
    radii = np.reshape(radii, (radii.shape[0], 1))
    ns = np.array(n_list)
    ns = np.reshape(ns, (ns.shape[0], 1))

    with h5py.File(save_file, 'a') as file:
        # Write to h5 file.
        file['pos'].resize(file['pos'].shape[0] + positions.shape[0], axis=0)
        file['pos'][-positions.shape[0]:] = positions

        file['force'].resize(file['force'].shape[0] + forces.shape[0], axis=0)
        file['force'][-forces.shape[0]:] = forces

        file['radii'].resize(file['radii'].shape[0] + radii.shape[0], axis=0)
        file['radii'][-radii.shape[0]:] = radii

        file['n'].resize(file['n'].shape[0] + ns.shape[0], axis=0)
        file['n'][-ns.shape[0]:] = ns


def create_dataset(save_loc, t_length):
    """ Function that creates a dataset. """
    with h5py.File(save_loc, "w") as file:    
        file.create_dataset("pos", shape=(0,t_length,3), maxshape=(None, t_length, 3))
        file.create_dataset("force", shape=(0,t_length,3),  maxshape=(None, t_length, 3))
        file.create_dataset("radii", shape=(0,1), maxshape=(None,1))
        file.create_dataset("n", shape=(0,1), maxshape=(None,1))


def generate_2d_data(file, t, simulations, sampling_rate, r_range, n_range,r_tiles, n_tiles, train_test_split,  verbose=True):
    """
    A function which improves upon the generate dataset function for the 2d case. The function will generate a n_tiles x r_tiles grid from which tiles will be selected at random and without replacement and used as the bounds of the uniform distribution. 

    The aim is to ensure a more even coverage down to a certain resolution.
    Should be able to handle n_tiles, r_tiles >=1. 

    Parameters:
    ----------
    file : str
        File name where generated data will be saved.
    t : float
        Time to run each simulation for.
    simulations : int
        Number of times to run the simulation. Should be a multiple of n_tiles*r_tiles
    sampling_rate : int
        The rate at which points are to be sampled from the simulated data. e.g with sampling rate of 10, 1 in 10 points will be saved.
    r_range : (float, float)
        Tuple of the range of radii in microns. 
    n_range : (float, float)
        Tuple of the range of the refractive index values.
    r_tiles : int
        The number of tiles for the range of radii.
    n_tiles : int
        The number of tiles for the range of refractive indices.
    train_test_split : float
        The training test split for the generated data.
    verbose : bool
        Verbose output.
    """
    # Generate vectors of boundaries.
    n = np.linspace(n_range[0], n_range[1], n_tiles+1)
    r = np.linspace(r_range[0], r_range[1], r_tiles+1)

    # Indices and counter.
    j = 0
    i = 0
    k = 0

    index_list = []
    # Generate indices of the box.
    if verbose:
        print('Generate refractive indices and radii.')

    while k < n_tiles*r_tiles:
        index_list.append((i,j))
        # If at end of row. 
        if i + 1 == (n_tiles):   
            i = 0
            j += 1 
        # Otherwise move along the row
        else:
            i +=1 
        k += 1


    # Now use index list to generate n and r.
    # Initialize list of n and r values and a counter
    n_r_list = []
    m = 0 
    while m < simulations:
        # Number of samples
        if m + len(index_list) > simulations:
            samples = simulations - m
        else:
            samples = len(index_list)

        # Sample from the index list to generate randomized indices.
        sampled_indices = random.sample(index_list, samples)
        m += samples

        # Loop through the randomized indices. 
        for i, j in sampled_indices:
            n_val = np.random.uniform(n[i], n[i+1])
            r_val = (1e-6)*np.random.uniform(r[j], r[j+1])
            n_r_list.append((n_val, r_val))

    # Shuffle the index and radius array.
    random.shuffle(n_r_list)
    
    if verbose:
        print('Refractive indices and radii generation complete.')

    if verbose:
        print("Beginning Simulation.")

    # Simulation parameters
    dt = 1e-4 # Simulation time step
    buffer = dt*100 # Generate 100 more points than needed so first 100 can be discarded.
    t += buffer

    t_length = int((t-buffer)/(dt*sampling_rate) ) # Number of points
    if verbose:
        print('Time Series Length: ', t_length)

    # Save locations
    MODEL_FILE_5DOF = "simulation_model/nn5dof_size_256.h5"
    nn = load_model(MODEL_FILE_5DOF)
    
    # Create a train and a test dataset.
    train_save = "data/" + file + "-train" + ".h5"
    test_save = "data/" + file + "-test" ".h5"

    # If the dataset does not exist, create it.
    if not os.path.exists(train_save):
        create_dataset(train_save, t_length)
        create_dataset(test_save, t_length)
        
    # Initialise value lists
    n_list = []
    r_list = []
    forces_list = []
    positions_list = []

    # Iterate across the radii and refractive indices
    k = 0 # Counter
    for n, r in n_r_list:
        if verbose:
            print("Simulation {}/{} For Radius: {:.3f} um, n: {:.3f}".format(k+1, simulations, r*1e6, n))

        # Run simulation. 
        x, fx = simulate(r, n, dt, t, nn, verbose)

        x = x[:,0,:]
        x = x[101:, :]
        x = x[::sampling_rate, :]

        fx = fx[:,0,:]
        fx = fx[101:,:]
        fx = fx[::sampling_rate, :] 

        positions_list.append(x)

        forces_list.append(fx)
        r_list.append(r)
        n_list.append(n)

        k+=1
        # If 100 Simulations have been run save results and obtain  
        if (i+1) % 100 == 0:
            print("Saving Progress")
            # index at which to split into training and testing.
            split_index = int(len(r_list)*train_test_split)

            # Write training and testing data.
            write_to_dataset(test_save, forces_list[:split_index], positions_list[:split_index], r_list[:split_index], n_list[:split_index])

            write_to_dataset(train_save, forces_list[split_index:], positions_list[split_index:], r_list[split_index:], n_list[split_index:])


            radii = []
            n_list = []
            positions = []
            forces_list = []

    # Save remaining lists.
    if len(forces_list) >= 1:
        # index at which to split into training and testing.
        split_index = int(len(r_list)*train_test_split)

        # Write training and testing data
        write_to_dataset(test_save, forces_list[:split_index], positions_list[:split_index], r_list[:split_index], n_list[:split_index])

        write_to_dataset(train_save, forces_list[split_index:], positions_list[split_index:], r_list[split_index:], n_list[split_index:])

    print("All Simulations Complete")
    





