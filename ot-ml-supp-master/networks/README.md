## Networks
Pre-trained networks used in the paper.
These networks can be loaded into Matlab or Python.

For Matlab, the network can be imported using 
[importKerasNetwork](https://au.mathworks.com/help/deeplearning/ref/importkerasnetwork.html),
provided in the TensorFlow-Keras Models support package.
The following example evaluates 20 forces using a 3-DOF network
```matlab
net = importKerasNetwork('./path/to/network.h5');

% Evaluate the network at 20 points (assuming 3-dof network)
ndof = 3;
npts = 20;
x = rand(ndof, npts);
force = double(predict(net, reshape(x, [1, ndof, 1, npts]))).';
```

In Python, the Keras package can be used directly.
The following example evaluates 2 forces using a 5-DOF network
```python
from keras.models import load_model

# Load the model
net = load_model('./path/to/network.h5')

# Predict a forces at a point
p1 = [1, 2, 3, 4, 5]
p2 = [0, 0, 0, 0, 0]
f = net.predict([[p1, p2]])
```

The input parameter range and scaling for the calculated force
depend on the particular network.
The forces detailed bellow are in units of Q (dimensionless trapping
efficiency).  The conversion from Q to Newtons is: `Newtons = Q*index_medium*power/speed0`.

# 3dof-position
Networks trained to estimate the optical force on a particle in
a single circularly polarised Gaussian beam.
This directory contains 10 groups of networks trained on the same
data set with different numbers of hidden nodes.
By averaging the predictions of multiple networks, the estimated
optical force becomes more accurate.
If you only want a single network, use `3dof-position/net0/nn3dof_size_256.h5`.

Parameters:
  1. X Position (Units: microns, Gaussian distribution, centred at 0, FWHM: ~5)
  2. Y Position (Units: microns, Gaussian distribution, centred at 0, FWHM: ~5)
  3. Z Position (Units: microns, Gaussian distribution, centred at 0, FWHM: ~5)
  
Calculates force (Units: 1000Q)

# 4dof-position-alpha
Network trained to model experiment with 2 micron diameter spheres
trapped in 2-D against the microscope cover-slip in a linearly
polarised beam mixture (approximate NA of 1.2).
The network has 4 DOF: particle position and beam mixture ratio.
The network includes xyz position in a doughnut-shaped region around
the beam, values outside this region are probably incorrect.
The region is bound by radius 0.5 micron to 2 micron and
heights from 0 to 1 micron.
The particle has a refractive index of 1.59, and the background
refractive index is 1.33.

Parameters:
  1. X Position (Units: microns, see range above)
  2. Y Position (Units: microns, see range above)
  3. Z Position (Units: microns, see range above)
  4. Beam mixture ratio (Range: 0 to 1)
  
Calculates force (Units: 20Q)

# 5dof-position-size-ri
Networks trained to estimate the optical force on a spherical particle
with different size and refractive index at different positions in
a single circularly polarised Gaussian beam.
This folder contains 7 networks with different numbers of hidden nodes.
The network with 64 nodes per layer is used
by [Tweezers-ML](https://ilent2.github.io/tweezers-ml/).

Background medium has refractive index of 1.33.
May not work very well for low temperature simulations.
For low temperature simulations: retrain network with smaller
distribution of particle positions.

Parameters:
  1. X Position
  2. Y Position
  3. Z Position
  4. Radius (Units: microns, range: 0.1 to 1)
  5. Refractive index (range: 1.33 to 2)

Calculates force (Units: `(radius_particle_mu).^2*(index_particle-1.33)*Q/index_medium`).
