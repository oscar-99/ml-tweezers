# Example 3-DOF dataset
This directory contains an example 3-DOF data set and a Jupyter
notebook for training a 3-DOF network.
A PDF copy of the Jupyter notebook is also supplied.
The 3-DOF data set, `input.mat` contains force and position information
for a spherical particle in a circularly polarised Gaussian beam.
The dataset contains 1,000,000 data points.
For similar pre-trained networks, see `../networks/3dof-position/`.

The network modelled a particle with refractive index 1.5 in a background
medium of 1.3.  The particle radius is equal to
1 wavelength in the background medium.
Illumination wavelength (in vacuum) is 1064 nm.
The T-matrix was calculated using for a spherical particle using the
`TmatrixMie` class in the [optical tweezers toolbox](https://github.com/ilent2/ott).
The beam is a circularly polarised Gaussian beam with numerical aperture
of 1.02.
