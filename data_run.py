# File will handle simulation of the data.

from simulation import generate_data

# Parameters for generating data.

t = 0.1
simulations = 5000
sampling_rate = 1
radius_range = (0.2,0.8)
n_range = (1.4, 1.7)
classes = 0

generate_data('cont-data-nr-01-1', t, simulations, sampling_rate, radius_range, n_range, classes, only_forces=False, append=True)