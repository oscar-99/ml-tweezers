# File to mainly handle visualizing and statistical analysis of the data.
import logging
import os

# Shutup TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model
import seaborn as sns

import keras

from utilities import loadup, store, ts_data_prep, remove_from_file
from network import ResNetTS
from simulation import simulate, generate_data, force_comp, generate_2d_data
from diag import stat_values, position_plot, hist, history_plot_regression, history_plot_classify, regression_error_plot, data_distribution_plot

sns.set()

# Statistical analysis of data
t = 0.1

sampling_rate = 1
radius_range = (0.4,0.6)
n_range = (1.5, 1.7)
r_tiles = 2
n_tiles = 2
resolution = 2
simulations = 1*r_tiles*n_tiles

n_edges = np.linspace(n_range[0], n_range[1], n_tiles+1)
r_edges = np.linspace(radius_range[0], radius_range[1], r_tiles+1)*1e-6

generate_2d_data('test', t, simulations, sampling_rate, radius_range, n_range, r_tiles, n_tiles)

data_distribution_plot('cont-data-nr-01-1', n_edges, r_edges)

data_distribution_plot('cont-data-nr-01-1', resolution*n_tiles, resolution*r_tiles)
 

# Plot some statistical analysis
# position_plot('radii_test', 'n', multiple=False)
# stat_values('radii_test', 'force', 'radii')
# stat_values('radii_test', 'pos', 'radii')

