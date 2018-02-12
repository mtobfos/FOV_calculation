import functions as fov
import numpy as np
import os

cwd = os.getcwd()

def load_config():
    config = np.load(cwd + '/configuration.npy')
    return config

def load_data(config):
    positions, data = fov.read_data(config)
    print('Data loaded, shape:', data.shape)
    return positions, data